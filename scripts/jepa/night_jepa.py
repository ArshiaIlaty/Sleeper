#!/usr/bin/env python3
"""Level-2 hierarchical night-JEPA (Challenge 2026 CI, pure exploration).

Attacks night DYNAMICS -- a different axis than the 8 per-epoch JEPA NO-SHIPs, all
of which pooled per-epoch reps into a night vector and found the SSL rep redundant
with (or worse than) the handcrafted-EEG champion under LOSO/n_pos=83.

Pipeline:
  L1  per-epoch descriptor : REUSE the V2 EEG encoder's CLS output -- the one rep
      with real label-free signal (jepa-only AUROC 0.629). `dumpep` encodes every
      packed epoch -> l1_epemb (N,128) f16, aligned to packed_index.
  L2  night sequence       : pool BLK_EPOCHS=10 epochs (5 min) -> block descriptor +
      majority stage -> a per-night SEQUENCE of block tokens. A small night-transformer
      + temporal I-JEPA: mask a contiguous span of blocks, context encoder sees the rest
      (+CLS), EMA teacher provides targets, a narrow cross-attn predictor fills the
      masked block latents from mask-tokens + block-pos/stage embeddings, smooth-L1 on
      the EMA-teacher LN block reps. Label-free single all-nights pretrain (leak-free;
      CV enforced in the gate head).
  emb night embedding + Exp-C stage-conditioned prediction-error:
      - night CLS + stage-aware mean-pool of teacher block reps (glob + W/N1/N2/N3/REM)
      - Exp-C: mask each block one-at-a-time, predict it from the rest -> per-block
        surprise; aggregate glob {mean,max,cv} + per-stage mean (5) + cross-stage
        modulation. Hypothesis: CI = blunted / disorganized stage-conditioned surprise.

Gate: jepa_gate_ab.py --cv site (LOSO, comparable to all 8 prior) and --cv patient
(LOPO-style StratifiedGroupKFold by pid, tighter CIs).
"""
import argparse
import math
import time

import numpy as np

BLK_EPOCHS = 10          # 10 x 30s = one 5-min block
MAX_BLK = 200            # cap night length (~16.7 h of blocks); longer truncated
N_STAGE = 5
STAGE_TAGS = ["wake", "n1", "n2", "n3", "rem"]


def _import_torch():
    import torch
    import torch.nn as nn
    return torch, nn


# ===================================================== L1: per-epoch descriptors
def dumpep(prefix, ckpt, out, *, batch=512, device="cuda", progress=None):
    """Encode every packed epoch with the frozen V2 EEG teacher -> (N,d) f16."""
    import torch
    from raw_eeg_jepa_v2 import build_epoch_encoder, load_index
    dat, _ = load_index(prefix)
    state = torch.load(ckpt, weights_only=False)
    cfg = state["cfg"]
    enc = build_epoch_encoder(cfg["d_model"], cfg["depth"], cfg["heads"], dropout=0.0)
    enc.load_state_dict(state["teacher"])
    enc.to(device).eval()
    d = cfg["d_model"]
    N = dat.shape[0]
    ep = np.empty((N, d), dtype=np.float16)
    with torch.no_grad():
        for s in range(0, N, batch):
            e = min(N, s + batch)
            xb = np.ascontiguousarray(dat[s:e]).astype(np.float32)
            x = torch.from_numpy(xb).to(device)
            with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
                z = enc.cls_embed(x)
            ep[s:e] = z.float().cpu().numpy().astype(np.float16)
            if progress and (s // batch) % 80 == 0:
                progress(f"  encoded {e}/{N} epochs")
    np.save(out, ep)
    if progress:
        progress(f"saved L1 epemb {ep.shape} -> {out}")


# ===================================================== L2: night block sequences
def build_blocks(prefix, epemb_path, out, *, progress=None):
    """Pool per-epoch descriptors into 5-min blocks -> padded per-night sequences."""
    from raw_eeg_jepa_v2 import load_index
    _, idx = load_index(prefix)
    ep = np.load(epemb_path, mmap_mode="r")                    # (N,d) f16
    d = ep.shape[1]
    offsets = idx["offsets"]
    pool = idx["epoch_pool"]
    nb, ns, nl = idx["night_bids"], idx["night_site"], idx["night_label"]
    n_night = len(nb)
    blocks = np.zeros((n_night, MAX_BLK, d), np.float16)
    bstage = np.full((n_night, MAX_BLK), -1, np.int8)
    bvalid = np.zeros((n_night, MAX_BLK), bool)
    nblk = np.zeros(n_night, np.int32)
    for ni in range(n_night):
        a, b = int(offsets[ni]), int(offsets[ni + 1])
        if b <= a:
            continue
        e = np.asarray(ep[a:b], np.float32)                    # (n_ep,d)
        p = np.asarray(pool[a:b], np.int64)                    # 0..4 or <0 (unknown)
        n_ep = e.shape[0]
        k = min(MAX_BLK, math.ceil(n_ep / BLK_EPOCHS))
        for j in range(k):
            s0 = j * BLK_EPOCHS
            s1 = min(n_ep, s0 + BLK_EPOCHS)
            if s1 <= s0:
                break
            blocks[ni, j] = e[s0:s1].mean(0).astype(np.float16)
            seg = p[s0:s1]
            seg = seg[(seg >= 0) & (seg < N_STAGE)]
            bstage[ni, j] = int(np.bincount(seg, minlength=N_STAGE).argmax()) if seg.size else -1
            bvalid[ni, j] = True
        nblk[ni] = int(bvalid[ni].sum())
        if progress and ni % 200 == 0:
            progress(f"  night {ni}/{n_night}")
    np.savez_compressed(out, blocks=blocks, bstage=bstage, bvalid=bvalid,
                        nblk=nblk, bids=nb, sites=ns, labels=nl.astype(int))
    if progress:
        v = nblk[nblk > 0]
        progress(f"saved L2 blocks {blocks.shape} -> {out}; nights={ (nblk>0).sum() } "
                 f"median_nblk={int(np.median(v))} min={int(v.min())} max={int(v.max())}")


# ===================================================== models
def build_night_encoder(d_in=128, d=128, depth=4, heads=4, dropout=0.1):
    torch, nn = _import_torch()

    class NightEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.d = d
            self.proj = nn.Linear(d_in, d)
            self.stage_emb = nn.Embedding(N_STAGE + 1, d)      # idx 0 = unknown(-1)
            pe = torch.zeros(MAX_BLK, d)
            pos = torch.arange(MAX_BLK).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe)
            self.cls = nn.Parameter(torch.zeros(1, 1, d))
            nn.init.trunc_normal_(self.cls, std=0.02)
            layer = nn.TransformerEncoderLayer(d, heads, d * 4, dropout,
                                               activation="gelu", batch_first=True,
                                               norm_first=True)
            self.enc = nn.TransformerEncoder(layer, depth)
            self.norm = nn.LayerNorm(d)

        def tokens(self, blocks, bstage):
            B, M, _ = blocks.shape
            st = (bstage + 1).clamp(0, N_STAGE)                # -1 -> 0
            return self.proj(blocks) + self.pe[:M].unsqueeze(0) + self.stage_emb(st)

        def encode(self, blocks, bstage, vis):
            """vis (B,M) bool: attendable (context) block tokens. CLS always visible."""
            B, M, _ = blocks.shape
            t = self.tokens(blocks, bstage)
            cls = self.cls.expand(B, -1, -1)
            h = torch.cat([cls, t], dim=1)                     # (B,1+M,d)
            pad_cls = torch.zeros(B, 1, dtype=torch.bool, device=vis.device)
            kpm = torch.cat([pad_cls, ~vis], dim=1)            # True = ignore
            return self.norm(self.enc(h, src_key_padding_mask=kpm))

    return NightEncoder()


def build_predictor(d=128, pred_dim=64, depth=3, heads=4, dropout=0.1):
    torch, nn = _import_torch()

    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.mem_proj = nn.Linear(d, pred_dim)
            self.mask_tok = nn.Parameter(torch.zeros(1, 1, pred_dim))
            nn.init.trunc_normal_(self.mask_tok, std=0.02)
            pe = torch.zeros(MAX_BLK, pred_dim)
            pos = torch.arange(MAX_BLK).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, pred_dim, 2).float() * (-math.log(10000.0) / pred_dim))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe)
            self.stage_emb = nn.Embedding(N_STAGE + 1, pred_dim)
            layer = nn.TransformerDecoderLayer(pred_dim, heads, pred_dim * 4, dropout,
                                               activation="gelu", batch_first=True,
                                               norm_first=True)
            self.dec = nn.TransformerDecoder(layer, depth)
            self.head = nn.Linear(pred_dim, d)

        def forward(self, mem, mem_vis, tgt_pos, tgt_stage):
            """mem (B,1+M,d) student ctx output; mem_vis (B,1+M) visible keys;
            tgt_pos (B,Kt) block idx; tgt_stage (B,Kt). -> predicted (B,Kt,d)."""
            B, Kt = tgt_pos.shape
            m = self.mem_proj(mem)                             # (B,1+M,pd)
            st = (tgt_stage + 1).clamp(0, N_STAGE)
            q = self.mask_tok.expand(B, Kt, -1) + self.pe[tgt_pos] + self.stage_emb(st)
            out = self.dec(q, m, memory_key_padding_mask=~mem_vis)
            return self.head(out)                              # (B,Kt,d)

    return Predictor()


def build_jepa(cfg):
    torch, nn = _import_torch()

    class NightJEPA(nn.Module):
        def __init__(self):
            super().__init__()
            self.student = build_night_encoder(cfg["d_in"], cfg["d"], cfg["depth"], cfg["heads"])
            self.teacher = build_night_encoder(cfg["d_in"], cfg["d"], cfg["depth"], cfg["heads"])
            self.teacher.load_state_dict(self.student.state_dict())
            for p in self.teacher.parameters():
                p.requires_grad_(False)
            self.pred = build_predictor(cfg["d"], cfg["pred_dim"], cfg["pred_depth"], cfg["heads"])

        @torch.no_grad()
        def ema(self, m):
            for pt, ps in zip(self.teacher.parameters(), self.student.parameters()):
                pt.mul_(m).add_(ps, alpha=1 - m)
            for bt, bs in zip(self.teacher.buffers(), self.student.buffers()):
                bt.copy_(bs)

        def loss(self, blocks, bstage, valid, ctx, tgt_pos, tgt_stage):
            with torch.no_grad():
                ht = self.teacher.encode(blocks, bstage, valid)          # (B,1+M,d)
                tgt = torch.gather(ht[:, 1:], 1,
                                   tgt_pos.unsqueeze(-1).expand(-1, -1, ht.shape[-1]))
                tgt = nn.functional.layer_norm(tgt, (tgt.shape[-1],))
            hs = self.student.encode(blocks, bstage, ctx)                # ctx-only
            ones = torch.ones(ctx.shape[0], 1, dtype=torch.bool, device=ctx.device)
            mem_vis = torch.cat([ones, ctx], dim=1)
            pred = self.pred(hs, mem_vis, tgt_pos, tgt_stage)            # (B,Kt,d)
            return nn.functional.smooth_l1_loss(pred, tgt)

    return NightJEPA()


# ===================================================== masking
def sample_masks(valid, nblk, rng, frac, device):
    """One contiguous target span of Kt blocks per night (I-JEPA temporal).
    Kt = per-batch minimum of max(2, round(frac*nblk)) so gathers stay rectangular."""
    torch, _ = _import_torch()
    B = valid.shape[0]
    ns = np.asarray(nblk).astype(int)
    per = np.maximum(2, np.round(frac * ns).astype(int))
    Kt = int(max(2, min(int(per.min()), 40)))
    ctx = valid.clone()
    tgt = np.zeros((B, Kt), dtype=np.int64)
    for i in range(B):
        n = int(ns[i])
        start = 0 if n <= Kt else int(rng.randint(0, n - Kt + 1))
        win = np.arange(start, start + Kt)
        tgt[i] = win
        ctx[i, torch.from_numpy(win).to(ctx.device)] = False
    return ctx, torch.from_numpy(tgt).to(device)


# ===================================================== pretrain
def pretrain(blocks_npz, out, *, epochs=60, batch=64, d=128, depth=4, heads=4,
             pred_dim=64, pred_depth=3, lr=1.5e-3, wd=0.05, frac=0.30,
             ema0=0.996, ema1=0.9999, device="cuda", seed=0, progress=None):
    torch, _ = _import_torch()
    z = np.load(blocks_npz, allow_pickle=True)
    blocks, bstage, bvalid, nblk = z["blocks"], z["bstage"], z["bvalid"], z["nblk"]
    d_in = blocks.shape[2]
    keep = np.where(nblk >= 6)[0]
    # length-sorted batches so the per-batch Kt is close to frac*nblk (not dragged
    # to 2 by the shortest night in a random batch)
    order = keep[np.argsort(nblk[keep])]
    batches = [order[s:s + batch] for s in range(0, len(order), batch)]
    batches = [b for b in batches if len(b) >= 4]

    cfg = dict(d_in=int(d_in), d=d, depth=depth, heads=heads,
               pred_dim=pred_dim, pred_depth=pred_depth)
    model = build_jepa(cfg).to(device)
    decay, no_decay = [], []
    for mod in (model.student, model.pred):
        for _, p in mod.named_parameters():
            (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.99))
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))
    rng = np.random.RandomState(seed)
    total = epochs * len(batches)
    step = 0
    t0 = time.time()
    trainable = list(model.student.parameters()) + list(model.pred.parameters())
    if progress:
        progress(f"pretrain nights={len(keep)} batches/ep={len(batches)} "
                 f"steps={total} d_in={d_in} cfg={cfg}")
    for ep in range(epochs):
        rng.shuffle(batches)
        losses = []
        for bi in batches:
            xb = torch.from_numpy(blocks[bi].astype(np.float32)).to(device)
            st = torch.from_numpy(bstage[bi].astype(np.int64)).to(device)
            vd = torch.from_numpy(bvalid[bi]).to(device)
            ctx, tgt_pos = sample_masks(vd, nblk[bi], rng, frac, device)
            tgt_stage = torch.gather(st, 1, tgt_pos)
            fd = step / max(1, total)
            for g in opt.param_groups:
                g["lr"] = 0.5 * lr * (1 + math.cos(math.pi * fd))
            m = ema0 + (ema1 - ema0) * fd
            with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
                loss = model.loss(xb, st, vd, ctx, tgt_pos, tgt_stage)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 3.0)
            scaler.step(opt)
            scaler.update()
            model.ema(m)
            losses.append(float(loss))
            step += 1
        if progress:
            progress(f"ep{ep + 1}/{epochs} loss={np.mean(losses):.4f} "
                     f"lr={opt.param_groups[0]['lr']:.2e} ({time.time() - t0:.0f}s)")
    torch.save({"cfg": cfg, "teacher": model.teacher.state_dict(),
                "student": model.student.state_dict(),
                "pred": model.pred.state_dict()}, out)
    if progress:
        progress(f"saved night-JEPA -> {out}")


# ===================================================== Exp-C: leave-one-block-out error
def _expc_errors(model, blk, st, vd, n, d, device):
    torch, nn = _import_torch()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
        ht = model.teacher.encode(blk, st, vd)                 # (1,1+M,d)
        tgt = nn.functional.layer_norm(ht[0, 1:1 + n], (d,))   # (n,d)
        blkn = blk.expand(n, -1, -1).contiguous()
        stn = st.expand(n, -1).contiguous()
        vdn = vd.expand(n, -1).clone()
        for b in range(n):
            vdn[b, b] = False                                  # mask block b for row b
        tgt_pos = torch.arange(n, device=device).unsqueeze(1)  # (n,1)
        tgt_stage = torch.gather(stn, 1, tgt_pos)
        hs = model.student.encode(blkn, stn, vdn)              # (n,1+M,d)
        ones = torch.ones(n, 1, dtype=torch.bool, device=device)
        mem_vis = torch.cat([ones, vdn], dim=1)
        pred = model.pred(hs, mem_vis, tgt_pos, tgt_stage)[:, 0].float()  # (n,d)
        err = nn.functional.smooth_l1_loss(pred, tgt.float(), reduction="none").mean(1)
    return err.cpu().numpy()                                   # (n,)


# ===================================================== embed
def embed(blocks_npz, ckpt, out, *, device="cuda", progress=None):
    torch, nn = _import_torch()
    z = np.load(blocks_npz, allow_pickle=True)
    blocks, bstage, bvalid, nblk = z["blocks"], z["bstage"], z["bvalid"], z["nblk"]
    bids, sites, labels = z["bids"], z["sites"], z["labels"]
    state = torch.load(ckpt, weights_only=False)
    cfg = state["cfg"]
    model = build_jepa(cfg).to(device).eval()
    model.teacher.load_state_dict(state["teacher"])
    model.student.load_state_dict(state["student"])
    model.pred.load_state_dict(state["pred"])
    d = cfg["d"]
    n_night = len(bids)
    n_pool = 1 + 1 + N_STAGE                                   # cls + glob + 5 stages
    EMB = np.zeros((n_night, n_pool * d), np.float32)
    expc_cols = ["expc_glob_mean", "expc_glob_max", "expc_glob_cv", "expc_stage_mod"] + \
                [f"expc_{t}_mean" for t in STAGE_TAGS]
    EXPC = np.zeros((n_night, len(expc_cols)), np.float32)
    for ni in range(n_night):
        n = int(nblk[ni])
        if n < 1:
            continue
        blk = torch.from_numpy(blocks[ni:ni + 1].astype(np.float32)).to(device)
        st = torch.from_numpy(bstage[ni:ni + 1].astype(np.int64)).to(device)
        vd = torch.from_numpy(bvalid[ni:ni + 1]).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
            ht = model.teacher.encode(blk, st, vd)             # (1,1+M,d)
        cls = ht[0, 0].float().cpu().numpy()
        reps = ht[0, 1:1 + n].float().cpu().numpy()            # (n,d)
        stg = np.asarray(bstage[ni, :n])
        parts = [cls, reps.mean(0)]
        for si in range(N_STAGE):
            m = stg == si
            parts.append(reps[m].mean(0) if m.any() else np.zeros(d, np.float32))
        EMB[ni] = np.concatenate(parts)
        if n >= 4:
            errs = _expc_errors(model, blk, st, vd, n, d, device)   # (n,)
            EXPC[ni, 0] = float(errs.mean())
            EXPC[ni, 1] = float(errs.max())
            EXPC[ni, 2] = float(errs.std() / (errs.mean() + 1e-6))
            ps = []
            for si in range(N_STAGE):
                m = stg == si
                if m.any():
                    v = float(errs[m].mean())
                    EXPC[ni, 4 + si] = v
                    ps.append(v)
            EXPC[ni, 3] = float(np.std(ps)) if len(ps) >= 2 else 0.0
        if progress and ni % 200 == 0:
            progress(f"  night {ni}/{n_night}")
    cols = [f"l2_cls_{i}" for i in range(d)] + [f"l2_glob_{i}" for i in range(d)]
    for t in STAGE_TAGS:
        cols += [f"l2_{t}_{i}" for i in range(d)]
    cols += expc_cols
    full = np.concatenate([EMB, EXPC], axis=1)
    np.savez_compressed(out, emb=full, cols=np.array(cols),
                        bids=bids, sites=sites, labels=labels.astype(int))
    if progress:
        progress(f"saved night emb {full.shape} (+{len(expc_cols)} Exp-C) -> {out}")


# ===================================================== CLI
def _p(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("dumpep", help="L1: encode packed epochs with V2 teacher")
    a.add_argument("--prefix", required=True)
    a.add_argument("--ckpt", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--batch", type=int, default=512)
    a.add_argument("--device", default="cuda")

    b = sub.add_parser("blocks", help="L2: pool epochs into 5-min block sequences")
    b.add_argument("--prefix", required=True)
    b.add_argument("--epemb", required=True)
    b.add_argument("--out", required=True)

    c = sub.add_parser("pretrain", help="temporal I-JEPA over night block sequences")
    c.add_argument("--blocks", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--epochs", type=int, default=60)
    c.add_argument("--batch", type=int, default=64)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--device", default="cuda")

    e = sub.add_parser("embed", help="night embedding + Exp-C prediction-error features")
    e.add_argument("--blocks", required=True)
    e.add_argument("--ckpt", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--device", default="cuda")

    args = ap.parse_args()
    if args.cmd == "dumpep":
        dumpep(args.prefix, args.ckpt, args.out, batch=args.batch, device=args.device, progress=_p)
    elif args.cmd == "blocks":
        build_blocks(args.prefix, args.epemb, args.out, progress=_p)
    elif args.cmd == "pretrain":
        pretrain(args.blocks, args.out, epochs=args.epochs, batch=args.batch,
                 device=args.device, seed=args.seed, progress=_p)
    elif args.cmd == "embed":
        embed(args.blocks, args.ckpt, args.out, device=args.device, progress=_p)


if __name__ == "__main__":
    main()
