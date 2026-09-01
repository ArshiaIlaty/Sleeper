"""Multimodal PSG epoch-JEPA (Challenge 2026 exploration, GPU).

Escalation from raw_eeg_jepa_v2.py (EEG-only). V2 validated the borrowed I-JEPA
recipe (jepa-only AUROC chance->0.629) but the representation was REDUNDANT with the
handcrafted EEG features. This model feeds the full 15-channel PSG montage
(export_raw_multimodal.py: 6 EEG + ECG + 2 EOG + chin EMG + airflow + chest/abd
effort + SpO2 + leg EMG) to the SAME architecture, so cross-modal structure (arousal
<-> HRV, apnea -> desaturation, REM EOG/EMG atonia) can surface as signal the
EEG-centric champion under-exploits.

Architecture = raw_eeg_jepa_v2 unchanged EXCEPT:
  * N_CH = 15 (per-channel patch tokens -> 15 x 30 = 450 tokens/epoch), the learned
    channel-embedding table now distinguishes modalities as well as derivations.
  * PRESENCE-AWARE: montage coverage varies per night (chan_mask). Absent channels
    are zero-filled but excluded everywhere -- padded out of the encoder's attention
    (src_key_padding_mask), never sampled as I-JEPA context/target, and dropped from
    the CLS readout's attention. So a night with 13/15 channels is encoded honestly.

Same leak-free protocol as V1/V2: single label-free all-nights pretrain, LOSO only in
the GBM head (jepa_gate_ab.py). Same emb.npz schema. Reuses its own packed memmap.
"""
import argparse
import copy
import glob
import math
import os
import sys
import time

import numpy as np

# ------------------------------------------------------------------ schema
FS = 64.0
EPOCH_LEN = 1920
N_CH = 15
PATCH = 64
N_PATCH = EPOCH_LEN // PATCH            # 30 time tokens/channel
N_TOK = N_CH * N_PATCH                  # 450 patch tokens/epoch (channel-major)

STAGE2IDX = {5: 0, 3: 1, 2: 2, 1: 3, 4: 4}
N_STAGE = 5
STAGE_TAGS = ["wake", "n1", "n2", "n3", "rem"]


# ============================================================= pack
def pack(raw_dir, out_prefix, progress=None):
    """Concatenate per-recording multimodal npz -> one float16 memmap + index.
    Index adds night_chan_mask (n_night,15) so presence can be looked up per epoch."""
    files = sorted(glob.glob(os.path.join(raw_dir, "*.npz")))
    if not files:
        raise SystemExit(f"no npz in {raw_dir}")
    night_bids, night_site, night_label, offsets = [], [], [], [0]
    night_mask, stages_all, metas = [], [], []
    n_total = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        ne = int(d["n_epoch"])
        if ne < 1:
            continue
        night_bids.append(str(d["bids"]))
        night_site.append(str(d["site"]))
        night_label.append(int(d["label"]))
        night_mask.append(np.asarray(d["chan_mask"]).astype(np.int8))
        stages_all.append(np.asarray(d["stages"]).astype(np.int8))
        n_total += ne
        offsets.append(n_total)
        metas.append((f, ne))
    if progress:
        progress(f"pack: {len(metas)} nights, {n_total} epochs "
                 f"-> {n_total * N_CH * EPOCH_LEN * 2 / 1e9:.1f} GB memmap")
    dat = np.lib.format.open_memmap(out_prefix + "_epochs.npy", mode="w+",
                                    dtype=np.float16, shape=(n_total, N_CH, EPOCH_LEN))
    epoch_night = np.empty(n_total, dtype=np.int32)
    ptr = 0
    for ni, (f, ne) in enumerate(metas):
        dat[ptr:ptr + ne] = np.load(f, allow_pickle=True)["epochs"]
        epoch_night[ptr:ptr + ne] = ni
        ptr += ne
        if progress and ni % 50 == 0:
            progress(f"  packed {ni + 1}/{len(metas)} nights ({ptr} epochs)")
    dat.flush()
    stages = np.concatenate(stages_all).astype(np.int8)
    pool_idx = np.array([STAGE2IDX.get(int(s), -1) for s in stages], dtype=np.int8)
    np.savez(out_prefix + "_index.npz",
             night_bids=np.array(night_bids), night_site=np.array(night_site),
             night_label=np.array(night_label, dtype=np.int32),
             night_chan_mask=np.vstack(night_mask).astype(np.int8),
             offsets=np.array(offsets, dtype=np.int64),
             epoch_night=epoch_night, epoch_stage=stages, epoch_pool=pool_idx,
             shape=np.array([n_total, N_CH, EPOCH_LEN], dtype=np.int64))
    if progress:
        progress(f"pack DONE -> {out_prefix}_epochs.npy + _index.npz")


def load_index(out_prefix):
    idx = np.load(out_prefix + "_index.npz", allow_pickle=True)
    dat = np.load(out_prefix + "_epochs.npy", mmap_mode="r")
    return dat, idx


# ============================================================= torch model
def _import_torch():
    import torch
    import torch.nn as nn
    return torch, nn


def _sinusoid(max_len, d):
    import torch
    pe = torch.zeros(max_len, d)
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def build_epoch_encoder(d_model=128, depth=6, heads=4, dropout=0.1):
    torch, nn = _import_torch()

    class MMEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch = nn.Conv1d(1, d_model, kernel_size=PATCH, stride=PATCH)
            self.chan_emb = nn.Embedding(N_CH, d_model)          # modality/derivation id
            self.register_buffer("time_pe", _sinusoid(N_PATCH, d_model),
                                 persistent=False)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls, std=0.02)
            nn.init.normal_(self.chan_emb.weight, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model, heads, dim_feedforward=2 * d_model, dropout=dropout,
                batch_first=True, norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, depth)
            self.norm = nn.LayerNorm(d_model)
            self.d_model = d_model

        def grid_pos(self):
            ce = self.chan_emb(torch.arange(N_CH, device=self.cls.device))  # (15,d)
            g = ce.unsqueeze(1) + self.time_pe.unsqueeze(0)                 # (15,30,d)
            return g.reshape(N_TOK, self.d_model)

        def patch_tokens(self, x):
            B = x.shape[0]
            xr = x.reshape(B * N_CH, 1, EPOCH_LEN)
            p = self.patch(xr).transpose(1, 2)               # (B*15,30,d)
            p = p.reshape(B, N_CH, N_PATCH, self.d_model)
            return p.reshape(B, N_TOK, self.d_model)

        def _run(self, seq, kpm=None):
            return self.norm(self.enc(seq, src_key_padding_mask=kpm))

        def encode_ctx(self, x, ctx_idx):
            """Context = gathered PRESENT tokens (no padding needed). Return (B,Kc,d)."""
            tok = self.patch_tokens(x) + self.grid_pos().unsqueeze(0)
            gi = ctx_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
            ctx = torch.gather(tok, 1, gi)
            B = ctx.shape[0]
            cls = self.cls.expand(B, -1, -1)
            return self._run(torch.cat([cls, ctx], dim=1))[:, 1:]

        def encode_full(self, x, present):
            """Teacher path over all 450 tokens; absent channels padded out. (B,450,d)."""
            tok = self.patch_tokens(x) + self.grid_pos().unsqueeze(0)
            B = tok.shape[0]
            cls = self.cls.expand(B, -1, -1)
            pad = torch.zeros(B, 1, dtype=torch.bool, device=tok.device)
            kpm = torch.cat([pad, ~present], dim=1)          # True = ignore
            return self._run(torch.cat([cls, tok], dim=1), kpm)[:, 1:]

        def cls_embed(self, x, present):
            """Per-epoch descriptor = CLS output; absent channels padded out. (B,d)."""
            tok = self.patch_tokens(x) + self.grid_pos().unsqueeze(0)
            B = tok.shape[0]
            cls = self.cls.expand(B, -1, -1)
            pad = torch.zeros(B, 1, dtype=torch.bool, device=tok.device)
            kpm = torch.cat([pad, ~present], dim=1)
            return self._run(torch.cat([cls, tok], dim=1), kpm)[:, 0]

    return MMEncoder()


def build_predictor(d_model, pred_dim, depth=3, heads=4, dropout=0.1):
    torch, nn = _import_torch()

    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(d_model, pred_dim)
            self.pos = nn.Linear(d_model, pred_dim)
            self.mask_token = nn.Parameter(torch.zeros(pred_dim))
            nn.init.normal_(self.mask_token, std=0.02)
            layer = nn.TransformerEncoderLayer(
                pred_dim, heads, dim_feedforward=2 * pred_dim, dropout=dropout,
                batch_first=True, norm_first=True, activation="gelu")
            self.blocks = nn.TransformerEncoder(layer, depth)
            self.norm = nn.LayerNorm(pred_dim)
            self.proj = nn.Linear(pred_dim, d_model)

        def forward(self, ctx_reps, ctx_idx, tgt_idx, grid_pos):
            cp = self.pos(grid_pos)                          # (450, pred_dim)
            x = self.embed(ctx_reps) + cp[ctx_idx]
            B, Kt = tgt_idx.shape
            mt = self.mask_token.view(1, 1, -1).expand(B, Kt, -1) + cp[tgt_idx]
            Kc = x.shape[1]
            h = self.norm(self.blocks(torch.cat([x, mt], dim=1)))
            return self.proj(h[:, Kc:])

    return Predictor()


class MMJEPA:
    def __init__(self, d_model=128, depth=6, heads=4, pred_dim=None, pred_depth=3,
                 dropout=0.1, seed=0):
        torch, nn = _import_torch()
        torch.manual_seed(seed)
        self.torch, self.nn = torch, nn
        self.student = build_epoch_encoder(d_model, depth, heads, dropout)
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        pred_dim = pred_dim or (d_model // 2)
        self.predictor = build_predictor(d_model, pred_dim, pred_depth, heads, dropout)
        self.d_model = d_model

    def trainable_params(self):
        return list(self.student.parameters()) + list(self.predictor.parameters())

    def train(self):
        self.student.train(); self.predictor.train()

    def eval(self):
        self.student.eval(); self.teacher.eval(); self.predictor.eval()

    def to(self, device):
        self.student.to(device); self.teacher.to(device); self.predictor.to(device)
        return self

    def ema_update(self, m):
        torch = self.torch
        with torch.no_grad():
            for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
                pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)
            for bs, bt in zip(self.student.buffers(), self.teacher.buffers()):
                bt.copy_(bs)

    def loss(self, x, present, ctx_idx, tgt_idx):
        torch, F = self.torch, self.torch.nn.functional
        with torch.no_grad():
            tgt = self.teacher.encode_full(x, present)       # (B,450,d)
            tgt = F.layer_norm(tgt, (self.d_model,))
            gi = tgt_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
            tgt = torch.gather(tgt, 1, gi)                   # (B,Kt,d)
        ctx = self.student.encode_ctx(x, ctx_idx)
        pred = self.predictor(ctx, ctx_idx, tgt_idx, self.student.grid_pos())
        return F.smooth_l1_loss(pred, tgt)


# ============================================================= presence-aware masking
def make_block_mask(B, mask_frac, rng, min_ch=2, max_ch=8, min_t=3, max_t=10):
    """(B,450) bool over the (15ch x 30t) grid, True=target/masked (blocks)."""
    N = N_TOK
    target = int(mask_frac * N)
    out = np.zeros((B, N), dtype=bool)
    for i in range(B):
        g = np.zeros((N_CH, N_PATCH), dtype=bool)
        got, guard = 0, 0
        while got < target and guard < 80:
            guard += 1
            bh = rng.randint(min_ch, max_ch + 1)
            bw = rng.randint(min_t, max_t + 1)
            top = rng.randint(0, N_CH - bh + 1)
            left = rng.randint(0, N_PATCH - bw + 1)
            before = g.sum()
            g[top:top + bh, left:left + bw] = True
            got += g.sum() - before
        out[i] = g.reshape(-1)
    return out


def ijepa_indices(mask, present):
    """Presence-aware equal-length gather indices. Target/context are drawn only from
    PRESENT tokens; absent tokens are excluded from both. Kc/Kt = per-batch minima."""
    B = mask.shape[0]
    tcand, ccand = [], []
    for i in range(B):
        p = np.where(present[i])[0]
        mp = mask[i][p]
        t = p[mp]
        c = p[~mp]
        if len(t) == 0:                                      # guarantee >=1 target
            t, c = c[-1:], c[:-1]
        if len(c) == 0:                                      # guarantee >=1 context
            c, t = t[-1:], t[:-1]
        tcand.append(t); ccand.append(c)
    Kt = max(1, min(len(t) for t in tcand))
    Kc = max(1, min(len(c) for c in ccand))
    tidx = np.zeros((B, Kt), dtype=np.int64)
    cidx = np.zeros((B, Kc), dtype=np.int64)
    for i in range(B):
        tidx[i] = tcand[i][:Kt]
        cidx[i] = ccand[i][:Kc]
    return cidx, tidx


def _tok_present(night_mask_rows):
    """(B,15) int8 -> (B,450) bool, each channel repeated over its 30 time tokens."""
    return np.repeat(night_mask_rows.astype(bool), N_PATCH, axis=1)


# ============================================================= pretraining
def _param_groups(params, wd):
    decay, no_decay = [], []
    for p in params:
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]


def pretrain(out_prefix, ckpt_out, *, exclude_site="", d_model=128, depth=6,
             heads=4, pred_depth=3, epochs=12, batch=192, lr=1.5e-3, wd=0.05,
             mask_frac=0.6, ema0=0.998, ema1=0.9999, seed=0, device="cuda",
             amp=True, limit_nights=0, max_steps=0, progress=None):
    torch = __import__("torch")
    torch.manual_seed(seed)
    dat, idx = load_index(out_prefix)
    epoch_night = idx["epoch_night"]
    night_site = idx["night_site"]
    night_mask = idx["night_chan_mask"]
    if exclude_site:
        held = set(np.where(night_site == exclude_site)[0].tolist())
        pool = np.where(~np.isin(epoch_night, list(held)))[0]
    else:
        pool = np.arange(dat.shape[0])
    if limit_nights:
        keep_n = set(range(min(limit_nights, len(night_site))))
        pool = pool[np.isin(epoch_night[pool], list(keep_n))]
    if progress:
        progress(f"pretrain MM on {len(pool)} epochs "
                 f"(exclude_site={exclude_site or 'NONE'}, device={device})")
    rng = np.random.RandomState(seed)
    model = MMJEPA(d_model, depth, heads, pred_depth=pred_depth, seed=seed).to(device)
    opt = torch.optim.AdamW(_param_groups(model.trainable_params(), wd),
                            lr=lr, betas=(0.9, 0.99))
    steps_per_epoch = max(1, len(pool) // batch)
    total_steps = epochs * steps_per_epoch if not max_steps else max_steps
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    use_amp = amp and device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    model.train()
    step, t0, stop = 0, time.time(), False
    for ep in range(epochs):
        order = rng.permutation(len(pool))
        ep_loss, ep_n = 0.0, 0
        for bi in range(steps_per_epoch):
            sel = pool[order[bi * batch:(bi + 1) * batch]]
            if len(sel) < 2:
                continue
            sel_sorted = np.sort(sel)
            xb = np.ascontiguousarray(dat[sel_sorted]).astype(np.float32)
            x = torch.from_numpy(xb).to(device, non_blocking=True)
            pres_ch = night_mask[epoch_night[sel_sorted]]      # (B,15)
            pres_tok = _tok_present(pres_ch)                   # (B,450)
            mask = make_block_mask(len(sel_sorted), mask_frac, rng)
            cidx, tidx = ijepa_indices(mask, pres_tok)
            present = torch.from_numpy(pres_tok).to(device)
            ci = torch.from_numpy(cidx).to(device)
            ti = torch.from_numpy(tidx).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = model.loss(x, present, ci, ti)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.trainable_params(), 3.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            m = ema0 + (ema1 - ema0) * (step / max(1, total_steps))
            model.ema_update(m)
            step += 1
            ep_loss += float(loss.item()); ep_n += 1
            if max_steps and step >= max_steps:
                stop = True
                break
        if progress:
            progress(f"  epoch {ep+1}/{epochs} loss={ep_loss/max(1,ep_n):.4f} "
                     f"step={step} elapsed={time.time()-t0:.0f}s")
        if stop:
            break
    state = {"teacher": {k: v.cpu() for k, v in model.teacher.state_dict().items()},
             "cfg": {"d_model": d_model, "depth": depth, "heads": heads},
             "exclude_site": exclude_site}
    torch.save(state, ckpt_out)
    if progress:
        progress(f"saved encoder -> {ckpt_out} ({time.time()-t0:.0f}s)")


# ============================================================= embedding
def embed(out_prefix, ckpt, emb_out, *, batch=384, device="cuda", progress=None):
    """Per-epoch CLS descriptor (presence-aware) -> stage-aware night pooling."""
    torch = __import__("torch")
    dat, idx = load_index(out_prefix)
    state = torch.load(ckpt, weights_only=False)
    cfg = state["cfg"]
    enc = build_epoch_encoder(cfg["d_model"], cfg["depth"], cfg["heads"], dropout=0.0)
    enc.load_state_dict(state["teacher"])
    enc.to(device).eval()
    d = cfg["d_model"]
    offsets = idx["offsets"]
    pool_idx = idx["epoch_pool"]
    epoch_night = idx["epoch_night"]
    night_mask = idx["night_chan_mask"]
    night_bids, night_site = idx["night_bids"], idx["night_site"]
    night_label = idx["night_label"]
    n_night = len(night_bids)

    N = dat.shape[0]
    ep_emb = np.empty((N, d), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, N, batch):
            e = min(N, s + batch)
            xb = np.ascontiguousarray(dat[s:e]).astype(np.float32)
            x = torch.from_numpy(xb).to(device)
            pres = torch.from_numpy(_tok_present(night_mask[epoch_night[s:e]])).to(device)
            with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
                z = enc.cls_embed(x, pres)
            ep_emb[s:e] = z.float().cpu().numpy()
            if progress and (s // batch) % 40 == 0:
                progress(f"  encoded {e}/{N} epochs")

    n_pool = 1 + N_STAGE
    emb = np.zeros((n_night, n_pool * d), dtype=np.float32)
    for ni in range(n_night):
        a, b = int(offsets[ni]), int(offsets[ni + 1])
        if b <= a:
            continue
        block = ep_emb[a:b]
        pidx = pool_idx[a:b]
        parts = [block.mean(axis=0)]
        for si in range(N_STAGE):
            m = pidx == si
            parts.append(block[m].mean(axis=0) if m.any() else np.zeros(d, np.float32))
        emb[ni] = np.concatenate(parts)
    pool_tags = ["glob"] + STAGE_TAGS
    cols = [f"jepa_{tag}_{i}" for tag in pool_tags for i in range(d)]
    np.savez_compressed(emb_out, emb=emb, cols=np.array(cols),
                        bids=night_bids, sites=night_site,
                        labels=night_label.astype(int))
    if progress:
        progress(f"saved embeddings {emb.shape} -> {emb_out}")


# ============================================================= CLI
def _p(m):
    print(m, file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pack")
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--out-prefix", required=True)

    t = sub.add_parser("pretrain")
    t.add_argument("--prefix", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--exclude-site", default="")
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--batch", type=int, default=192)
    t.add_argument("--d-model", type=int, default=128)
    t.add_argument("--depth", type=int, default=6)
    t.add_argument("--pred-depth", type=int, default=3)
    t.add_argument("--lr", type=float, default=1.5e-3)
    t.add_argument("--mask-frac", type=float, default=0.6)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cuda")
    t.add_argument("--no-amp", action="store_true")
    t.add_argument("--limit-nights", type=int, default=0)
    t.add_argument("--max-steps", type=int, default=0)

    e = sub.add_parser("embed")
    e.add_argument("--prefix", required=True)
    e.add_argument("--ckpt", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--batch", type=int, default=384)
    e.add_argument("--device", default="cuda")

    args = ap.parse_args()
    if args.cmd == "pack":
        pack(args.raw_dir, args.out_prefix, progress=_p)
    elif args.cmd == "pretrain":
        pretrain(args.prefix, args.out, exclude_site=args.exclude_site,
                 d_model=args.d_model, depth=args.depth, pred_depth=args.pred_depth,
                 epochs=args.epochs, batch=args.batch, lr=args.lr,
                 mask_frac=args.mask_frac, seed=args.seed, device=args.device,
                 amp=not args.no_amp, limit_nights=args.limit_nights,
                 max_steps=args.max_steps, progress=_p)
    elif args.cmd == "embed":
        embed(args.prefix, args.ckpt, args.out, batch=args.batch,
              device=args.device, progress=_p)


if __name__ == "__main__":
    main()
