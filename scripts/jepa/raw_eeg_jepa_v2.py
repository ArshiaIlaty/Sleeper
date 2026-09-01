"""Raw multichannel-EEG epoch-JEPA V2 (Challenge 2026 exploration, GPU).

V1 (raw_eeg_jepa.py) hit a floor: jepa-only AUROC = 0.5030 (literal chance) under
LOSO, so the mean-pooled night embedding carried no linear CI signal. V2 folds in
the three techniques that all four borrowed repos (ECG-JEPA, EEG-VJEPA, PhysioJEPA,
SignalJEPA) converge on, each targeting one V1 failure mode:

  1. PER-CHANNEL (C x T) TOKENS + learned channel embedding.
     V1 collapsed the 6 derivations with a single Conv1d(6->d) at the patch step,
     erasing topography. V2 patch-embeds each channel independently (shared conv
     weights, PatchTST-style) -> 6 x 30 = 180 tokens, disambiguated by a learned
     channel-embedding table + sinusoidal time PE. (ECG-JEPA / EEG-VJEPA caveat.)

  2. I-JEPA MASKING (context encoder sees ONLY kept tokens).
     V1 used data2vec in-place mask tokens (too-easy pretext). V2 gathers a small
     context set, runs the encoder on it alone, and a NARROW predictor (dim/2)
     regresses the EMA-teacher's layer-normed latents at the masked target tokens
     from learnable mask-tokens + target position embeddings. (All four repos.)

  3. REGISTER/CLS-TOKEN READOUT.
     V1's night vector = mean over 30 layernormed patch tokens then mean over
     epochs -> washed out to near-constant. V2 prepends a CLS/register token whose
     encoder output is the per-epoch descriptor (a learned aggregate), then stage-
     aware mean-pools those across the night. (ECG-JEPA / EEG-VJEPA readout.)

Everything else matches V1's leak-free protocol: a single label-free all-nights
pretrain, LOSO enforced only in the GBM head (jepa_gate_ab.py). Reuses V1's packed
memmap (packed_epochs.npy + packed_index.npz) and the emb.npz gate schema unchanged.
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
EPOCH_LEN = 1920                       # 30 s @ 64 Hz
N_CH = 6                               # F3,F4,C3,C4,O1,O2 (canonical order)
PATCH = 64                             # 1 s patches
N_PATCH = EPOCH_LEN // PATCH           # 30 time tokens/channel
N_TOK = N_CH * N_PATCH                 # 180 patch tokens/epoch (channel-major)

# CAISR stage codes {1:N3,2:N2,3:N1,4:REM,5:Wake,9:Unknown}. Pool index order:
# Wake,N1,N2,N3,REM -> 0..4; anything else (Unknown) -> -1 (pooled only in global).
STAGE2IDX = {5: 0, 3: 1, 2: 2, 1: 3, 4: 4}
N_STAGE = 5
STAGE_TAGS = ["wake", "n1", "n2", "n3", "rem"]


# ============================================================= index (reuse V1 pack)
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
    """Per-channel patch tokens + channel/time position + CLS register token."""
    torch, nn = _import_torch()

    class EpochEncoderV2(nn.Module):
        def __init__(self):
            super().__init__()
            # SHARED across channels (PatchTST channel-independence): 1 s patches.
            self.patch = nn.Conv1d(1, d_model, kernel_size=PATCH, stride=PATCH)
            self.chan_emb = nn.Embedding(N_CH, d_model)                 # topography
            self.register_buffer("time_pe", _sinusoid(N_PATCH, d_model),
                                 persistent=False)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))         # register token
            nn.init.normal_(self.cls, std=0.02)
            nn.init.normal_(self.chan_emb.weight, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model, heads, dim_feedforward=2 * d_model, dropout=dropout,
                batch_first=True, norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, depth)
            self.norm = nn.LayerNorm(d_model)
            self.d_model = d_model

        def grid_pos(self):
            ce = self.chan_emb(torch.arange(N_CH, device=self.cls.device))  # (6,d)
            g = ce.unsqueeze(1) + self.time_pe.unsqueeze(0)                 # (6,30,d)
            return g.reshape(N_TOK, self.d_model)                           # (180,d)

        def patch_tokens(self, x):
            """x (B,6,1920) -> (B,180,d) channel-major patch embeddings (no pos)."""
            B = x.shape[0]
            xr = x.reshape(B * N_CH, 1, EPOCH_LEN)
            p = self.patch(xr).transpose(1, 2)                # (B*6, 30, d)
            p = p.reshape(B, N_CH, N_PATCH, self.d_model)
            return p.reshape(B, N_TOK, self.d_model)

        def encode_ctx(self, x, ctx_idx):
            """Encoder over CLS + the gathered context tokens; return (B,Kc,d)."""
            tok = self.patch_tokens(x) + self.grid_pos().unsqueeze(0)     # (B,180,d)
            gi = ctx_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
            ctx = torch.gather(tok, 1, gi)                                # (B,Kc,d)
            B = ctx.shape[0]
            cls = self.cls.expand(B, -1, -1)
            h = self.norm(self.enc(torch.cat([cls, ctx], dim=1)))
            return h[:, 1:]                                               # drop CLS

        def encode_full(self, x):
            """Teacher path: encoder over CLS + all 180 tokens; return (B,180,d)."""
            tok = self.patch_tokens(x) + self.grid_pos().unsqueeze(0)
            B = tok.shape[0]
            cls = self.cls.expand(B, -1, -1)
            h = self.norm(self.enc(torch.cat([cls, tok], dim=1)))
            return h[:, 1:]

        def cls_embed(self, x):
            """Per-epoch descriptor = CLS/register output. Return (B,d)."""
            tok = self.patch_tokens(x) + self.grid_pos().unsqueeze(0)
            B = tok.shape[0]
            cls = self.cls.expand(B, -1, -1)
            h = self.norm(self.enc(torch.cat([cls, tok], dim=1)))
            return h[:, 0]

    return EpochEncoderV2()


def build_predictor(d_model, pred_dim, depth=3, heads=4, dropout=0.1):
    torch, nn = _import_torch()

    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(d_model, pred_dim)
            self.pos = nn.Linear(d_model, pred_dim)          # project grid_pos
            self.mask_token = nn.Parameter(torch.zeros(pred_dim))
            nn.init.normal_(self.mask_token, std=0.02)
            layer = nn.TransformerEncoderLayer(
                pred_dim, heads, dim_feedforward=2 * pred_dim, dropout=dropout,
                batch_first=True, norm_first=True, activation="gelu")
            self.blocks = nn.TransformerEncoder(layer, depth)
            self.norm = nn.LayerNorm(pred_dim)
            self.proj = nn.Linear(pred_dim, d_model)

        def forward(self, ctx_reps, ctx_idx, tgt_idx, grid_pos):
            cp = self.pos(grid_pos)                          # (180, pred_dim)
            x = self.embed(ctx_reps) + cp[ctx_idx]           # (B,Kc,pred_dim)
            B, Kt = tgt_idx.shape
            mt = self.mask_token.view(1, 1, -1).expand(B, Kt, -1) + cp[tgt_idx]
            Kc = x.shape[1]
            h = self.norm(self.blocks(torch.cat([x, mt], dim=1)))
            return self.proj(h[:, Kc:])                      # (B,Kt,d_model)

    return Predictor()


class RawJEPAv2:
    """Student encoder + EMA teacher + narrow I-JEPA predictor.
    Not an nn.Module so the teacher stays out of any EMA deepcopy."""

    def __init__(self, d_model=128, depth=6, heads=4, pred_dim=None, pred_depth=3,
                 dropout=0.1, seed=0):
        torch, nn = _import_torch()
        torch.manual_seed(seed)
        self.torch, self.nn = torch, nn
        self.student = build_epoch_encoder(d_model, depth, heads, dropout)
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        pred_dim = pred_dim or (d_model // 2)                # narrow predictor
        self.predictor = build_predictor(d_model, pred_dim, pred_depth, heads, dropout)
        self.d_model = d_model
        self.pred_dim = pred_dim

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

    def loss(self, x, ctx_idx, tgt_idx):
        torch, F = self.torch, self.torch.nn.functional
        with torch.no_grad():
            tgt = self.teacher.encode_full(x)                # (B,180,d)
            tgt = F.layer_norm(tgt, (self.d_model,))
            gi = tgt_idx.unsqueeze(-1).expand(-1, -1, self.d_model)
            tgt = torch.gather(tgt, 1, gi)                   # (B,Kt,d)
        ctx = self.student.encode_ctx(x, ctx_idx)            # (B,Kc,d)
        pred = self.predictor(ctx, ctx_idx, tgt_idx, self.student.grid_pos())
        return F.smooth_l1_loss(pred, tgt)


# ============================================================= I-JEPA masking
def make_block_mask(B, mask_frac, rng, min_ch=2, max_ch=6, min_t=3, max_t=10):
    """(B, 180) bool over the (6-channel x 30-time) grid, True=target/masked.
    Rectangular blocks in (channel, time) grown until ~mask_frac is covered;
    each row is guaranteed >=1 context and >=1 target token."""
    N = N_TOK
    target = int(mask_frac * N)
    out = np.zeros((B, N), dtype=bool)
    for i in range(B):
        g = np.zeros((N_CH, N_PATCH), dtype=bool)
        got, guard = 0, 0
        while got < target and guard < 60:
            guard += 1
            bh = rng.randint(min_ch, max_ch + 1)
            bw = rng.randint(min_t, max_t + 1)
            top = rng.randint(0, N_CH - bh + 1)
            left = rng.randint(0, N_PATCH - bw + 1)
            before = g.sum()
            g[top:top + bh, left:left + bw] = True
            got += g.sum() - before
        flat = g.reshape(-1)
        if flat.all():                                       # keep >=1 context
            flat[rng.randint(0, N)] = False
        if not flat.any():                                   # keep >=1 target
            flat[rng.randint(0, N)] = True
        out[i] = flat
    return out


def ijepa_indices(mask):
    """From (B,180) bool -> equal-length (B,Kc) context idx, (B,Kt) target idx.
    Kc/Kt = per-batch minimum counts (I-JEPA batches gather-equal lengths)."""
    B = mask.shape[0]
    Kc = max(1, int((~mask).sum(1).min()))
    Kt = max(1, int(mask.sum(1).min()))
    cidx = np.zeros((B, Kc), dtype=np.int64)
    tidx = np.zeros((B, Kt), dtype=np.int64)
    for i in range(B):
        c = np.where(~mask[i])[0]
        t = np.where(mask[i])[0]
        cidx[i] = c[:Kc]
        tidx[i] = t[:Kt]
    return cidx, tidx


# ============================================================= pretraining
def _param_groups(nn_params, wd):
    """AdamW: weight-decay only on >=2-D weights (exclude bias/norm/emb/cls/mask)."""
    decay, no_decay = [], []
    for p in nn_params:
        if not p.requires_grad:
            continue
        (decay if p.ndim >= 2 else no_decay).append(p)
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]


def pretrain(out_prefix, ckpt_out, *, exclude_site="", d_model=128, depth=6,
             heads=4, pred_depth=3, epochs=15, batch=256, lr=1.5e-3, wd=0.05,
             mask_frac=0.6, ema0=0.998, ema1=0.9999, seed=0, device="cuda",
             amp=True, limit_nights=0, max_steps=0, progress=None):
    torch = __import__("torch")
    torch.manual_seed(seed)
    dat, idx = load_index(out_prefix)
    epoch_night = idx["epoch_night"]
    night_site = idx["night_site"]
    if exclude_site:
        held = set(np.where(night_site == exclude_site)[0].tolist())
        pool = np.where(~np.isin(epoch_night, list(held)))[0]
    else:
        pool = np.arange(dat.shape[0])
    if limit_nights:                                          # smoke: first K nights
        keep_n = set(range(min(limit_nights, len(night_site))))
        pool = pool[np.isin(epoch_night[pool], list(keep_n))]
    if progress:
        progress(f"pretrain V2 on {len(pool)} epochs "
                 f"(exclude_site={exclude_site or 'NONE'}, device={device})")
    rng = np.random.RandomState(seed)
    model = RawJEPAv2(d_model, depth, heads, pred_depth=pred_depth, seed=seed).to(device)
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
            sel_sorted = np.sort(sel)                         # memmap read locality
            xb = np.ascontiguousarray(dat[sel_sorted]).astype(np.float32)
            x = torch.from_numpy(xb).to(device, non_blocking=True)
            mask = make_block_mask(len(sel_sorted), mask_frac, rng)
            cidx, tidx = ijepa_indices(mask)
            ci = torch.from_numpy(cidx).to(device)
            ti = torch.from_numpy(tidx).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = model.loss(x, ci, ti)
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
def embed(out_prefix, ckpt, emb_out, *, batch=512, device="cuda", progress=None):
    """Per-epoch CLS/register descriptor -> stage-aware night pooling:
    [global] + per-stage means (Wake,N1,N2,N3,REM). Missing stage -> zeros."""
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
    night_bids, night_site = idx["night_bids"], idx["night_site"]
    night_label = idx["night_label"]
    n_night = len(night_bids)

    # 1) encode every epoch -> per-epoch CLS descriptor
    N = dat.shape[0]
    ep_emb = np.empty((N, d), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, N, batch):
            e = min(N, s + batch)
            xb = np.ascontiguousarray(dat[s:e]).astype(np.float32)
            x = torch.from_numpy(xb).to(device)
            with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
                z = enc.cls_embed(x)                          # (b,d)
            ep_emb[s:e] = z.float().cpu().numpy()
            if progress and (s // batch) % 40 == 0:
                progress(f"  encoded {e}/{N} epochs")

    # 2) stage-aware pooling per night
    n_pool = 1 + N_STAGE
    emb = np.zeros((n_night, n_pool * d), dtype=np.float32)
    for ni in range(n_night):
        a, b = int(offsets[ni]), int(offsets[ni + 1])
        if b <= a:
            continue
        block = ep_emb[a:b]
        pidx = pool_idx[a:b]
        parts = [block.mean(axis=0)]                          # global
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

    t = sub.add_parser("pretrain")
    t.add_argument("--prefix", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--exclude-site", default="")
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--batch", type=int, default=256)
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
    e.add_argument("--batch", type=int, default=512)
    e.add_argument("--device", default="cuda")

    args = ap.parse_args()
    if args.cmd == "pretrain":
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
