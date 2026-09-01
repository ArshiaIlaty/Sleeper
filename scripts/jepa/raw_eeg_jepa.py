"""Raw multichannel-EEG epoch-JEPA (Challenge 2026 exploration, GPU).

Escalation from the light per-epoch-feature TS-JEPA (which hit a floor: the
handcrafted spectral summaries carry no CI signal the champion lacks). Here the
encoder sees the RAW waveform of the six standard scalp derivations, so it can
learn morphology (spindles, K-complexes, slow-wave shape, spectral detail) that
no summary feature exposes.

JEPA objective (data2vec-style latent masking, same family as the TS-JEPA):
  * A conv stem patchifies each 30 s epoch (6, 1920) into N_PATCH=30 patch tokens
    (1 s / 64-sample patches, channel-mixed).
  * Student sees the epoch with a random subset of patch tokens replaced by a
    learnable mask-token; an EMA teacher sees the clean epoch.
  * The student's predictor regresses the teacher's layer-normed latents at the
    masked patches (smooth-L1). No reconstruction, no contrastive negatives.

Two-level hierarchy:
  * Level 1 (this JEPA) = per-epoch encoder, pretrained on ALL epochs of the
    training nights (label-free -> no target leakage under LOSO).
  * Level 2 = stage-aware pooling of frozen per-epoch embeddings into a night
    vector (global + per-stage means over Wake/N1/N2/N3/REM), fused with the
    champion 436-feature set and gated under LOSO by jepa_gate_ab.py.

Subcommands:
  pack     per-recording npz (from export_raw_eeg.py) -> one memmap + index
  pretrain epoch-level JEPA on the memmap (GPU); saves the EMA-teacher encoder
  embed    per-epoch encode -> stage-aware night pooling -> emb.npz (gate schema)

Everything stays on the box; embeddings are aggregate arrays keyed by bids_folder.
"""
import argparse
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
PATCH = 64                            # 1 s patches
N_PATCH = EPOCH_LEN // PATCH           # 30 tokens/epoch

# CAISR stage codes {1:N3,2:N2,3:N1,4:REM,5:Wake,9:Unknown}. Pool index order:
# Wake,N1,N2,N3,REM -> 0..4; anything else (Unknown) -> -1 (pooled only in global).
STAGE2IDX = {5: 0, 3: 1, 2: 2, 1: 3, 4: 4}
N_STAGE = 5
STAGE_TAGS = ["wake", "n1", "n2", "n3", "rem"]


# ============================================================= pack
def pack(raw_dir, out_prefix, progress=None):
    """Concatenate every per-recording npz into one float16 memmap + an index.

    Writes <out_prefix>_epochs.npy (memmap (N,6,1920) float16) and
    <out_prefix>_index.npz with per-night (bids/site/label/offsets) and per-epoch
    (night idx, stage-pool idx) arrays. Global-shuffle SSL then just fancy-indexes
    the memmap; night pooling slices by offset.
    """
    files = sorted(glob.glob(os.path.join(raw_dir, "*.npz")))
    if not files:
        raise SystemExit(f"no npz in {raw_dir}")
    night_bids, night_site, night_label, offsets = [], [], [], [0]
    stages_all = []
    metas = []
    n_total = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        ne = int(d["n_epoch"])
        if ne < 1:
            continue
        night_bids.append(str(d["bids"]))
        night_site.append(str(d["site"]))
        night_label.append(int(d["label"]))
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
        ep = np.load(f, allow_pickle=True)["epochs"]
        dat[ptr:ptr + ne] = ep
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


def build_epoch_encoder(d_model=128, depth=4, heads=4, dropout=0.1):
    torch, nn = _import_torch()

    class EpochEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            # channel-mixed 1 s patch embedding: (B,6,1920) -> (B,d,30)
            self.patch = nn.Conv1d(N_CH, d_model, kernel_size=PATCH, stride=PATCH)
            self.register_buffer("pe", _sinusoid(N_PATCH, d_model), persistent=False)
            layer = nn.TransformerEncoderLayer(
                d_model, heads, dim_feedforward=2 * d_model, dropout=dropout,
                batch_first=True, norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, depth)
            self.norm = nn.LayerNorm(d_model)
            self.d_model = d_model

        def tokens(self, x):
            return self.patch(x).transpose(1, 2)            # (B,30,d)

        def forward(self, x, mask_pos=None, mask_token=None):
            """x (B,6,1920). Masked patch tokens (if given) are replaced by
            mask_token BEFORE positional encoding (data2vec-style)."""
            t = self.tokens(x)
            if mask_pos is not None and mask_token is not None:
                t = torch.where(mask_pos.unsqueeze(-1), mask_token.to(t.dtype), t)
            t = t + self.pe.unsqueeze(0)
            h = self.enc(t)
            return self.norm(h)                             # (B,30,d)

    return EpochEncoder()


class RawJEPA:
    """Student epoch-encoder + EMA teacher + predictor MLP + mask token.
    Not an nn.Module so the teacher/mask_token stay out of any EMA deepcopy."""

    def __init__(self, d_model=128, depth=4, heads=4, dropout=0.1, seed=0):
        import copy
        torch, nn = _import_torch()
        torch.manual_seed(seed)
        self.torch, self.nn = torch, nn
        self.student = build_epoch_encoder(d_model, depth, heads, dropout)
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        self.d_model = d_model

    def trainable_params(self):
        return (list(self.student.parameters())
                + list(self.predictor.parameters()) + [self.mask_token])

    def train(self):
        self.student.train(); self.predictor.train()

    def eval(self):
        self.student.eval(); self.teacher.eval(); self.predictor.eval()

    def to(self, device):
        self.student.to(device); self.teacher.to(device); self.predictor.to(device)
        self.mask_token.data = self.mask_token.data.to(device)
        return self

    def ema_update(self, m):
        torch = self.torch
        with torch.no_grad():
            for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
                pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)
            for bs, bt in zip(self.student.buffers(), self.teacher.buffers()):
                bt.copy_(bs)

    def loss(self, x, mask_pos):
        torch, F = self.torch, self.torch.nn.functional
        with torch.no_grad():
            tgt = self.teacher(x)                           # (B,30,d)
            tgt = F.layer_norm(tgt, (self.d_model,))
        h = self.student(x, mask_pos, self.mask_token)
        pred = self.predictor(h)
        if mask_pos.sum() == 0:
            return None
        return F.smooth_l1_loss(pred[mask_pos], tgt[mask_pos])


def make_patch_mask(B, n_patch, mask_frac, span, rng):
    """(B,n_patch) bool, True=masked; contiguous spans within each row."""
    m = np.zeros((B, n_patch), dtype=bool)
    target = max(1, int(mask_frac * n_patch))
    for i in range(B):
        got, guard = 0, 0
        while got < target and guard < 4 * (target // max(1, span) + 1):
            guard += 1
            s = rng.randint(0, n_patch)
            e = min(n_patch, s + max(1, span))
            got += int((~m[i, s:e]).sum())
            m[i, s:e] = True
    return m


# ============================================================= pretraining
def pretrain(out_prefix, ckpt_out, *, exclude_site="", d_model=128, depth=4,
             heads=4, epochs=15, batch=256, lr=1.5e-3, wd=1e-4, mask_frac=0.5,
             span=4, ema0=0.996, ema1=0.9995, seed=0, device="cuda",
             amp=True, limit_nights=0, max_steps=0, progress=None):
    torch = __import__("torch")
    torch.manual_seed(seed)
    dat, idx = load_index(out_prefix)
    epoch_night = idx["epoch_night"]
    night_site = idx["night_site"]
    # eligible epochs = those whose night's site is not held out
    if exclude_site:
        held = set(np.where(night_site == exclude_site)[0].tolist())
        pool = np.where(~np.isin(epoch_night, list(held)))[0]
    else:
        pool = np.arange(dat.shape[0])
    if limit_nights:                                        # smoke: first K nights
        keep_n = set(range(min(limit_nights, len(night_site))))
        pool = pool[np.isin(epoch_night[pool], list(keep_n))]
    if progress:
        progress(f"pretrain on {len(pool)} epochs "
                 f"(exclude_site={exclude_site or 'NONE'}, device={device})")
    rng = np.random.RandomState(seed)
    model = RawJEPA(d_model, depth, heads, seed=seed).to(device)
    opt = torch.optim.AdamW(model.trainable_params(), lr=lr, weight_decay=wd)
    steps_per_epoch = max(1, len(pool) // batch)
    total_steps = epochs * steps_per_epoch if not max_steps else max_steps
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    use_amp = amp and device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    model.train()
    step, t0 = 0, time.time()
    stop = False
    for ep in range(epochs):
        order = rng.permutation(len(pool))
        ep_loss, ep_n = 0.0, 0
        for bi in range(steps_per_epoch):
            sel = pool[order[bi * batch:(bi + 1) * batch]]
            if len(sel) < 2:
                continue
            sel_sorted = np.sort(sel)                        # read locality on memmap
            xb = np.ascontiguousarray(dat[sel_sorted]).astype(np.float32)
            x = torch.from_numpy(xb).to(device, non_blocking=True)
            mk = torch.from_numpy(
                make_patch_mask(len(sel_sorted), N_PATCH, mask_frac, span, rng)
            ).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = model.loss(x, mk)
            if loss is None:
                continue
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
    """Per-epoch encode (mean over patch tokens) then stage-aware night pooling:
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

    # 1) encode every epoch -> per-epoch embedding (mean over 30 patch tokens)
    N = dat.shape[0]
    ep_emb = np.empty((N, d), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, N, batch):
            e = min(N, s + batch)
            xb = np.ascontiguousarray(dat[s:e]).astype(np.float32)
            x = torch.from_numpy(xb).to(device)
            with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
                h = enc(x)                                   # (b,30,d)
            ep_emb[s:e] = h.float().mean(dim=1).cpu().numpy()
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
        parts = [block.mean(axis=0)]                         # global
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
    t.add_argument("--epochs", type=int, default=15)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--d-model", type=int, default=128)
    t.add_argument("--depth", type=int, default=4)
    t.add_argument("--lr", type=float, default=1.5e-3)
    t.add_argument("--mask-frac", type=float, default=0.5)
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
    if args.cmd == "pack":
        pack(args.raw_dir, args.out_prefix, progress=_p)
    elif args.cmd == "pretrain":
        pretrain(args.prefix, args.out, exclude_site=args.exclude_site,
                 d_model=args.d_model, depth=args.depth, epochs=args.epochs,
                 batch=args.batch, lr=args.lr, mask_frac=args.mask_frac,
                 seed=args.seed, device=args.device, amp=not args.no_amp,
                 limit_nights=args.limit_nights, max_steps=args.max_steps,
                 progress=_p)
    elif args.cmd == "embed":
        embed(args.prefix, args.ckpt, args.out, batch=args.batch,
              device=args.device, progress=_p)


if __name__ == "__main__":
    main()
