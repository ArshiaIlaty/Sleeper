"""Temporal JEPA over per-epoch sleep-EEG feature sequences (Challenge 2026).

A JEPA-family self-supervised objective (data2vec-style latent masking with an
EMA target/"teacher" encoder), adapted to 1-D epoch sequences. It is squarely
joint-embedding-predictive: an EMA teacher produces *latent* targets, the student
predicts those latents at masked positions (no signal reconstruction, no
contrastive negatives).

Why this member of the JEPA family here:
  * The submission container is CPU-only and cannot regenerate external foundation
    embeddings (the SleepFM/GNN-fusion lesson). A tiny encoder we train and ship is
    the only learned representation that could ever run in-container.
  * The one lever that matters is label efficiency (n_pos=84). SSL pretrains on ALL
    unlabeled nights; only a small head needs the 84 positives.
  * Input reuses the already-exported per-epoch spectral features (no re-export).

Input per night: a sequence of ~900 epochs. Each token =
  16 continuous EEG features  +  a stage id (wake/n1/n2/n3/rem)
computed on the central derivation. Token embedding = Linear(16) + stage-embedding
+ sinusoidal positional encoding over epoch index.

The module is importable (load_nights / fit_norm / pretrain_encoder / embed_nights)
and has a CLI (`pretrain`, `embed`). Everything is CPU-friendly and deterministic
given a seed. No data leaves the box; embeddings are aggregate arrays keyed by
bids_folder.
"""
import argparse
import copy
import csv
import math
import sys
import time

import numpy as np

# ------------------------------------------------------------------ schema
# 16 continuous per-epoch features (order fixed; used as the model's input dims).
FEATURE_COLS = [
    "abs_delta", "abs_theta", "abs_alpha", "abs_sigma", "abs_beta",
    "rel_delta", "rel_theta", "rel_alpha", "rel_sigma", "rel_beta",
    "theta_alpha", "delta_sigma", "rem_slowing", "has_complexity",
    "sampen", "permen",
]
LOG_IDX = [0, 1, 2, 3, 4]                     # abs_* powers -> log1p before z-score
N_FEAT = len(FEATURE_COLS)

STAGE_MAP = {"wake": 0, "n1": 1, "n2": 2, "n3": 3, "rem": 4}
N_STAGE = 5
# pooled representations produced per night: global + these stages (n2, n3, rem)
POOL_STAGES = (2, 3, 4)

MAX_T = 256                                    # cap epochs/night (even-subsample longer):
                                               # attention is O(T^2); 256 tokens still span
                                               # the whole night (~100 s/token) and keep the
                                               # transformer CPU-tractable. Bump for Stage-2.
CLIP = 5.0                                     # z-score clip


# ============================================================= data loading
def _f(s):
    try:
        v = float(s)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def load_nights(csv_path, progress=None):
    """Read the per-epoch CSV into per-night sequences.

    Returns (nights, order) where nights[bids] = dict(feats (T,16) float32,
    stage (T,) int8, site str, label int) sorted by epoch_index and capped/
    subsampled to MAX_T; order is the bids_folder list in first-seen order.
    """
    col_idx = None
    raw = {}                                    # bids -> list[(epoch_index, feats, stage)]
    meta = {}                                   # bids -> (site, label)
    n = 0
    with open(csv_path, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        hpos = {c: i for i, c in enumerate(header)}
        col_idx = [hpos[c] for c in FEATURE_COLS]
        i_bids, i_site, i_lab = hpos["bids_folder"], hpos["site"], hpos["label"]
        i_stage, i_epi = hpos["stage"], hpos["epoch_index"]
        for row in rd:
            n += 1
            bids = row[i_bids]
            feats = np.array([_f(row[j]) for j in col_idx], dtype=np.float32)
            stg = STAGE_MAP.get(row[i_stage].strip().lower(), 0)
            try:
                epi = int(float(row[i_epi]))
            except (TypeError, ValueError):
                epi = len(raw.get(bids, []))
            raw.setdefault(bids, []).append((epi, feats, stg))
            if bids not in meta:
                lab = str(row[i_lab]).strip().lower()
                y = 1 if lab in ("1", "true", "yes") else 0
                meta[bids] = (row[i_site], y)
            if progress and n % 200000 == 0:
                progress(f"  read {n} rows, {len(raw)} nights")

    nights, order = {}, []
    for bids, rows in raw.items():
        rows.sort(key=lambda r: r[0])
        feats = np.vstack([r[1] for r in rows])
        stage = np.array([r[2] for r in rows], dtype=np.int8)
        if feats.shape[0] > MAX_T:              # evenly subsample very long nights
            keep = np.linspace(0, feats.shape[0] - 1, MAX_T).astype(int)
            feats, stage = feats[keep], stage[keep]
        site, y = meta[bids]
        nights[bids] = {"feats": feats, "stage": stage, "site": site, "label": y}
        order.append(bids)
    return nights, order


def fit_norm(nights, bids_subset):
    """Per-feature mean/std over the given nights' epochs (log1p on abs_* first)."""
    chunks = []
    for b in bids_subset:
        x = nights[b]["feats"].astype(np.float64).copy()
        x[:, LOG_IDX] = np.log1p(np.clip(x[:, LOG_IDX], 0, None))
        chunks.append(x)
    allx = np.vstack(chunks)
    mean = np.nanmean(allx, axis=0)
    std = np.nanstd(allx, axis=0)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    mean[~np.isfinite(mean)] = 0.0
    return mean.astype(np.float32), std.astype(np.float32)


def apply_norm(feats, mean, std):
    x = feats.astype(np.float32).copy()
    x[:, LOG_IDX] = np.log1p(np.clip(x[:, LOG_IDX], 0, None))
    x = (x - mean) / std
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(x, -CLIP, CLIP)


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


def build_encoder(d_model=64, depth=3, heads=4, dropout=0.1):
    torch, nn = _import_torch()

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(N_FEAT, d_model)
            self.stage_emb = nn.Embedding(N_STAGE, d_model)
            self.register_buffer("pe", _sinusoid(MAX_T, d_model), persistent=False)
            layer = nn.TransformerEncoderLayer(
                d_model, heads, dim_feedforward=2 * d_model, dropout=dropout,
                batch_first=True, norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, depth)
            self.norm = nn.LayerNorm(d_model)
            self.d_model = d_model

        def embed_tokens(self, feats, stage):
            return self.proj(feats) + self.stage_emb(stage)

        def forward(self, feats, stage, pad_mask, mask_pos=None, mask_token=None):
            """feats (B,T,16), stage (B,T), pad_mask (B,T) True=pad.
            If mask_pos/mask_token given, masked (non-pad) tokens are replaced by
            the learnable mask_token BEFORE positional encoding (data2vec-style)."""
            tok = self.embed_tokens(feats, stage)
            if mask_pos is not None and mask_token is not None:
                tok = torch.where(mask_pos.unsqueeze(-1), mask_token.to(tok.dtype), tok)
            T = tok.shape[1]
            tok = tok + self.pe[:T].unsqueeze(0)
            h = self.enc(tok, src_key_padding_mask=pad_mask)
            return self.norm(h)

    return Encoder()


class TSJEPA:
    """Wraps a student encoder, its EMA teacher, a predictor MLP and the mask token.
    Not an nn.Module itself so the teacher/mask_token stay out of any EMA deepcopy."""

    def __init__(self, d_model=64, depth=3, heads=4, dropout=0.1, seed=0):
        torch, nn = _import_torch()
        torch.manual_seed(seed)
        self.torch, self.nn = torch, nn
        self.student = build_encoder(d_model, depth, heads, dropout)
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

    @property
    def modules_to_device(self):
        return [self.student, self.teacher, self.predictor]

    def to(self, device):
        for m in self.modules_to_device:
            m.to(device)
        self.mask_token.data = self.mask_token.data.to(device)
        return self

    def ema_update(self, m):
        torch = self.torch
        with torch.no_grad():
            for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
                pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)
            for bs, bt in zip(self.student.buffers(), self.teacher.buffers()):
                bt.copy_(bs)

    def loss(self, feats, stage, pad_mask, mask_pos):
        """Predict layer-normed teacher latents at masked positions (smooth-L1)."""
        torch, F = self.torch, self.torch.nn.functional
        with torch.no_grad():
            tgt = self.teacher(feats, stage, pad_mask)              # (B,T,d)
            tgt = F.layer_norm(tgt, (self.d_model,))               # instance-norm targets
        h = self.student(feats, stage, pad_mask, mask_pos, self.mask_token)
        pred = self.predictor(h)
        sel = mask_pos & (~pad_mask)
        if sel.sum() == 0:
            return None
        return F.smooth_l1_loss(pred[sel], tgt[sel])


# ============================================================= masking / batching
def make_span_mask(lengths, Tmax, mask_frac=0.5, span=10, rng=None):
    """Boolean (B,Tmax) span mask, True=masked, only within each row's true length."""
    rng = rng or np.random.RandomState(0)
    m = np.zeros((len(lengths), Tmax), dtype=bool)
    for i, L in enumerate(lengths):
        if L <= 2:
            continue
        target = int(mask_frac * L)
        got = 0
        guard = 0
        while got < target and guard < 10 * (target // span + 1):
            guard += 1
            s = rng.randint(0, max(1, L - 1))
            e = min(L, s + max(1, span))
            newly = int((~m[i, s:e]).sum())
            m[i, s:e] = True
            got += newly
    return m


def collate(nights, bids_batch, mean, std):
    """Pad a batch to its max length. Returns feats,stage,pad_mask,lengths (numpy)."""
    seqs = [(apply_norm(nights[b]["feats"], mean, std), nights[b]["stage"])
            for b in bids_batch]
    lengths = [s[0].shape[0] for s in seqs]
    Tmax = max(lengths)
    B = len(seqs)
    feats = np.zeros((B, Tmax, N_FEAT), dtype=np.float32)
    stage = np.zeros((B, Tmax), dtype=np.int64)
    pad = np.ones((B, Tmax), dtype=bool)
    for i, (fx, st) in enumerate(seqs):
        L = fx.shape[0]
        feats[i, :L] = fx
        stage[i, :L] = st
        pad[i, :L] = False
    return feats, stage, pad, np.array(lengths)


# ============================================================= pretraining
def pretrain_encoder(nights, train_bids, *, d_model=64, depth=3, heads=4,
                     epochs=30, batch=32, lr=1e-3, wd=1e-4, mask_frac=0.5, span=10,
                     ema0=0.996, ema1=0.999, seed=0, device="cpu", progress=None):
    """Self-supervised pretraining on `train_bids` (labels unused). Returns
    (state_dict of the EMA teacher, norm_mean, norm_std)."""
    torch = __import__("torch")
    torch.manual_seed(seed)
    try:
        torch.set_num_threads(8)
    except Exception:
        pass
    rng = np.random.RandomState(seed)
    mean, std = fit_norm(nights, train_bids)
    model = TSJEPA(d_model, depth, heads, seed=seed).to(device)
    opt = torch.optim.AdamW(model.trainable_params(), lr=lr, weight_decay=wd)
    bids = list(train_bids)
    steps_per_epoch = max(1, len(bids) // batch)
    total_steps = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    step = 0
    model.train()
    for ep in range(epochs):
        rng.shuffle(bids)
        ep_loss, ep_n = 0.0, 0
        for bi in range(steps_per_epoch):
            bb = bids[bi * batch:(bi + 1) * batch]
            if len(bb) < 2:
                continue
            feats, stage, pad, lengths = collate(nights, bb, mean, std)
            mask = make_span_mask(lengths, feats.shape[1], mask_frac, span, rng)
            ft = torch.from_numpy(feats).to(device)
            st = torch.from_numpy(stage).to(device)
            pd = torch.from_numpy(pad).to(device)
            mk = torch.from_numpy(mask).to(device)
            loss = model.loss(ft, st, pd, mk)
            if loss is None:
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_params(), 3.0)
            opt.step()
            sched.step()
            m = ema0 + (ema1 - ema0) * (step / max(1, total_steps))
            model.ema_update(m)
            step += 1
            ep_loss += float(loss.item()); ep_n += 1
        if progress:
            progress(f"  epoch {ep+1}/{epochs} loss={ep_loss/max(1,ep_n):.4f}")
    state = {"teacher": {k: v.cpu() for k, v in model.teacher.state_dict().items()},
             "cfg": {"d_model": d_model, "depth": depth, "heads": heads},
             "mean": mean, "std": std}
    return state


# ============================================================= embedding
def _pool(latents, stage, length):
    """latents (T,d) numpy, stage (T,), length. Global masked-mean + per-stage means
    for POOL_STAGES; missing stage -> zeros. Returns (d*(1+len(POOL_STAGES)),)."""
    d = latents.shape[1]
    lat = latents[:length]
    stg = stage[:length]
    parts = [lat.mean(axis=0) if length > 0 else np.zeros(d, np.float32)]
    for s in POOL_STAGES:
        m = stg == s
        parts.append(lat[m].mean(axis=0) if m.any() else np.zeros(d, np.float32))
    return np.concatenate(parts).astype(np.float32)


def embed_nights(state, nights, bids_list, *, batch=32, device="cpu", progress=None):
    """Embed each night with the trained (EMA teacher) encoder. Returns
    (emb (N, d*(1+|POOL_STAGES|)) float32, cols list[str])."""
    torch = __import__("torch")
    cfg = state["cfg"]
    mean, std = state["mean"], state["std"]
    enc = build_encoder(cfg["d_model"], cfg["depth"], cfg["heads"], dropout=0.0)
    enc.load_state_dict(state["teacher"])
    enc.to(device).eval()
    d = cfg["d_model"]
    out = []
    with torch.no_grad():
        for bi in range(0, len(bids_list), batch):
            bb = bids_list[bi:bi + batch]
            feats, stage, pad, lengths = collate(nights, bb, mean, std)
            h = enc(torch.from_numpy(feats).to(device),
                    torch.from_numpy(stage.astype(np.int64)).to(device),
                    torch.from_numpy(pad).to(device)).cpu().numpy()
            for k, b in enumerate(bb):
                out.append(_pool(h[k], nights[b]["stage"], int(lengths[k])))
            if progress and (bi // batch) % 5 == 0:
                progress(f"  embedded {min(bi+batch,len(bids_list))}/{len(bids_list)}")
    pool_tags = ["glob"] + [f"st{s}" for s in POOL_STAGES]
    cols = [f"jepa_{tag}_{i}" for tag in pool_tags for i in range(d)]
    return np.vstack(out).astype(np.float32), cols


# ============================================================= CLI
def _progress(m):
    print(m, file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pretrain")
    p.add_argument("--csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--exclude-site", default="", help="hold this site OUT of pretraining")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")

    e = sub.add_parser("embed")
    e.add_argument("--csv", required=True)
    e.add_argument("--ckpt", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--device", default="cpu")

    args = ap.parse_args()
    torch = __import__("torch")

    if args.cmd == "pretrain":
        t0 = time.time()
        nights, order = load_nights(args.csv, progress=_progress)
        _progress(f"loaded {len(order)} nights in {time.time()-t0:.1f}s")
        train_bids = [b for b in order
                      if not (args.exclude_site and nights[b]["site"] == args.exclude_site)]
        _progress(f"pretrain on {len(train_bids)} nights "
                  f"(exclude_site={args.exclude_site or 'NONE'})")
        state = pretrain_encoder(nights, train_bids, d_model=args.d_model,
                                 depth=args.depth, epochs=args.epochs, batch=args.batch,
                                 seed=args.seed, device=args.device, progress=_progress)
        torch.save(state, args.out)
        _progress(f"saved encoder -> {args.out} ({time.time()-t0:.1f}s)")

    elif args.cmd == "embed":
        nights, order = load_nights(args.csv, progress=_progress)
        state = torch.load(args.ckpt, weights_only=False)
        emb, cols = embed_nights(state, nights, order, device=args.device,
                                 progress=_progress)
        sites = np.array([nights[b]["site"] for b in order])
        labels = np.array([nights[b]["label"] for b in order], dtype=int)
        np.savez_compressed(args.out, emb=emb, cols=np.array(cols),
                            bids=np.array(order), sites=sites, labels=labels)
        _progress(f"saved embeddings {emb.shape} -> {args.out}")


if __name__ == "__main__":
    main()
