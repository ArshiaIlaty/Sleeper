#!/usr/bin/env python3
"""End-to-end supervised fine-tuning of the pretrained V2 EEG encoder (Challenge 2026 CI).

The 9 prior JEPA runs all used the encoder as a FROZEN feature extractor (pool CLS
descriptors -> night vector -> GBM) and found the rep redundant with handcrafted EEG.
This drops that assumption: initialise the encoder from the label-free V2 pretrain and
FINE-TUNE it end-to-end with the CI label, through a stage-aware attentive-pooling (MIL)
night head. Different failure mode -- the real risk here is overfitting 84 positives
across 3 sites under LOSO, so: freeze the low-level patch/chan embeddings, low encoder LR,
high head LR, dropout, class-balanced batch sampling, bag-resampled epochs (augmentation).

Produces OUT-OF-FOLD probabilities (each night predicted by a model that never trained on
it) for --cv site (LOSO) and --cv patient (LOPO). Pair against the champion in ft_eval.py.

NOTE: the encoder init was pretrained label-free on ALL nights, so a LOSO headline still
carries the mild "test-site distribution" leak of the shared-emb screen. As with V2, if
even this optimistic version shows no lift, a per-fold re-pretrain can only be worse.
"""
import argparse
import time

import numpy as np


def _import_torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_night_clf(ckpt, *, dropout=0.4, freeze_patch=True, device="cuda"):
    """V2 backbone (init from pretrained teacher) + stage-aware gated-attention pool + head."""
    torch, nn = _import_torch()
    from raw_eeg_jepa_v2 import build_epoch_encoder, N_STAGE
    state = torch.load(ckpt, weights_only=False)
    cfg = state["cfg"]
    backbone = build_epoch_encoder(cfg["d_model"], cfg["depth"], cfg["heads"], dropout=dropout)
    backbone.load_state_dict(state["teacher"])
    d = cfg["d_model"]

    class AttnPool(nn.Module):
        """Ilse et al. gated-attention MIL pooling over a night's epoch descriptors."""
        def __init__(self, d, dh=64):
            super().__init__()
            self.V = nn.Linear(d, dh)
            self.U = nn.Linear(d, dh)
            self.w = nn.Linear(dh, 1)

        def forward(self, h, mask):                          # h (B,K,d), mask (B,K) bool
            a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))   # (B,K,1)
            a = a.masked_fill(~mask.unsqueeze(-1), -1e4)
            a = torch.softmax(a, dim=1)
            return (a * h).sum(1)                            # (B,d)

    class NightClf(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.stage_emb = nn.Embedding(N_STAGE + 1, d)
            nn.init.normal_(self.stage_emb.weight, std=0.02)
            self.pool = AttnPool(d)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(d, 1)

        def epoch_desc(self, x):                             # x (M,6,1920) -> (M,d)
            return self.backbone.cls_embed(x)

        def from_desc(self, z, stage, mask):                 # z (B,K,d)
            z = z + self.stage_emb((stage + 1).clamp(0, N_STAGE))
            return self.head(self.drop(self.pool(z, mask))).squeeze(-1)   # (B,)

        def forward(self, x, stage, mask):                   # x (B,K,6,1920)
            B, K = x.shape[:2]
            z = self.epoch_desc(x.reshape(B * K, *x.shape[2:])).reshape(B, K, -1)
            return self.from_desc(z, stage, mask)

    model = NightClf().to(device)
    if freeze_patch:                                         # keep low-level SSL features fixed
        for p in model.backbone.patch.parameters():
            p.requires_grad_(False)
        model.backbone.chan_emb.weight.requires_grad_(False)
    return model, d


def _param_groups(model, enc_lr, head_lr, wd):
    enc, head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (enc if n.startswith("backbone.") else head).append(p)
    return [{"params": enc, "lr": enc_lr, "weight_decay": wd},
            {"params": head, "lr": head_lr, "weight_decay": wd}]


def train_fold(model, dat, nights, tr_idx, epoch_pool, *, epochs, k, batch,
               enc_lr, head_lr, wd, device, rng, progress=None, tag=""):
    """nights: list of (bids, site, label, a, b). tr_idx: indices into nights (train)."""
    torch, nn = _import_torch()
    y = np.array([nights[i][2] for i in tr_idx])
    pos = np.array(tr_idx)[y == 1]
    neg = np.array(tr_idx)[y == 0]
    opt = torch.optim.AdamW(_param_groups(model, enc_lr, head_lr, wd), betas=(0.9, 0.99))
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda"))
    bce = nn.BCEWithLogitsLoss()
    steps = max(1, len(tr_idx) // batch)
    half = max(1, batch // 2)
    model.train()
    for ep in range(epochs):
        losses = []
        for _ in range(steps):
            # class-balanced batch: half positive nights, half negative (with replacement)
            bi = np.concatenate([rng.choice(pos, half), rng.choice(neg, batch - half)])
            xb, sb, yb = _make_batch(dat, nights, bi, epoch_pool, k, rng, device)
            with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
                logit = model(xb, sb, torch.ones_like(sb, dtype=torch.bool))
                loss = bce(logit, yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 3.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss))
        if progress:
            progress(f"    {tag} ep{ep + 1}/{epochs} loss={np.mean(losses):.4f}")


def _make_batch(dat, nights, bi, epoch_pool, k, rng, device):
    torch, _ = _import_torch()
    B = len(bi)
    X = np.zeros((B, k, dat.shape[1], dat.shape[2]), np.float32)
    S = np.zeros((B, k), np.int64)
    Y = np.zeros(B, np.float32)
    for r, ni in enumerate(bi):
        _, _, lab, a, b = nights[ni]
        n = b - a
        sel = a + (rng.randint(0, n, k) if n >= 1 else np.zeros(k, int))
        X[r] = np.asarray(dat[sel], np.float32)
        S[r] = np.asarray(epoch_pool[sel], np.int64)
        Y[r] = lab
    return (torch.from_numpy(X).to(device), torch.from_numpy(S).to(device),
            torch.from_numpy(Y).to(device))


def predict_night(model, dat, night, epoch_pool, *, chunk=256, device="cuda"):
    """Encode ALL epochs of a night (no subsampling) -> attentive pool -> prob."""
    torch, _ = _import_torch()
    _, _, _, a, b = night
    n = b - a
    if n < 1:
        return 0.5
    model.eval()
    zs = []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
        for s in range(a, b, chunk):
            e = min(b, s + chunk)
            x = torch.from_numpy(np.asarray(dat[s:e], np.float32)).to(device)
            zs.append(model.epoch_desc(x).float())
        z = torch.cat(zs, 0).unsqueeze(0)                    # (1,n,d)
        st = torch.from_numpy(np.asarray(epoch_pool[a:b], np.int64)).to(device).unsqueeze(0)
        mask = torch.ones(1, n, dtype=torch.bool, device=device)
        logit = model.from_desc(z, st, mask)
    return float(torch.sigmoid(logit)[0])


def run(prefix, ckpt, out, *, cv="site", epochs=10, k=48, batch=16, enc_lr=3e-5,
        head_lr=5e-4, wd=0.05, dropout=0.4, n_splits=5, seed=0, device="cuda",
        progress=None):
    torch, _ = _import_torch()
    from raw_eeg_jepa_v2 import load_index
    dat, idx = load_index(prefix)
    offsets = idx["offsets"]
    epoch_pool = idx["epoch_pool"]
    nb, ns, nl = idx["night_bids"], idx["night_site"], idx["night_label"]
    nights = [(str(nb[i]), str(ns[i]), int(nl[i]), int(offsets[i]), int(offsets[i + 1]))
              for i in range(len(nb))]
    sites = np.array([x[1] for x in nights])
    labels = np.array([x[2] for x in nights])

    if cv == "site":
        folds = [(s, np.where(sites == s)[0]) for s in np.unique(sites)]
    else:
        from sklearn.model_selection import StratifiedGroupKFold
        groups = np.array([x[0] for x in nights])            # pid == bids (one rec/patient)
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = [(f"fold{f}", te) for f, (_, te) in
                 enumerate(sgkf.split(np.zeros(len(nights)), labels, groups=groups))]

    prob = np.full(len(nights), np.nan, np.float32)
    rng = np.random.RandomState(seed)
    t0 = time.time()
    for fname, te in folds:
        tr = np.setdiff1d(np.arange(len(nights)), te)
        if len(np.unique(labels[tr])) < 2:
            continue
        model, _ = build_night_clf(ckpt, dropout=dropout, device=device)
        if progress:
            progress(f"  [{cv}] fold {fname}: train={len(tr)} (pos {int(labels[tr].sum())}) "
                     f"test={len(te)} (pos {int(labels[te].sum())}) [{time.time()-t0:.0f}s]")
        train_fold(model, dat, nights, list(tr), epoch_pool, epochs=epochs, k=k, batch=batch,
                   enc_lr=enc_lr, head_lr=head_lr, wd=wd, device=device, rng=rng,
                   progress=progress, tag=fname)
        for i in te:
            prob[i] = predict_night(model, dat, nights[i], epoch_pool, device=device)
        del model
        torch.cuda.empty_cache()
    np.savez_compressed(out,
                        bids=np.array([x[0] for x in nights]),
                        sites=sites, labels=labels, prob=prob,
                        cv=cv, n_splits=n_splits, seed=seed)
    if progress:
        cov = int(np.isfinite(prob).sum())
        progress(f"saved OOF probs {cov}/{len(nights)} -> {out} ({time.time()-t0:.0f}s)")


def _p(m):
    print(m, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="packed")
    ap.add_argument("--ckpt", default="encoder_raw_v2_all.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cv", choices=["site", "patient"], default="site")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--enc-lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run(args.prefix, args.ckpt, args.out, cv=args.cv, epochs=args.epochs, k=args.k,
        batch=args.batch, enc_lr=args.enc_lr, head_lr=args.head_lr, wd=args.wd,
        dropout=args.dropout, n_splits=args.n_splits, seed=args.seed,
        device=args.device, progress=_p)


if __name__ == "__main__":
    main()
