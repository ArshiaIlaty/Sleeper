#!/usr/bin/env python3
"""Batch B — DANN-MLP: learned site-invariant representation via a gradient-reversal site head.

The user's proposal: force the encoder to learn CI-relevant physiology that is USELESS for predicting
site (domain-adversarial / DANN). Our SleepFM-embedding probe found site was 96-98% RF-decodable;
this tests whether an ADVERSARIAL objective removes that shortcut from a representation we can
actually build, and whether removing it lifts LOSO transfer.

Design decisions grounded in the prior work:
  * Input = per-site RANK-normed features (transductive/label-free -> hidden-site-legal; already in
    [0,1] so NN-friendly). Rank-norm is our winning deployable batch-effect remover; DANN is tested
    as a lever ON TOP of it (does a LEARNED invariance beat the hand-crafted rank + the blunt
    variance-ratio feature drop?).
  * lambda sweep {0, 0.01, 0.1, 0.3, 1.0}. lambda=0 is the NON-adversarial MLP -> isolates the
    adversarial delta from the (expected) tree-vs-MLP gap. Compare all vs the GBM champion anchor.
  * GRL: forward identity, backward multiply by -lambda (ramped 0->target over training, the standard
    DANN schedule for stability). Site head is 2-way per fold (the two TRAINING sites; held-out site
    contributes no label and is never seen by the site head -> LOSO-honest).
  * 3-seed probability ENSEMBLE per fold to damp NN variance; metrics + all decision rules
    (levers_test.decision_rules) on the pooled OOF.
  * Site-decodability probe: RF 3-way site accuracy on (raw feats, rank feats, and the learned
    embedding z at each lambda) -> the "RF site accuracy before/after" the user asked for.

Caveats carried in the writeup: only 3 sites (site head sees 2 -> data-starved adversary; encoder
can overfit); metric is age-adjusted so we watch AC-AUROC not just AUROC.

Run on pdmle with the torch venv:  /tmp/torchvenv/bin/python (torch) + PYTHONPATH for sklearn/pandas/
scipy/feature_prep/evaluate_model/levers_test. Aggregate metrics only.

Usage:
  PYTHONPATH=<USP>:<repo>:<levers-dir> /tmp/torchvenv/bin/python dann_mlp.py \
      --exports <exports> --repo <repo> --levers-dir <dir> --cohort large
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

LAMBDAS = [0.0, 0.01, 0.1, 0.3, 1.0]
SEEDS = [0, 1, 2]


def site_probe(Xin, sites, seed=0):
    """RF 3-way site-decodability: (accuracy, balanced_accuracy). Median-imputed."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    Xf = np.asarray(Xin, float)
    med = np.nanmedian(Xf, 0); med = np.where(np.isfinite(med), med, 0.0)
    Xf = np.where(np.isfinite(Xf), Xf, med)
    clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed,
                                 class_weight="balanced")
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pred = cross_val_predict(clf, Xf, sites, cv=cv, n_jobs=None)
    return accuracy_score(sites, pred), balanced_accuracy_score(sites, pred)


def build_torch():
    import torch
    import torch.nn as nn

    class GRL(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lambd):
            ctx.lambd = float(lambd)
            return x.view_as(x)

        @staticmethod
        def backward(ctx, g):
            return -ctx.lambd * g, None

    class DANN(nn.Module):
        def __init__(self, d_in, n_site, emb=64, h=(256, 128), p=0.3):
            super().__init__()
            layers, d = [], d_in
            for hh in h:
                layers += [nn.Linear(d, hh), nn.BatchNorm1d(hh), nn.ReLU(), nn.Dropout(p)]
                d = hh
            layers += [nn.Linear(d, emb), nn.ReLU()]
            self.enc = nn.Sequential(*layers)
            self.ci = nn.Sequential(nn.Linear(emb, 32), nn.ReLU(), nn.Linear(32, 1))
            self.site = nn.Sequential(nn.Linear(emb, 32), nn.ReLU(), nn.Linear(32, n_site))

        def forward(self, x, lambd=0.0):
            z = self.enc(x)
            return self.ci(z).squeeze(1), self.site(GRL.apply(z, lambd)), z

    return torch, nn, DANN


def train_one(torch, nn, DANN, Xtr, ytr, str_site_tr, Xval, yval, lambd, seed,
              epochs=150, bs=256, patience=25):
    """Train encoder+CI+site heads on TRAIN (2 sites). Early stop on val AUROC. Returns model."""
    from sklearn.metrics import roc_auc_score
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device("cpu")
    site_codes = {s: i for i, s in enumerate(sorted(np.unique(str_site_tr)))}
    s_tr = np.array([site_codes[s] for s in str_site_tr])
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    st = torch.tensor(s_tr, dtype=torch.long)
    Xv = torch.tensor(Xval, dtype=torch.float32)
    model = DANN(Xtr.shape[1], len(site_codes)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_w = torch.tensor([max((ytr == 0).sum(), 1) / max((ytr == 1).sum(), 1)], dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    ce = nn.CrossEntropyLoss()
    n = Xtr.shape[0]
    best_auc, best_state, bad = -1.0, None, 0
    for ep in range(epochs):
        model.train()
        p = ep / max(epochs - 1, 1)
        lam = lambd * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)   # DANN ramp 0->lambda
        perm = np.random.permutation(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            ci_logit, site_logit, _ = model(Xt[idx], lam)
            loss = bce(ci_logit, yt[idx]) + ce(site_logit, st[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xv, 0.0)[0]).numpy()
        try:
            auc = roc_auc_score(yval, pv) if len(np.unique(yval)) > 1 else 0.5
        except ValueError:
            auc = 0.5
        if auc > best_auc + 1e-4:
            best_auc, best_state, bad = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cvdir")
    ap.add_argument("--levers-dir", required=True)
    ap.add_argument("--cohort", default="large")
    args = ap.parse_args()
    for p in [args.repo, args.cvdir, args.levers_dir]:
        if p and p not in sys.path:
            sys.path.insert(0, p)
    import feature_prep as fp
    import evaluate_model as ev
    import levers_test as lv
    torch, nn, DANN = build_torch()
    from sklearn.model_selection import train_test_split

    X, y, ages, sites, names, bmi_idx = lv.load_csv(args.exports, args.cohort)
    protect = [bmi_idx] if bmi_idx is not None else []
    print(f"cohort={args.cohort} n={len(y)} pos={int(y.sum())} feats={X.shape[1]} "
          f"sites={ {s: int((sites == s).sum()) for s in np.unique(sites)} }")

    Xr = lv.site_rank(X, sites, protect)
    Xr = np.where(np.isfinite(Xr), Xr, 0.5).astype(np.float32)   # neutral rank for missing

    # ---- anchor: GBM champion + site-decodability of raw vs rank feats ----
    pooled_g, _, oof_g = lv.run_loso(Xr, y, ages, sites, fp, ev, names, model="hgb")
    dr_g = lv.decision_rules(oof_g, y, ages, ev)
    acc_raw, bal_raw = site_probe(np.where(np.isfinite(X), X, np.nan), sites)
    acc_rk, bal_rk = site_probe(Xr, sites)
    maj = max(np.bincount([{"S0001": 0, "I0002": 1, "I0006": 2}.get(s, 0) for s in sites])) / len(sites)
    print("\n" + "=" * 100)
    print("ANCHORS")
    print("=" * 100)
    print(f"  GBM champion (rank feats):  AC-AUROC={pooled_g['age_auroc']:.4f} "
          f"AUROC={pooled_g['auroc']:.4f} AUPRC={pooled_g['auprc']:.4f} "
          f"rwd@pi={dr_g['pi']:+.4f} transfer={dr_g['transfer']:+.4f}")
    print(f"  RF site-decodability (3-way; majority={maj:.3f}, balanced-chance=0.333):")
    print(f"     raw feats : acc={acc_raw:.3f}  balanced_acc={bal_raw:.3f}")
    print(f"     rank feats: acc={acc_rk:.3f}  balanced_acc={bal_rk:.3f}")

    # ---- DANN lambda sweep ----
    print("\n" + "=" * 100)
    print("DANN-MLP  —  LOSO (3-seed ensemble), site head = 2 training sites via GRL")
    print("=" * 100)
    print(f"{'lambda':>7}{'AC-AUROC':>10}{'AUROC':>9}{'AUPRC':>9}{'rwd@pi':>9}{'transfer':>10}"
          f"{'oracle':>9}{'z_site_acc':>12}")
    print("-" * 100)
    uniq_sites = np.unique(sites)
    for lam in LAMBDAS:
        oof_prob = np.full(len(y), np.nan)
        osite = np.array(sites, dtype=object)
        for s in uniq_sites:
            te = sites == s; tr = ~te
            if te.sum() == 0 or len(np.unique(y[tr])) < 2:
                continue
            # train/val split within the 2 training sites, stratified by y
            idx_tr = np.where(tr)[0]
            tr_i, val_i = train_test_split(idx_tr, test_size=0.15, random_state=0,
                                           stratify=y[idx_tr])
            probs_seeds = []
            for sd in SEEDS:
                model = train_one(torch, nn, DANN, Xr[tr_i], y[tr_i], sites[tr_i],
                                  Xr[val_i], y[val_i], lam, sd)
                with torch.no_grad():
                    probs_seeds.append(
                        torch.sigmoid(model(torch.tensor(Xr[te], dtype=torch.float32), 0.0)[0]).numpy())
            oof_prob[te] = np.mean(probs_seeds, axis=0)
        # metrics via levers plumbing
        yt, yp, ya = y, oof_prob, ages
        a2p = ev.compute_prevalence(ya, y, ages, gap=2)
        pooled = {"age_auroc": float(ev.compute_auroc_age(yt, yp, ya, 2)),
                  "auroc": float(ev.compute_auroc(yt, yp)),
                  "auprc": float(ev.compute_auprc(yt, yp))}
        dr = lv.decision_rules((osite, yt, yp, ya), y, ages, ev)
        # embedding site-decodability: one all-data encoder at this lambda
        idx_all = np.arange(len(y))
        tr_i, val_i = train_test_split(idx_all, test_size=0.15, random_state=0, stratify=y)
        mall = train_one(torch, nn, DANN, Xr[tr_i], y[tr_i], sites[tr_i], Xr[val_i], y[val_i], lam, 0)
        with torch.no_grad():
            Z = mall(torch.tensor(Xr, dtype=torch.float32), 0.0)[2].numpy()
        z_acc, _ = site_probe(Z, sites)
        print(f"{lam:>7.2f}{pooled['age_auroc']:>10.4f}{pooled['auroc']:>9.4f}{pooled['auprc']:>9.4f}"
              f"{dr['pi']:>+9.4f}{dr['transfer']:>+10.4f}{dr['oracle']:>+9.4f}{z_acc:>12.3f}")

    print("\nDONE_DANN")


if __name__ == "__main__":
    main()
