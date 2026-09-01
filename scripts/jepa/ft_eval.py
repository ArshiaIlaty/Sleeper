#!/usr/bin/env python3
"""Evaluate end-to-end fine-tuned V2 encoder OOF probs against the champion.

Same fitter (feature_prep GBM + bmi imputer) + metric path (AC-AUROC / AUROC / reward)
+ paired-bootstrap protocol as jepa_gate_ab.py, so the verdict is directly comparable to
all 9 prior JEPA screens. Three arms on the SAME folds:

  champion   : 436 handcrafted features -> GBM
  jepa-ft    : the fine-tuned encoder's OOF probability, used directly (no refit)
  fusion     : [436 features, jepa_ft_oof_prob] -> GBM   (ft prob is a valid OOF feature)

Reward uses a per-arm prevalence-matched threshold (flag the top ~prevalence fraction) so
calibration differences between the GBM probs and the (balanced-trained) ft sigmoid don't
distort it; AC-AUROC (the primary, challenge-aligned metric) is threshold-free.

Usage (system python3, repo on PYTHONPATH):
  PYTHONPATH=<repo> python3 ft_eval.py --cache <cache.npz> --repo <repo> \
    --ftprob ftprob_site.npz --cv site --n-boot 2000
"""
import argparse
import sys
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


def load_cache(path):
    d = np.load(path, allow_pickle=True)
    return (d["X"].astype(np.float32), d["y"].astype(int), d["ages"].astype(float),
            np.asarray([str(s) for s in d["sites"]]),
            np.asarray([str(p) for p in d["pids"]]),
            [str(n) for n in d["feature_names"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ftprob", required=True)
    ap.add_argument("--cv", choices=["site", "patient"], default="site")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.repo not in sys.path:
        sys.path.insert(0, args.repo)
    import feature_prep as fp
    import evaluate_model as ev

    X, y, ages, sites, pids, names = load_cache(args.cache)
    ft = np.load(args.ftprob, allow_pickle=True)
    ftmap = {str(b): float(p) for b, p in zip(ft["bids"], ft["prob"]) if np.isfinite(p)}

    # align cache rows to nights with a finite ft OOF prob (pid == bids)
    keep = np.array([i for i, p in enumerate(pids) if p in ftmap])
    X, y, ages, sites, pids = X[keep], y[keep], ages[keep], sites[keep], pids[keep]
    ftp = np.array([ftmap[p] for p in pids], np.float32)
    print(f"aligned n={len(y)} y+={int(y.sum())} "
          f"sites={dict(zip(*np.unique(sites, return_counts=True)))} cv={args.cv}", flush=True)

    if args.cv == "site":
        folds = [sites == s for s in np.unique(sites)]
    else:
        from sklearn.model_selection import StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        ia = np.arange(len(y))
        folds = [np.isin(ia, te) for _, te in sgkf.split(X, y, groups=pids)]

    def gbm_oof(with_ft):
        oof = np.full(len(y), np.nan, np.float32)
        Xa = np.concatenate([X, ftp[:, None]], 1) if with_ft else X
        fnames = list(names) + (["jepa_ft_prob"] if with_ft else [])
        for te in folds:
            tr = ~te
            if len(np.unique(y[tr])) < 2:
                continue
            imp = fp.fit_bmi_imputer(Xa[tr], sites[tr], fnames)
            Xi_tr = fp.apply_bmi_imputer(Xa[tr], sites[tr], imp)
            Xi_te = fp.apply_bmi_imputer(Xa[te], sites[te], imp)
            clf = fp.fit_clf(Xi_tr, y[tr])
            oof[te] = clf.predict_proba(Xi_te)[:, 1]
        return oof

    champ = gbm_oof(False)
    fusion = gbm_oof(True)
    jepa = ftp
    ok = np.isfinite(champ) & np.isfinite(fusion) & np.isfinite(jepa)
    champ, fusion, jepa = champ[ok], fusion[ok], jepa[ok]
    yv, av = y[ok], ages[ok]
    a2p = ev.compute_prevalence(av, y, ages, gap=2)
    prev = float(yv.mean())

    def binarize(p):                      # flag the top ~prevalence fraction
        thr = np.quantile(p, 1.0 - prev)
        return (p >= thr).astype(int)

    def safe(fn, *a):
        try:
            return float(fn(*a))
        except Exception:
            return float("nan")

    def report(tag, p):
        aca = safe(ev.compute_auroc_age, yv, p, av, 2)
        auc = safe(roc_auc_score, yv, p)
        rew = safe(ev.compute_reward, yv, binarize(p), av, a2p)
        print(f"  {tag:<12} AC-AUROC={aca:.4f}  AUROC={auc:.4f}  reward@prev={rew:+.4f}", flush=True)

    cv_hdr = "LOSO (test site unseen)" if args.cv == "site" \
        else f"LOPO ({args.n_splits}-fold StratifiedGroupKFold by pid)"
    print(f"\n{cv_hdr}:", flush=True)
    report("champion", champ)
    report("jepa-ft", jepa)
    report("fusion", fusion)

    rng = np.random.RandomState(args.seed)
    n = len(yv)

    def boot(hi, lo):
        da, dr = [], []
        for _ in range(args.n_boot):
            ix = rng.randint(0, n, n)
            yb, ab = yv[ix], av[ix]
            if len(np.unique(yb)) < 2:
                continue
            p2 = ev.compute_prevalence(ab, y, ages, gap=2)
            da.append(safe(ev.compute_auroc_age, yb, hi[ix], ab, 2)
                      - safe(ev.compute_auroc_age, yb, lo[ix], ab, 2))
            dr.append(safe(ev.compute_reward, yb, binarize(hi)[ix], ab, p2)
                      - safe(ev.compute_reward, yb, binarize(lo)[ix], ab, p2))
        da = np.array([x for x in da if np.isfinite(x)])
        dr = np.array([x for x in dr if np.isfinite(x)])
        ci = lambda v: (v.mean(), np.percentile(v, 2.5), np.percentile(v, 97.5))
        return ci(da), ci(dr)

    print(f"\nPaired bootstrap n={args.n_boot}:", flush=True)
    ship = False
    for label, hi in [("fusion - champion", fusion), ("jepa-ft - champion", jepa)]:
        (ma, la, ha), (mr, lr, hr) = boot(hi, champ)
        print(f"  Δ[{label:<20}] AC-AUROC={ma:+.4f} [{la:+.4f},{ha:+.4f}]  "
              f"reward={mr:+.4f} [{lr:+.4f},{hr:+.4f}]", flush=True)
        if label == "fusion - champion":
            ship = (la > 0) or (la >= -1e-6 and mr > 0 and lr >= -1e-6)
    _cv = args.cv.upper()
    print(f"\nVERDICT: {f'SHIP -- fine-tuned JEPA beats champion under {_cv}-CV' if ship else f'DO NOT SHIP -- no {_cv}-CV improvement over champion'}", flush=True)
    print("FT_GATE_SHIP" if ship else "FT_GATE_NOSHIP", flush=True)


if __name__ == "__main__":
    main()
