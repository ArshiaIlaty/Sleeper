#!/usr/bin/env python3
"""Site-harmonization LOSO experiment — attack the flat cross-site transfer.

LOSO AC-AUROC has been stuck ~0.54-0.66 while within-distribution 80/20 reaches
0.83. That gap is domain shift: feature distributions differ across BIDMC / Emory
/ Kaiser (montage, population, CAISR calibration). Harmonization removes per-site
location/scale shifts so a model trained on two sites transfers to the third.

Three arms, all through the production stack (bmi-impute -> site-MoE + Kaiser ->
site_decade thresholds), differing ONLY in a feature transform applied AFTER bmi
imputation and BEFORE model fit:

  baseline    no transform (current production LOSO)
  adaptive    per-site robust z-score (median/IQR) using EACH site's own stats,
              INCLUDING the held-out test site. NOT deployable per-record (we'd
              need the test site's batch stats at inference) -- this is the
              CEILING / diagnostic: if even this is flat, shift isn't the problem.
  deployable  per-site robust z-score fit on TRAINING sites only; the unknown
              held-out site falls back to pooled-training stats. This IS
              shippable: store per-site center/scale in the model, apply by the
              record's site at inference (global fallback for unseen sites).

Robust (median/IQR) not mean/SD: PSG features are heavy-tailed; IQR resists the
outliers. Constant/degenerate features (IQR==0) are left untouched. Labels are
never used by the transform (no leakage beyond the stated per-arm site-stat use).

Run on the box against the plus cache (fast, no waveform pass; its core already
includes autonomic + the arch__ block is present too). Aggregate only.

Usage:
  PYTHONPATH=/data-temp/physio-viewer/bench/repo python3 harmonize_loso.py \
    --cache /data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz \
    --repo /data-temp/physio-viewer/bench/repo \
    --out /data-temp/physio-viewer/exports/harmonize_loso
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np


def robust_stats(Xsub):
    """Per-column median and IQR (75-25). IQR<=0 -> scale 1 (leave column as-is)."""
    med = np.nanmedian(Xsub, axis=0)
    q75 = np.nanpercentile(Xsub, 75, axis=0)
    q25 = np.nanpercentile(Xsub, 25, axis=0)
    iqr = q75 - q25
    scale = np.where(np.isfinite(iqr) & (iqr > 1e-9), iqr, 1.0)
    med = np.where(np.isfinite(med), med, 0.0)
    return med, scale


def harmonize(Xtr, sites_tr, Xte, sites_te, mode):
    """Return (Xtr_h, Xte_h). mode in {baseline, adaptive, deployable}."""
    if mode == "baseline":
        return Xtr, Xte
    Xtr_h = Xtr.copy()
    Xte_h = Xte.copy()
    # training sites: each scaled by its own stats (both arms do this)
    tr_stats = {}
    for s in np.unique(sites_tr):
        m = sites_tr == s
        med, scale = robust_stats(Xtr[m])
        tr_stats[s] = (med, scale)
        Xtr_h[m] = (Xtr[m] - med) / scale
    # pooled-training fallback for unseen sites
    gmed, gscale = robust_stats(Xtr)
    # test site
    for s in np.unique(sites_te):
        m = sites_te == s
        if mode == "adaptive":
            med, scale = robust_stats(Xte[m])           # test site's OWN stats (ceiling)
        else:  # deployable
            med, scale = tr_stats.get(s, (gmed, gscale))  # known-site stats or global
        Xte_h[m] = (Xte[m] - med) / scale
    return Xtr_h, Xte_h


def loso_harmonized(X, y, ages, sites, fp, ev, names, mode):
    from sklearn.metrics import roc_auc_score  # noqa: F401 (ev wraps its own)
    all_true, all_prob, all_age = [], [], []
    folds = []
    for site in np.unique(sites):
        te = sites == site
        tr = ~te
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        imp = fp.fit_bmi_imputer(X[tr], sites[tr], names)
        Xtr = fp.apply_bmi_imputer(X[tr], sites[tr], imp)
        Xte = fp.apply_bmi_imputer(X[te], sites[te], imp)
        Xtr, Xte = harmonize(Xtr, sites[tr], Xte, sites[te], mode)
        models = fp.fit_site_models(Xtr, y[tr], sites[tr])
        models = fp.fit_kaiser_finetuned(models, Xtr, y[tr], sites[tr])
        prob = fp.predict_with_kaiser_override(models, Xte, sites[te], use_kaiser_finetuned=True)
        a2p = ev.compute_prevalence(ages[te], y, ages, gap=2)
        binary = (prob > float(y[tr].mean())).astype(int)
        fr = ev.compute_reward(y[te], binary, ages[te], a2p)
        aa = ev.compute_auroc_age(y[te], prob, ages[te], 2)
        folds.append((str(site), int(te.sum()), float(fr), float(aa)))
        all_true.extend(y[te]); all_prob.extend(prob); all_age.extend(ages[te])
    yt, yp, ya = np.asarray(all_true), np.asarray(all_prob), np.asarray(all_age)
    a2p_all = ev.compute_prevalence(ya, y, ages, gap=2)
    bin_pi = (yp > float(y.mean())).astype(int)
    return {
        "reward_at_pi": float(ev.compute_reward(yt, bin_pi, ya, a2p_all)),
        "auroc": float(ev.compute_auroc(yt, yp)),
        "auprc": float(ev.compute_auprc(yt, yp)),
        "age_auroc": float(ev.compute_auroc_age(yt, yp, ya, 2)),
        "age_weighted": float(ev.compute_auroc_weighted(yt, yp, ya, 2)),
    }, folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.repo not in sys.path:
        sys.path.insert(0, args.repo)
    import feature_prep as fp
    import evaluate_model as ev
    try:
        from run_local_cv import SITE_NAMES
    except Exception:
        SITE_NAMES = {}

    c = np.load(args.cache, allow_pickle=True)
    X = c["X"].astype(np.float32); y = c["y"].astype(int)
    ages = c["ages"].astype(float); sites = np.asarray([str(s) for s in c["sites"]])
    names = [str(n) for n in c["feature_names"]]
    print(f"cache: n={len(y)} pos={int(y.sum())} feats={X.shape[1]} "
          f"sites={dict(zip(*np.unique(sites, return_counts=True)))}", flush=True)

    results = {}
    for mode in ["baseline", "adaptive", "deployable"]:
        pooled, folds = loso_harmonized(X, y, ages, sites, fp, ev, names, mode)
        results[mode] = dict(pooled=pooled, folds=folds)
        print(f"\n[{mode}]", flush=True)
        for s, n, r, aa in folds:
            print(f"    hold-out {SITE_NAMES.get(s, s):>6} n={n:5d}  reward={r:+.3f}  AC-AUROC={aa:.3f}", flush=True)
        print(f"    POOLED  AC-AUROC={pooled['age_auroc']:.3f}  AUROC={pooled['auroc']:.3f}  "
              f"reward@pi={pooled['reward_at_pi']:+.3f}  AUPRC={pooled['auprc']:.3f}", flush=True)

    b = results["baseline"]["pooled"]
    print("\n" + "=" * 78, flush=True)
    print("DELTA vs baseline (positive = harmonization HELPS cross-site)", flush=True)
    print("=" * 78, flush=True)
    for mode in ["adaptive", "deployable"]:
        p = results[mode]["pooled"]
        print(f"  {mode:<11} AC-AUROC Δ={p['age_auroc']-b['age_auroc']:+.3f}   "
              f"AUROC Δ={p['auroc']-b['auroc']:+.3f}   reward@pi Δ={p['reward_at_pi']-b['reward_at_pi']:+.3f}", flush=True)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "harmonize_loso_summary.json"), "w") as fh:
            json.dump({"n": int(len(y)), "pos": int(y.sum()), "results": results}, fh, indent=2)
        print(f"\nwrote harmonize_loso_summary.json to {args.out}", flush=True)
    print("DONE_HARMONIZE_LOSO", flush=True)


if __name__ == "__main__":
    main()
