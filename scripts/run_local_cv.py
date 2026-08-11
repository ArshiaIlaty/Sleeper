#!/usr/bin/env python3
"""Run LOSO + 70/15/15 on a LOCAL feature cache, through the production stack.

Thin standalone runner that avoids the S3 coupling in cross_validate_s3.py /
train_val_test_eval.py (which import s3_io/boto3 and upload results to a bucket we
can't reach from this account) while reusing the EXACT production model + scoring:

  * LOSO: per-fold HistGradientBoosting site mixture-of-experts + Kaiser fine-tune +
    BMI imputation (feature_prep.fit_site_models / fit_kaiser_finetuned /
    fit_bmi_imputer), scored with evaluate_model.compute_* and the reward at the
    training-prevalence decision rule (matches cross_validate_s3.loso_eval).
  * 70/15/15: model fit on train, reward thresholds fit on val, single locked test
    (mirrors train_val_test_eval.run, site_decade thresholds, Kaiser alt head).

Runs on two caches so we see the lift from the new features on identical footing:
  local_baseline  = team_code extract_all only
  local_plus      = extract_all + NK + clinical-report features

Usage (on pdmle):
  python3 run_local_cv.py --cache-dir /path/to/cache --repo /path/to/repo
"""
from __future__ import annotations

import argparse
import io
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

SITE_NAMES = {"S0001": "BIDMC", "I0002": "Emory", "I0006": "Kaiser"}


def load_cache(path):
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def safe(fn, *a):
    try:
        return float(fn(*a))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------- LOSO (production)
def loso(X, y, ages, sites, fp, ev, feat_names):
    """Mirror cross_validate_s3.loso_eval but with the full production site-MoE +
    Kaiser fine-tune + BMI imputer per fold (fit on the training sites only)."""
    all_true, all_prob, all_age = [], [], []
    fold_rows = []
    for site in np.unique(sites):
        te = sites == site
        tr = ~te
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        imp = fp.fit_bmi_imputer(X[tr], sites[tr], feat_names)
        Xtr = fp.apply_bmi_imputer(X[tr], sites[tr], imp)
        Xte = fp.apply_bmi_imputer(X[te], sites[te], imp)
        models = fp.fit_site_models(Xtr, y[tr], sites[tr])
        models = fp.fit_kaiser_finetuned(models, Xtr, y[tr], sites[tr])
        prob = fp.predict_with_kaiser_override(models, Xte, sites[te], use_kaiser_finetuned=True)
        binary = (prob > float(y[tr].mean())).astype(int)
        a2p = ev.compute_prevalence(ages[te], y, ages, gap=2)
        fr = safe(ev.compute_reward, y[te], binary, ages[te], a2p)
        fold_rows.append((str(site), int(te.sum()), fr,
                          safe(ev.compute_auroc_age, y[te], prob, ages[te], 2)))
        all_true.extend(y[te]); all_prob.extend(prob); all_age.extend(ages[te])

    yt = np.asarray(all_true); yp = np.asarray(all_prob); ya = np.asarray(all_age)
    a2p_all = ev.compute_prevalence(ya, y, ages, gap=2)
    bin_pi = (yp > float(y.mean())).astype(int)
    best_r, best_thr = float("-inf"), float(y.mean())
    for thr in [0.05, 0.076, 0.10, 0.15, float(y.mean())]:
        r = safe(ev.compute_reward, yt, (yp > thr).astype(int), ya, a2p_all)
        if r > best_r:
            best_r, best_thr = r, thr
    pooled = {
        "reward_at_pi": safe(ev.compute_reward, yt, bin_pi, ya, a2p_all),
        "best_reward": best_r, "best_threshold": best_thr,
        "auroc": safe(ev.compute_auroc, yt, yp),
        "auprc": safe(ev.compute_auprc, yt, yp),
        "age_auroc": safe(ev.compute_auroc_age, yt, yp, ya, 2),
        "age_weighted": safe(ev.compute_auroc_weighted, yt, yp, ya, 2),
        "accuracy": safe(ev.compute_accuracy, yt, bin_pi),
        "f_measure": safe(ev.compute_f_measure, yt, bin_pi),
    }
    return pooled, fold_rows


# ------------------------------------------------------------ 70/15/15 (production)
def stratify_key(y, sites):
    keys = np.array([f"{s}_{int(lab)}" for s, lab in zip(sites, y)], dtype=object)
    _, counts = np.unique(keys, return_counts=True)
    return y.astype(int) if counts.min() < 2 else keys


def make_splits(y, sites, train_frac=0.70, val_frac=0.15, seed=42):
    idx = np.arange(len(y))
    strata = stratify_key(y, sites)
    test_frac = 1.0 - train_frac - val_frac
    idx_tv, idx_te = train_test_split(idx, test_size=test_frac, random_state=seed, stratify=strata)
    val_of_tv = val_frac / (train_frac + val_frac)
    strata_tv = stratify_key(y[idx_tv], sites[idx_tv])
    idx_tr, idx_va = train_test_split(idx_tv, test_size=val_of_tv, random_state=seed, stratify=strata_tv)
    train = np.zeros(len(y), bool); val = np.zeros(len(y), bool); test = np.zeros(len(y), bool)
    train[idx_tr] = True; val[idx_va] = True; test[idx_te] = True
    return train, val, test


def tvt(X, y, ages, sites, fp, ev, feat_names, seed=42):
    """Mirror train_val_test_eval.run: fit on train, thresholds on val, locked test."""
    train, val, test = make_splits(y, sites, seed=seed)
    imp = fp.fit_bmi_imputer(X[train], sites[train], feat_names)
    Xtr = fp.apply_bmi_imputer(X[train], sites[train], imp)
    Xva = fp.apply_bmi_imputer(X[val], sites[val], imp)
    Xte = fp.apply_bmi_imputer(X[test], sites[test], imp)
    models = fp.fit_site_models(Xtr, y[train], sites[train])
    models = fp.fit_kaiser_finetuned(models, Xtr, y[train], sites[train])

    def predict(Xs, site_arr):
        return fp.predict_with_kaiser_override(models, Xs, site_arr, use_kaiser_finetuned=True)

    probs_va = predict(Xva, sites[val])
    reward_thr = fp.fit_reward_thresholds(y[val], probs_va, ages[val], sites[val], mode="site_decade")
    probs_te = predict(Xte, sites[test])
    bin_te = fp.apply_reward_thresholds(probs_te, ages[test], sites[test], reward_thr)
    prev_y, prev_ages = y[train], ages[train]
    a2p = ev.compute_prevalence(ages[test], prev_y, prev_ages, gap=2)
    return {
        "counts": {"train": int(train.sum()), "val": int(val.sum()), "test": int(test.sum())},
        "test": {
            "n": int(test.sum()), "prevalence": float(y[test].mean()),
            "n_pos_pred": int(bin_te.sum()),
            "reward": safe(ev.compute_reward, y[test], bin_te, ages[test], a2p),
            "reward_at_pi": safe(ev.compute_reward, y[test],
                                 (probs_te > float(prev_y.mean())).astype(int), ages[test], a2p),
            "auroc": safe(roc_auc_score, y[test], probs_te),
            "auprc": safe(average_precision_score, y[test], probs_te),
            "age_auroc": safe(ev.compute_auroc_age, y[test], probs_te, ages[test], 2),
            "age_weighted": safe(ev.compute_auroc_weighted, y[test], probs_te, ages[test], 2),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--seeds", default="42,1,7")
    args = ap.parse_args()

    if args.repo not in sys.path:
        sys.path.insert(0, args.repo)
    import feature_prep as fp
    import evaluate_model as ev

    seeds = [int(s) for s in args.seeds.split(",")]
    caches = {
        "baseline (team_code extract_all)": os.path.join(args.cache_dir, "feature_matrix_local_baseline.npz"),
        "plus (extract_all + NK + report)": os.path.join(args.cache_dir, "feature_matrix_local_plus.npz"),
    }

    print("=" * 78)
    print("LOSO (leave-one-site-out) — production site-MoE + Kaiser + BMI-impute")
    print("=" * 78)
    loso_out = {}
    for name, path in caches.items():
        c = load_cache(path)
        X = c["X"].astype(np.float32); y = c["y"].astype(int)
        ages = c["ages"].astype(float); sites = np.asarray(c["sites"])
        names = [str(n) for n in c["feature_names"]]
        pooled, folds = loso(X, y, ages, sites, fp, ev, names)
        loso_out[name] = pooled
        print(f"\n{name}  [{X.shape[0]}x{X.shape[1]}]")
        for s, n, r, aa in folds:
            print(f"    hold-out {SITE_NAMES.get(s,s):>6} n={n:4d}  reward={r:+.3f}  age-AUROC={aa:.3f}")
        print(f"  POOLED: reward@pi={pooled['reward_at_pi']:+.3f}  best={pooled['best_reward']:+.3f}"
              f"  AUROC={pooled['auroc']:.3f}  age-AUROC={pooled['age_auroc']:.3f}"
              f"  AUPRC={pooled['auprc']:.3f}")

    print("\n" + "=" * 78)
    print("70/15/15 (fit=train, thresholds=val, locked test) — mean over seeds", seeds)
    print("=" * 78)
    for name, path in caches.items():
        c = load_cache(path)
        X = c["X"].astype(np.float32); y = c["y"].astype(int)
        ages = c["ages"].astype(float); sites = np.asarray(c["sites"])
        names = [str(n) for n in c["feature_names"]]
        rows = [tvt(X, y, ages, sites, fp, ev, names, seed=s) for s in seeds]
        def mean(k): return float(np.nanmean([r["test"][k] for r in rows]))
        def std(k): return float(np.nanstd([r["test"][k] for r in rows]))
        print(f"\n{name}  [{X.shape[0]}x{X.shape[1]}]  test n≈{rows[0]['counts']['test']}")
        for k in ["reward", "reward_at_pi", "auroc", "age_auroc", "age_weighted", "auprc"]:
            print(f"    {k:<14} {mean(k):+.3f} ± {std(k):.3f}")

    print("\n" + "=" * 78)
    print("SUMMARY — pooled LOSO reward (primary leaderboard metric)")
    for name, p in loso_out.items():
        print(f"  {name:<38} reward@pi={p['reward_at_pi']:+.3f}  best={p['best_reward']:+.3f}")


if __name__ == "__main__":
    main()
