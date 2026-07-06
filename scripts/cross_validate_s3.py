#!/usr/bin/env python3
"""Leave-one-site-out CV on S3-cached features (official Challenge metrics).

Usage:
  python scripts/cross_validate_s3.py --version v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluate_model import compute_auroc_age, compute_prevalence, compute_reward  # noqa: E402
from extract_features_s3 import load_feature_cache  # noqa: E402
from s3_io import BUCKET, FEATURE_PREFIX, upload_bytes  # noqa: E402

SITE_NAMES = {
    "S0001": "BIDMC",
    "I0002": "Emory",
    "I0006": "Kaiser",
}


def fit_clf(Xtr: np.ndarray, ytr: np.ndarray):
    base = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    n_min = int(np.bincount(ytr).min()) if len(np.unique(ytr)) > 1 else 0
    if n_min >= 3 and len(ytr) >= 30:
        clf = CalibratedClassifierCV(base, method="isotonic", cv=min(3, n_min))
        clf.fit(Xtr, ytr)
    else:
        base.fit(Xtr, ytr)
        clf = base
    return clf


def safe_auroc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def safe_auprc(y, p):
    try:
        return float(average_precision_score(y, p))
    except Exception:
        return float("nan")


def loso_eval(
    X: np.ndarray,
    y: np.ndarray,
    ages: np.ndarray,
    sites: np.ndarray,
    *,
    verbose: bool = True,
) -> dict:
    """Run LOSO CV; return pooled OOF metrics and per-fold rows."""
    fold_rows = []
    all_true, all_prob, all_age = [], [], []

    for site in np.unique(sites):
        te = sites == site
        tr = ~te
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue

        clf = fit_clf(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        binary = (prob > float(y[tr].mean())).astype(int)
        age_to_prev = compute_prevalence(ages[te], y, ages, gap=2)
        try:
            age_auroc = float(compute_auroc_age(y[te], prob, ages[te], gap=2))
        except Exception:
            age_auroc = float("nan")
        try:
            reward = float(compute_reward(y[te], binary, ages[te], age_to_prev))
        except Exception:
            reward = float("nan")

        row = {
            "site": str(site),
            "site_name": SITE_NAMES.get(str(site), str(site)),
            "n_test": int(te.sum()),
            "plain_auroc": safe_auroc(y[te], prob),
            "auprc": safe_auprc(y[te], prob),
            "age_auroc": age_auroc,
            "reward_at_pi_train": reward,
        }
        fold_rows.append(row)
        if verbose:
            print(
                f"  hold-out {site} ({row['site_name']:>6}): n={row['n_test']:3d}  "
                f"AUROC={row['plain_auroc']:.3f}  age-AUROC={row['age_auroc']:.3f}  "
                f"reward={row['reward_at_pi_train']:+.3f}"
            )
        all_true.extend(y[te])
        all_prob.extend(prob)
        all_age.extend(ages[te])

    all_true = np.asarray(all_true)
    all_prob = np.asarray(all_prob)
    all_age = np.asarray(all_age)
    age_to_prev_all = compute_prevalence(ages, y, ages, gap=2)
    best_thr, best_reward = 0.076, float("-inf")
    thr_results = {}
    for thr in [0.05, 0.076, 0.10, 0.15, float(y.mean())]:
        binary = (all_prob > thr).astype(int)
        r = float(compute_reward(all_true, binary, all_age, age_to_prev_all))
        thr_results[str(round(thr, 4))] = r
        if r > best_reward:
            best_reward, best_thr = r, thr

    pooled = {
        "plain_auroc": safe_auroc(all_true, all_prob),
        "auprc": safe_auprc(all_true, all_prob),
        "age_auroc": float(compute_auroc_age(all_true, all_prob, all_age, gap=2)),
        "reward_at_pi": float(compute_reward(
            all_true, (all_prob > float(y.mean())).astype(int), all_age, age_to_prev_all)),
        "best_reward": best_reward,
        "best_threshold": best_thr,
        "reward_by_threshold": thr_results,
    }
    return {"folds": fold_rows, "pooled": pooled}


def run(version: str = "v1", out_dir: str | None = None) -> dict:
    cache = load_feature_cache(version)
    X = cache["X"].astype(np.float32)
    y = cache["y"].astype(int)
    ages = cache["ages"].astype(float)
    sites = np.asarray(cache["sites"])

    print(f"Loaded s3 feature cache v{version}: {X.shape[0]} pts, {X.shape[1]} feats, "
          f"prevalence={y.mean():.3f}")
    print(f"Sites: {dict(zip(*np.unique(sites, return_counts=True)))}\n")

    result = loso_eval(X, y, ages, sites, verbose=True)
    fold_rows = result["folds"]
    pooled = result["pooled"]

    print(f"\nPooled LOSO plain-AUROC = {pooled['plain_auroc']:.3f}")
    print(f"Pooled LOSO AUPRC       = {pooled['auprc']:.3f}")
    print(f"Pooled LOSO age-AUROC   = {pooled['age_auroc']:.3f}")

    print("\nReward vs threshold (pooled OOF):")
    for thr, r in pooled["reward_by_threshold"].items():
        tag = "  <- overall prevalence" if abs(float(thr) - y.mean()) < 0.01 else ""
        print(f"  thr={float(thr):.3f}: reward={r:+.3f}{tag}")

    summary = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_patients": int(len(y)),
        "n_features": int(X.shape[1]),
        "prevalence": float(y.mean()),
        "folds": fold_rows,
        "pooled": pooled,
    }

    out_dir = out_dir or os.path.join(ROOT, "eda")
    try:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"loso_cv_{version}.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved: {out_path}")
    except OSError as exc:
        print(f"\nLocal save skipped ({exc}); writing to S3 instead.")
        out_path = None

    s3_key = f"{FEATURE_PREFIX.rstrip('/')}/loso_cv_{version}.json"
    upload_bytes(BUCKET, s3_key, json.dumps(summary, indent=2).encode("utf-8"), "application/json")
    print(f"Saved: s3://{BUCKET}/{s3_key}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="LOSO cross-validation on S3 feature cache.")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    run(version=args.version, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
