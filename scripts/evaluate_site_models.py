#!/usr/bin/env python3
"""LOSO evaluation: global vs BMI imputation vs site-specific (deployed) models."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, SCRIPTS, os.path.join(ROOT, "claude")):
    if p not in sys.path:
        sys.path.insert(0, p)

from cross_validate_s3 import SITE_NAMES  # noqa: E402
from evaluate_model import compute_auroc_age, compute_prevalence, compute_reward  # noqa: E402
from extract_features_s3 import load_feature_cache  # noqa: E402
from feature_presets import select_indices  # noqa: E402
from feature_prep import apply_bmi_imputer, fit_bmi_imputer, fit_site_models, predict_site_model  # noqa: E402
from s3_io import BUCKET, FEATURE_PREFIX, upload_bytes  # noqa: E402


def _mask_preset(X, names, preset):
    keep = select_indices(names, preset)
    out = np.full_like(X, np.nan, dtype=np.float32)
    out[:, keep] = X[:, keep]
    return out


def _pooled_metrics(y, prob, ages, train_y):
    age_to_prev = compute_prevalence(ages, y, ages, gap=2)
    pi = float(train_y.mean())
    return {
        "plain_auroc": float(roc_auc_score(y, prob)),
        "auprc": float(average_precision_score(y, prob)),
        "age_auroc": float(compute_auroc_age(y, prob, ages, gap=2)),
        "reward_at_pi": float(compute_reward(y, (prob > pi).astype(int), ages, age_to_prev)),
    }


def loso_eval(X, y, ages, sites, feature_names, *, impute_bmi: bool):
    probs = np.zeros(len(y), dtype=np.float64)
    for site in np.unique(sites):
        te = sites == site
        tr = ~te
        Xtr, Xte = X[tr].copy(), X[te].copy()
        str_ = sites[tr]
        ste = sites[te]
        if impute_bmi:
            imp = fit_bmi_imputer(Xtr, str_, feature_names)
            Xtr = apply_bmi_imputer(Xtr, str_, imp)
            Xte = apply_bmi_imputer(Xte, ste, imp)
        models = fit_site_models(Xtr, y[tr], str_)
        probs[te] = predict_site_model(models, Xte, str(site))
    return _pooled_metrics(y, probs, ages, y)


def site_stratified_oof(X, y, sites, feature_names, *, impute_bmi: bool, n_splits: int = 5):
    """Within-site CV using site-specific models (deployment scenario)."""
    probs = np.full(len(y), np.nan, dtype=np.float64)
    for site in np.unique(sites):
        m = sites == site
        Xs, ys = X[m], y[m]
        n_pos, n_neg = int(ys.sum()), int((1 - ys).sum())
        if n_pos < 2 or n_neg < 2 or m.sum() < n_splits * 2:
            continue
        n_folds = min(n_splits, n_pos, n_neg)
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        site_arr = np.full(len(ys), str(site))
        for tr_i, te_i in skf.split(Xs, ys):
            Xtr, Xte = Xs[tr_i].copy(), Xs[te_i]
            if impute_bmi:
                imp = fit_bmi_imputer(Xtr, site_arr[tr_i], feature_names)
                Xtr = apply_bmi_imputer(Xtr, site_arr[tr_i], imp)
                Xte = apply_bmi_imputer(Xte, site_arr[te_i], imp)
            models = fit_site_models(Xtr, ys[tr_i], site_arr[tr_i], min_site_n=20)
            idx = np.where(m)[0]
            probs[idx[te_i]] = predict_site_model(models, Xte, str(site))
    mask = np.isfinite(probs)
    if mask.sum() < 10:
        return {"plain_auroc": float("nan"), "n": int(mask.sum())}
    return {"plain_auroc": float(roc_auc_score(y[mask], probs[mask])), "n": int(mask.sum())}


def run(version: str = "v3", preset: str = "submit") -> dict:
    cache = load_feature_cache(version)
    feature_names = [str(n) for n in cache["feature_names"]]
    X = _mask_preset(cache["X"].astype(np.float32), feature_names, preset)
    y = cache["y"].astype(int)
    ages = cache["ages"].astype(float)
    sites = np.asarray(cache["sites"])

    print(f"=== Site model evaluation ({version}, preset={preset}) ===\n")
    rows = {}
    for label, impute in (("global", False), ("bmi_impute", True)):
        m = loso_eval(X, y, ages, sites, feature_names, impute_bmi=impute)
        rows[label] = m
        print(
            f"LOSO {label:14s} AUROC={m['plain_auroc']:.3f}  "
            f"age-AUROC={m['age_auroc']:.3f}  reward={m['reward_at_pi']:+.3f}"
        )

    oof = site_stratified_oof(X, y, sites, feature_names, impute_bmi=True)
    rows["site_stratified_oof"] = oof
    print(f"\nSite-stratified OOF (deployed site models + BMI impute): "
          f"AUROC={oof['plain_auroc']:.3f}  n={oof['n']}")

    print("\nPer-site LOSO (BMI impute + global fallback for held-out site):")
    for hold in sorted(np.unique(sites)):
        te = sites == hold
        tr = ~te
        imp = fit_bmi_imputer(X[tr], sites[tr], feature_names)
        Xtr = apply_bmi_imputer(X[tr], sites[tr], imp)
        Xte = apply_bmi_imputer(X[te], sites[te], imp)
        models = fit_site_models(Xtr, y[tr], sites[tr])
        prob = predict_site_model(models, Xte, str(hold))
        try:
            auroc = float(roc_auc_score(y[te], prob))
        except Exception:
            auroc = float("nan")
        print(f"  {hold} ({SITE_NAMES.get(str(hold), hold)}): AUROC={auroc:.3f}  n={te.sum()}")

    summary = {
        "version": version,
        "preset": preset,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": rows,
    }
    s3_key = f"{FEATURE_PREFIX.rstrip('/')}/site_models_{version}_{preset}.json"
    upload_bytes(BUCKET, s3_key, json.dumps(summary, indent=2).encode(), "application/json")
    out = os.path.join(ROOT, "eda", f"site_models_{version}_{preset}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out}")
    print(f"Saved: s3://{BUCKET}/{s3_key}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v3")
    ap.add_argument("--preset", default="submit")
    args = ap.parse_args()
    run(args.version, args.preset)
