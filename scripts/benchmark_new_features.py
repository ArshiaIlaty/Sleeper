#!/usr/bin/env python3
"""Benchmark the new signal features against the baseline on the standard cohort.

Reproduces Team SDG's official LOSO methodology (scripts/cross_validate_s3.py):
per-fold HistGradientBoosting + isotonic calibration, leave-one-site-out, scored
with the Challenge's own evaluate_model.compute_* functions. But instead of the S3
feature cache it reads the three LOCAL standard-cohort CSVs and compares feature
sets so we can see whether the new features move the primary Reward:

  A. baseline      -- CAISR sleep architecture + autonomic (features_standard.csv),
                      the "submit" preset WITHOUT age -> should reproduce ~0.168.
  B. + nk          -- add per-stage NeuroKit HRV/EEG-complexity/resp-rate.
  C. + nk + report -- also add EEG spectral/spindles/oxygenation/resp-events/REM.

Identifiers and known leakage columns are always excluded from X (dataset, site,
label, time_to_event, time_to_last_visit, channel names, timing). `age` is excluded
from features (the metric already discounts age) but kept aside for scoring.

Usage (on pdmle, as arshia_ilaty_physio26):
    python3 benchmark_new_features.py --exports /data-temp/physio-viewer/exports
"""
from __future__ import annotations

import argparse
import os
import sys
import csv

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

# official Challenge scoring — importable from the repo root OR from the script's
# own directory (when deployed alongside a copy of evaluate_model.py).
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from evaluate_model import (compute_auroc, compute_auprc, compute_auroc_age,
                            compute_auroc_weighted, compute_accuracy,
                            compute_f_measure, compute_prevalence, compute_reward)

SITE_NAMES = {"S0001": "BIDMC", "I0002": "Emory", "I0006": "Kaiser"}

# Columns that are NEVER features: identifiers, target, leakage, provenance/timing,
# and the per-recording channel names / compute-time bookkeeping the exporters add.
NON_FEATURE = {
    "dataset", "bids_folder", "session", "site", "site_name", "label",
    "age",                      # excluded from X; the reward already discounts age
    "sex", "race", "ethnicity", "bmi",   # demographics dropped (matches caisr_autonomic; keeps it signal-only)
    "time_to_event", "time_to_last_visit",       # outcome-timing leakage
    "ecg_channel", "eeg_channel", "rsp_channel", "spo2_channel", "eog_channel",
    "has_resp_caisr", "report_seconds", "nk_seconds",
    "n_beats_total", "n_beats_clean",
}


def _f(s):
    try:
        v = float(s)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def load_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def key(r):
    return (r.get("bids_folder", ""), r.get("session", ""))


def build_matrix(rows, feature_cols):
    X = np.empty((len(rows), len(feature_cols)), dtype=np.float32)
    for i, r in enumerate(rows):
        for j, c in enumerate(feature_cols):
            X[i, j] = _f(r.get(c))
    return X


def fit_clf(Xtr, ytr):
    base = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=1.0, early_stopping=True,
        validation_fraction=0.15, random_state=42)
    n_min = int(np.bincount(ytr).min()) if len(np.unique(ytr)) > 1 else 0
    if n_min >= 3 and len(ytr) >= 30:
        clf = CalibratedClassifierCV(base, method="isotonic", cv=min(3, n_min))
        clf.fit(Xtr, ytr)
    else:
        base.fit(Xtr, ytr)
        clf = base
    return clf


def loso(X, y, ages, sites):
    """Leave-one-site-out; return pooled out-of-fold metrics + per-fold rewards."""
    all_true, all_prob, all_age = [], [], []
    fold = {}
    for site in np.unique(sites):
        te = sites == site
        tr = ~te
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        clf = fit_clf(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        binary = (prob > float(y[tr].mean())).astype(int)
        a2p = compute_prevalence(ages[te], y, ages, gap=2)
        try:
            fr = float(compute_reward(y[te], binary, ages[te], a2p))
        except Exception:
            fr = float("nan")
        fold[str(site)] = (int(te.sum()), fr)
        all_true.extend(y[te]); all_prob.extend(prob); all_age.extend(ages[te])

    yt = np.asarray(all_true); yp = np.asarray(all_prob); ya = np.asarray(all_age)
    a2p_all = compute_prevalence(ya, y, ages, gap=2)

    # binary decision at the training prevalence (the deployed rule), plus a sweep
    bin_pi = (yp > float(y.mean())).astype(int)
    best_r, best_thr = float("-inf"), float(y.mean())
    for thr in [0.05, 0.076, 0.10, 0.15, float(y.mean())]:
        r = float(compute_reward(yt, (yp > thr).astype(int), ya, a2p_all))
        if r > best_r:
            best_r, best_thr = r, thr

    def safe(fn, *a):
        try:
            return float(fn(*a))
        except Exception:
            return float("nan")

    return {
        "reward_at_pi": safe(compute_reward, yt, bin_pi, ya, a2p_all),
        "best_reward": best_r, "best_threshold": best_thr,
        "auroc": safe(compute_auroc, yt, yp),
        "auprc": safe(compute_auprc, yt, yp),
        "auroc_age": safe(compute_auroc_age, yt, yp, ya, 2),
        "auroc_weighted": safe(compute_auroc_weighted, yt, yp, ya, 2),
        "accuracy": safe(compute_accuracy, yt, bin_pi),
        "f_measure": safe(compute_f_measure, yt, bin_pi),
        "folds": fold,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", default="/data-temp/physio-viewer/exports")
    args = ap.parse_args()
    ex = args.exports

    base = load_csv(os.path.join(ex, "features_standard.csv"))
    nk = {key(r): r for r in load_csv(os.path.join(ex, "nk_features_standard.csv"))}
    rep = {key(r): r for r in load_csv(os.path.join(ex, "report_features_standard.csv"))}

    # keep only recordings with a valid binary label and finite age (scoring needs age)
    rows, ys, ages, sites = [], [], [], []
    for r in base:
        lab = r.get("label", "")
        if lab not in ("True", "False", "0", "1"):
            continue
        age = _f(r.get("age"))
        if not np.isfinite(age):
            continue
        rows.append(r)
        ys.append(1 if lab in ("True", "1") else 0)
        ages.append(age)
        sites.append(r.get("site", ""))
    y = np.asarray(ys, int)
    ages = np.asarray(ages, float)
    sites = np.asarray(sites)

    base_cols = [c for c in base[0].keys() if c not in NON_FEATURE]
    nk_cols = [c for c in next(iter(nk.values())).keys() if c not in NON_FEATURE]
    rep_cols = [c for c in next(iter(rep.values())).keys() if c not in NON_FEATURE]

    # attach nk + report columns onto the baseline rows by (bids_folder, session)
    n_nk_missing = n_rep_missing = 0
    for r in rows:
        k = key(r)
        n = nk.get(k)
        if n:
            for c in nk_cols:
                r["nk__" + c] = n.get(c)
        else:
            n_nk_missing += 1
        p = rep.get(k)
        if p:
            for c in rep_cols:
                r["rep__" + c] = p.get(c)
        else:
            n_rep_missing += 1

    nk_feat = ["nk__" + c for c in nk_cols]
    rep_feat = ["rep__" + c for c in rep_cols]

    sets = {
        "A. baseline (CAISR+autonomic, no age)": base_cols,
        "B. baseline + NK per-stage": base_cols + nk_feat,
        "C. baseline + NK + report": base_cols + nk_feat + rep_feat,
    }

    print(f"cohort: {len(rows)} recordings | CI+={int(y.sum())} CI-={int((1-y).sum())} "
          f"prevalence={y.mean():.3f}")
    print(f"sites: {dict(zip(*np.unique(sites, return_counts=True)))}")
    print(f"nk rows missing: {n_nk_missing} | report rows missing: {n_rep_missing}")
    print(f"feature counts -> baseline={len(base_cols)} nk={len(nk_feat)} report={len(rep_feat)}\n")

    hdr = "%-40s %6s %8s %8s %8s %8s %8s" % (
        "feature set", "nfeat", "REWARD", "best_R", "AUROC", "age-AUR", "AUPRC")
    print(hdr); print("-" * len(hdr))
    results = {}
    for name, cols in sets.items():
        X = build_matrix(rows, cols)
        m = loso(X, y, ages, sites)
        results[name] = m
        print("%-40s %6d %+8.3f %+8.3f %8.3f %8.3f %8.3f" % (
            name, len(cols), m["reward_at_pi"], m["best_reward"],
            m["auroc"], m["auroc_age"], m["auprc"]))

    print("\nPer-fold reward (hold-out site, at training prevalence):")
    for name, m in results.items():
        parts = ["%s=%+.3f(n=%d)" % (SITE_NAMES.get(s, s), r, n)
                 for s, (n, r) in m["folds"].items()]
        print("  %-40s %s" % (name, "  ".join(parts)))

    print("\nFull metric table for set C (baseline + NK + report):")
    c = results["C. baseline + NK + report"]
    for k2 in ["reward_at_pi", "best_reward", "best_threshold", "auroc",
               "auroc_age", "auroc_weighted", "auprc", "accuracy", "f_measure"]:
        print("  %-16s %+.4f" % (k2, c[k2]))


if __name__ == "__main__":
    main()
