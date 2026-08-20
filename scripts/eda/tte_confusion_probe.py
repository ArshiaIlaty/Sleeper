#!/usr/bin/env python3
"""Probe: LOSO out-of-fold predictions x Time_to_Event, by confusion quadrant.

Runs the EXACT production LOSO stack (feature_prep site-MoE + Kaiser fine-tune +
BMI imputer) to get an out-of-fold probability per recording, thresholds each
fold at its training prevalence (matches run_local_cv.loso / cross_validate_s3),
classifies every recording into TP/FP/TN/FN, joins Time_to_Event (days from PSG
to first CI ICD code) from demographics by BidsFolder, and prints the breakdown.

Text-only probe — no figure. Runs on the pdmle box under the physio26 account.

  PYTHONPATH=/data-temp/physio-viewer/bench/repo python3 tte_confusion_probe.py \
      --cache /data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz \
      --demo  /data-temp/shared-physionet26-dataset/extracted/demographics.csv
"""
from __future__ import annotations
import argparse
import csv
import numpy as np

import feature_prep as fp
import evaluate_model as ev


def loso_oof(X, y, ages, sites, feat_names):
    """Return per-recording OOF prob + per-fold-prevalence binary, aligned to input order."""
    prob = np.full(len(y), np.nan)
    binr = np.full(len(y), -1, dtype=int)
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
        p = fp.predict_with_kaiser_override(models, Xte, sites[te], use_kaiser_finetuned=True)
        prob[te] = p
        binr[te] = (p > float(y[tr].mean())).astype(int)
    return prob, binr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--demo", required=True)
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    X = d["X"].astype(np.float32); y = d["y"].astype(int)
    ages = d["ages"].astype(float); sites = np.asarray(d["sites"])
    pids = np.asarray([str(p) for p in d["pids"]])
    names = [str(n) for n in d["feature_names"]]

    prob, binr = loso_oof(X, y, ages, sites, names)

    # join Time_to_Event by BidsFolder
    tte_by_pid = {}
    with open(args.demo) as fh:
        for r in csv.DictReader(fh):
            pid = r.get("BidsFolder", "")
            v = r.get("Time_to_Event", "")
            try:
                tte_by_pid[pid] = float(v) if v not in ("", None) else np.nan
            except (ValueError, TypeError):
                tte_by_pid[pid] = np.nan
    tte = np.array([tte_by_pid.get(p, np.nan) for p in pids])

    valid = binr >= 0
    quad = np.full(len(y), "??", dtype=object)
    quad[valid & (y == 1) & (binr == 1)] = "TP"
    quad[valid & (y == 1) & (binr == 0)] = "FN"
    quad[valid & (y == 0) & (binr == 1)] = "FP"
    quad[valid & (y == 0) & (binr == 0)] = "TN"

    def desc(mask):
        t = tte[mask]
        n = int(mask.sum())
        n_def = int(np.isfinite(t).sum())
        td = t[np.isfinite(t)]
        if td.size:
            q = np.percentile(td, [0, 25, 50, 75, 100])
            yrs = td / 365.25
            stat = (f"defined={n_def}/{n}  days[min/Q1/med/Q3/max]="
                    f"{q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}/{q[3]:.0f}/{q[4]:.0f}"
                    f"  yrs med={np.median(yrs):.2f}")
        else:
            stat = f"defined={n_def}/{n}  (no finite Time_to_Event)"
        return n, stat

    print(f"cohort n={len(y)}  valid-oof={int(valid.sum())}  "
          f"CI+={int((y==1).sum())}  CI-={int((y==0).sum())}  prev={y.mean():.3f}")
    print(f"overall Time_to_Event defined for {int(np.isfinite(tte).sum())}/{len(tte)} recordings")
    print("-" * 78)
    for q in ["TP", "FN", "FP", "TN"]:
        m = quad == q
        n, stat = desc(m)
        print(f"  {q}  n={n:4d}   {stat}")
    print("-" * 78)
    # converters only: TP vs FN — the key comparison
    for q in ["TP", "FN"]:
        m = (quad == q) & np.isfinite(tte)
        if m.sum():
            print(f"  {q} converter Time_to_Event (days): "
                  f"{np.sort(tte[m]).astype(int).tolist()}")


if __name__ == "__main__":
    main()
