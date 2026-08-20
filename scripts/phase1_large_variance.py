#!/usr/bin/env python3
"""Phase-1 large-cohort variance / transfer test (challenge pivot gate).

Question: does going from 84 -> 497 positives collapse the +-0.45 reward variance
and improve cross-site transfer? We answer it with the CHEAP feature families that
already exist for BOTH cohorts (features_ + arch_ + report_), building each cohort's
matrix IDENTICALLY from the CSVs so the ONLY difference is n (positives). This is a
variance/decision-stability test, NOT a final-performance number -- NK + micro are
deferred to Phase 2 and would only be re-extracted for large if this gate passes.

For each cohort we run, through the exact production stack (run_local_cv):
  * 70/15/15 seed sweep (many random splits) -> reward mean +- SD and NEW-vs-noise
    stability. The SD across seeds is the headline: does it shrink at 497 positives?
  * LOSO (leave-one-site-out) -> per-site reward + age-AUROC (cross-site transfer).

Feature set (both cohorts, identical): base features_ columns UNPREFIXED (so the
BMI imputer finds "bmi") + arch__ + rep__. Identifiers/leakage excluded.

Usage (on pdmle, as arshia_ilaty_physio26):
  PYTHONPATH=/data-temp/physio-viewer/bench/repo python3 phase1_large_variance.py \
    --exports /data-temp/physio-viewer/exports \
    --repo /data-temp/physio-viewer/bench/repo --cvdir /data-temp/physio-viewer/bench \
    --seeds 50
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd

JOIN = ["bids_folder", "session"]
# never features: identifiers, demographics kept separately, leakage, bookkeeping
NON_FEATURE = {
    "dataset", "bids_folder", "session", "site", "site_name", "label", "age",
    "sex", "race", "ethnicity", "time_to_event", "time_to_last_visit",
    "ecg_channel", "eeg_channel", "rsp_channel", "spo2_channel", "eog_channel",
    "emg_channel", "has_resp_caisr", "report_seconds", "nk_seconds",
    "micro_seconds", "n_beats_total", "n_beats_clean", "arousal_fs",
    "n_epochs_scored", "duration_hours", "n_epochs",
}
# "bmi" is intentionally NOT in NON_FEATURE: the feature_prep BMI imputer locates
# it by the exact name "bmi", so it must stay an unprefixed feature column.


def read_keyed(path):
    df = pd.read_csv(path, dtype={"bids_folder": str, "session": str}, low_memory=False)
    for k in JOIN:
        df[k] = df[k].fillna("").astype(str).str.strip()
    return df.drop_duplicates(JOIN)


def build_matrix(exports, cohort):
    """Assemble X, y, ages, sites, feature_names from the 3 cheap CSV families."""
    base = read_keyed(os.path.join(exports, f"features_{cohort}.csv"))
    arch = read_keyed(os.path.join(exports, f"arch_features_{cohort}.csv"))
    rep = read_keyed(os.path.join(exports, f"report_features_{cohort}.csv"))

    # prefix arch / report feature cols; keep base cols as-is (bmi stays "bmi")
    def prefix(df, pfx):
        keep = JOIN
        cols = {c: f"{pfx}{c}" for c in df.columns
                if c not in keep and c not in NON_FEATURE}
        return df[keep + list(cols)].rename(columns=cols)

    arch_p = prefix(arch, "arch__")
    rep_p = prefix(rep, "rep__")
    m = base.merge(arch_p, on=JOIN, how="left", validate="one_to_one") \
            .merge(rep_p, on=JOIN, how="left", validate="one_to_one")

    # target + meta
    lab = m["label"]
    y = lab.map({True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}).astype(float)
    keep = y.isin([0, 1]).values
    m = m[keep].reset_index(drop=True)
    y = y[keep].astype(int).values
    ages = pd.to_numeric(m["age"], errors="coerce").astype(float).values
    sites = m["site"].astype(str).values

    # base feature columns = numeric base cols not in NON_FEATURE (includes bmi)
    base_feats = [c for c in base.columns
                  if c not in JOIN and c not in NON_FEATURE]
    feat_cols = base_feats + [c for c in m.columns if c.startswith(("arch__", "rep__"))]
    X = m[feat_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32).values
    return X, y, ages, sites, feat_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cvdir", required=True)
    ap.add_argument("--seeds", type=int, default=50)
    args = ap.parse_args()
    for p in (args.repo, args.cvdir):
        if p not in sys.path:
            sys.path.insert(0, p)
    import feature_prep as fp
    import evaluate_model as ev
    import run_local_cv as rlc

    for cohort in ("standard", "large"):
        X, y, ages, sites, names = build_matrix(args.exports, cohort)
        import collections
        sc = collections.Counter(sites)
        print("=" * 88)
        print(f"COHORT {cohort}:  n={len(y)}  positives={int(y.sum())}  "
              f"({100*y.mean():.1f}%)  features={X.shape[1]}")
        print(f"  sites: {dict(sc)}")

        # ---- 70/15/15 seed sweep: the variance headline ----
        seeds = list(range(args.seeds))
        rw, rwpi, aa, au = [], [], [], []
        for s in seeds:
            try:
                r = rlc.tvt(X, y, ages, sites, fp, ev, names, seed=s)["test"]
                rw.append(r["reward"]); rwpi.append(r["reward_at_pi"])
                aa.append(r["age_auroc"]); au.append(r["auroc"])
            except Exception as e:
                print(f"    seed {s} failed: {type(e).__name__}: {e}")
        rw, rwpi, aa, au = map(lambda v: np.array(v, float), (rw, rwpi, aa, au))
        def ms(v): return f"{np.nanmean(v):+.3f} +- {np.nanstd(v):.3f}"
        print(f"  70/15/15 over {len(rw)} seeds:")
        print(f"    reward         {ms(rw)}   [min {np.nanmin(rw):+.3f}, max {np.nanmax(rw):+.3f}]")
        print(f"    reward_at_pi   {ms(rwpi)}")
        print(f"    age_auroc      {ms(aa)}")
        print(f"    auroc          {ms(au)}")

        # ---- LOSO: cross-site transfer ----
        pooled, folds = rlc.loso(X, y, ages, sites, fp, ev, names)
        print(f"  LOSO pooled: reward@pi={pooled['reward_at_pi']:+.3f}  "
              f"AUROC={pooled['auroc']:.3f}  age-AUROC={pooled['age_auroc']:.3f}  "
              f"AUPRC={pooled['auprc']:.3f}")
        for site, n, r, agev in folds:
            print(f"      hold-out {rlc.SITE_NAMES.get(site, site):>6} n={n:5d}  "
                  f"reward={r:+.3f}  age-AUROC={agev:.3f}")

    print("\nDONE_PHASE1")


if __name__ == "__main__":
    main()
