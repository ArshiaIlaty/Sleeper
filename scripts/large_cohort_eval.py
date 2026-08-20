#!/usr/bin/env python3
"""Large-cohort (6530 / 497 pos) vs standard (1103 / 84 pos) — full-feature training eval.

Deadline-day question: with the large cohort's 5.9x positives, what do our headline
metrics (reward, reward@pi, AUROC, AC-AUROC, AUPRC) look like under the exact production
stack? The learning curve said the model is positive-count-limited and still climbing at
n=84, so this is the payoff test.

WHY BUILT FROM CSVs (not the 436-feature .npz cache): that cache's base block comes from
team_code.extract_all_features -- a raw-EDF WAVEFORM pass (~9 h over 6530 recordings), not
feasible today. Phase-1 established the CSV-assembly path (phase1_large_variance.py); this
extends it to ALL FIVE feature families + one-hot demographics, and builds STANDARD THE
SAME WAY so the comparison is strictly apples-to-apples and standard acts as a calibration
anchor (should land near the known LOSO 0.635 AC-AUROC / 0.712 AUROC / +0.274 reward@pi).

Feature assembly (identical for both cohorts):
  * base features_.csv numeric cols (stage %, architecture, event indices, ...), UNPREFIXED
    so the BMI imputer still finds "bmi" -- minus identifiers/leakage/bookkeeping.
  * demographics one-hot to MATCH team_code.extract_demographic_features: sex_{f,m,o},
    race_{asian,black,other,unavail,white}, eth_{hispanic,not_hispanic,unavail}. age kept
    as a column for scoring/thresholds but the production model excludes it (preset submit).
  * arch__ / nk__ / rep__ / micro__ prefixed blocks, left-joined on (bids_folder, session).
CAVEAT (logged): the large base CSV lacks the whole-night autonomic HRV/SpO2 SUMMARY block
(that needs the waveform pass), but the nk__ block carries PER-STAGE HRV, so autonomic
physiology is still represented. Columns are ALIGNED to the intersection present in BOTH
cohorts, so neither cohort gets a feature the other lacks -> honest standard-vs-large delta.

Model + scoring = run_local_cv.loso / .tvt (production site-MoE + Kaiser fine-tune + BMI
imputer + site_decade reward thresholds; evaluate_model.compute_*). LOSO is the honest
new-site proxy; the 70/15/15 seed sweep is the within-distribution variance headline.
Aggregate stats only leave the box.

Usage (on pdmle, as arshia_ilaty_physio26):
  PYTHONPATH=/data-temp/physio-viewer/bench/repo python3 large_cohort_eval.py \
    --exports /data-temp/physio-viewer/exports \
    --repo /data-temp/physio-viewer/bench/repo --cvdir /data-temp/physio-viewer/bench \
    --out /data-temp/physio-viewer/exports/large_cohort_eval --seeds 30
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import pandas as pd

JOIN = ["bids_folder", "session"]

# base-CSV columns that are identifiers / leakage / bookkeeping -> never features.
# "bmi" is deliberately NOT here: feature_prep's BMI imputer locates it by exact name.
NON_FEATURE = {
    "dataset", "bids_folder", "session", "site", "site_name", "label", "age",
    "sex", "race", "ethnicity", "time_to_event", "time_to_last_visit",
    "ecg_channel", "eeg_channel", "rsp_channel", "spo2_channel", "eog_channel",
    "emg_channel", "has_resp_caisr", "report_seconds", "nk_seconds",
    "micro_seconds", "n_beats_total", "n_beats_clean", "arousal_fs",
    "n_epochs_scored", "duration_hours", "n_epochs",
}

FAMILIES = [("arch_features", "arch__"), ("nk_features", "nk__"),
            ("report_features", "rep__"), ("micro_features", "micro__")]


def read_keyed(path):
    df = pd.read_csv(path, dtype={"bids_folder": str, "session": str}, low_memory=False)
    for k in JOIN:
        df[k] = df[k].fillna("").astype(str).str.strip()
    return df.drop_duplicates(JOIN)


def demo_onehot(df):
    """One-hot sex/race/ethnicity to match team_code.extract_demographic_features order."""
    sex = df["sex"].astype(str)
    race = df["race"].astype(str)
    eth = df["ethnicity"].astype(str)
    out = pd.DataFrame(index=df.index)
    out["sex_f"] = (sex == "Female").astype(np.float32)
    out["sex_m"] = (sex == "Male").astype(np.float32)
    out["sex_o"] = (~sex.isin(["Female", "Male"])).astype(np.float32)
    rmap = {"asian": "Asian", "black": "Black", "other": "Others",
            "unavail": "Unavailable", "white": "White"}
    for key, val in rmap.items():
        out[f"race_{key}"] = (race == val).astype(np.float32)
    out["eth_hispanic"] = (eth == "Hispanic").astype(np.float32)
    out["eth_not_hispanic"] = (eth == "Not Hispanic").astype(np.float32)
    out["eth_unavail"] = (~eth.isin(["Hispanic", "Not Hispanic"])).astype(np.float32)
    return out


def build_matrix(exports, cohort):
    base = read_keyed(os.path.join(exports, f"features_{cohort}.csv"))
    # base numeric feature cols (keep bmi unprefixed); demographics handled via one-hot
    base_feats = [c for c in base.columns if c not in JOIN and c not in NON_FEATURE]
    demo = demo_onehot(base)
    m = base[JOIN + base_feats].copy()
    for c in demo.columns:
        m[c] = demo[c].values
    feat_cols = base_feats + list(demo.columns)

    for stem, pfx in FAMILIES:
        path = os.path.join(exports, f"{stem}_{cohort}.csv")
        if not os.path.exists(path):
            print(f"  ! {cohort}: {stem} missing -> skipped", flush=True)
            continue
        fam = read_keyed(path)
        cols = {c: f"{pfx}{c}" for c in fam.columns if c not in JOIN and c not in NON_FEATURE}
        fam_p = fam[JOIN + list(cols)].rename(columns=cols)
        m = m.merge(fam_p, on=JOIN, how="left", validate="one_to_one")
        feat_cols += list(cols.values())

    lab = base.set_index([*JOIN]).reindex(m.set_index([*JOIN]).index)["label"].values
    y = pd.Series(lab).map({True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}).astype(float).values
    keep = np.isin(y, [0, 1])
    m = m[keep].reset_index(drop=True)
    y = y[keep].astype(int)
    ages = pd.to_numeric(base.set_index(JOIN).reindex(m.set_index(JOIN).index)["age"],
                         errors="coerce").astype(float).values
    sites = base.set_index(JOIN).reindex(m.set_index(JOIN).index)["site"].astype(str).values
    X = m[feat_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32).values
    return X, y, ages, sites, feat_cols


def align(names_a, names_b):
    """Common feature columns, in the order of the first cohort (deterministic)."""
    sb = set(names_b)
    common = [n for n in names_a if n in sb]
    return common


def subset(X, names, common):
    idx = [names.index(c) for c in common]
    return X[:, idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cvdir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args()
    for p in (args.repo, args.cvdir):
        if p not in sys.path:
            sys.path.insert(0, p)
    import feature_prep as fp
    import evaluate_model as ev
    import run_local_cv as rlc

    # build both cohorts, then align to common feature columns
    Xs, ys, ages_s, sites_s, names_s = build_matrix(args.exports, "standard")
    Xl, yl, ages_l, sites_l, names_l = build_matrix(args.exports, "large")
    common = align(names_s, names_l)
    Xs = subset(Xs, names_s, common)
    Xl = subset(Xl, names_l, common)
    print(f"standard: n={len(ys)} pos={int(ys.sum())} feats={Xs.shape[1]}  "
          f"sites={dict(zip(*np.unique(sites_s, return_counts=True)))}", flush=True)
    print(f"large:    n={len(yl)} pos={int(yl.sum())} feats={Xl.shape[1]}  "
          f"sites={dict(zip(*np.unique(sites_l, return_counts=True)))}", flush=True)
    print(f"common feature columns: {len(common)}  "
          f"(standard-only dropped: {len(names_s)-len(common)}, "
          f"large-only dropped: {len(names_l)-len(common)})", flush=True)

    seeds = list(range(args.seeds))
    results = {}
    for tag, (X, y, ages, sites) in [("standard", (Xs, ys, ages_s, sites_s)),
                                     ("large", (Xl, yl, ages_l, sites_l))]:
        print("\n" + "=" * 90, flush=True)
        print(f"COHORT {tag.upper()}  n={len(y)}  positives={int(y.sum())} "
              f"({100*y.mean():.1f}%)  features={X.shape[1]}", flush=True)
        print("=" * 90, flush=True)

        # ---- LOSO (honest new-site proxy) ----
        pooled, folds = rlc.loso(X, y, ages, sites, fp, ev, common)
        print("  LOSO (leave-one-site-out) — production site-MoE + Kaiser + BMI-impute:", flush=True)
        for s, n, r, aa in folds:
            print(f"      hold-out {rlc.SITE_NAMES.get(s, s):>6} n={n:5d}  "
                  f"reward={r:+.3f}  age-AUROC={aa:.3f}", flush=True)
        print(f"    POOLED  reward@pi={pooled['reward_at_pi']:+.3f}  best={pooled['best_reward']:+.3f}"
              f"  AUROC={pooled['auroc']:.3f}  AC-AUROC={pooled['age_auroc']:.3f}"
              f"  AUPRC={pooled['auprc']:.3f}", flush=True)

        # ---- 70/15/15 seed sweep (within-distribution variance headline) ----
        keys = ["reward", "reward_at_pi", "auroc", "age_auroc", "age_weighted", "auprc"]
        acc = {k: [] for k in keys}
        for sd in seeds:
            try:
                t = rlc.tvt(X, y, ages, sites, fp, ev, common, seed=sd)["test"]
                for k in keys:
                    acc[k].append(t[k])
            except Exception as e:
                print(f"      seed {sd} failed: {type(e).__name__}: {e}", flush=True)
        print(f"  70/15/15 over {len(acc['reward'])} seeds (mean ± SD):", flush=True)
        tvt_summary = {}
        for k in keys:
            v = np.array(acc[k], float)
            tvt_summary[k] = dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v)))
            print(f"      {k:<14} {np.nanmean(v):+.3f} ± {np.nanstd(v):.3f}", flush=True)

        results[tag] = dict(
            n=int(len(y)), positives=int(y.sum()), features=int(X.shape[1]),
            loso=dict(reward_at_pi=pooled["reward_at_pi"], best_reward=pooled["best_reward"],
                      auroc=pooled["auroc"], ac_auroc=pooled["age_auroc"], auprc=pooled["auprc"],
                      folds=[dict(site=s, n=n, reward=r, ac_auroc=aa) for s, n, r, aa in folds]),
            tvt=tvt_summary)

    print("\n" + "=" * 90, flush=True)
    print("STANDARD (anchor)  vs  LARGE  —  primary metrics", flush=True)
    print("=" * 90, flush=True)
    for metric, path in [("LOSO reward@pi", ("loso", "reward_at_pi")),
                         ("LOSO AUROC", ("loso", "auroc")),
                         ("LOSO AC-AUROC", ("loso", "ac_auroc")),
                         ("LOSO AUPRC", ("loso", "auprc"))]:
        sv = results["standard"][path[0]][path[1]]
        lv = results["large"][path[0]][path[1]]
        print(f"  {metric:<18} standard={sv:+.3f}   large={lv:+.3f}   Δ={lv-sv:+.3f}", flush=True)
    for metric in ["reward", "auroc", "age_auroc"]:
        sv = results["standard"]["tvt"][metric]["mean"]; ss = results["standard"]["tvt"][metric]["sd"]
        lv = results["large"]["tvt"][metric]["mean"]; ls = results["large"]["tvt"][metric]["sd"]
        print(f"  70/15/15 {metric:<10} standard={sv:+.3f}±{ss:.3f}   large={lv:+.3f}±{ls:.3f}"
              f"   ΔSD={ls-ss:+.3f}", flush=True)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "large_cohort_eval_summary.json"), "w") as fh:
            json.dump({"results": results, "common_features": common, "seeds": len(seeds)}, fh, indent=2)
        print(f"\nwrote large_cohort_eval_summary.json to {args.out}", flush=True)
    print("DONE_LARGE_COHORT_EVAL", flush=True)


if __name__ == "__main__":
    main()
