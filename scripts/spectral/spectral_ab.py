#!/usr/bin/env python3
"""LOSO A/B: champion vs champion + spectral-shape features (standard cohort).

Rebuilds the champion feature matrix exactly as `levers_test.load_csv` does, but also
recovers the (bids_folder, session) key per row so the per-recording spectral features
from `export_spectral_extra.py` align row-for-row. Then runs BOTH arms through the
production stack -- per-site rank-norm (bmi protected) -> HGB LOSO -> decision rules --
and reports pooled metrics, per-held-out-site AUROC, and the four decision-rule rewards
with deltas. Cross-checks the rebuilt champion X against lv.load_csv before trusting the
alignment. Only the aggregate JSON/table leaves the box (DUA); per-recording CSVs stay.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import levers_test as lv
import feature_prep as fp
import evaluate_model as ev

SITE_NAMES = {"S0001": "BIDMC", "I0002": "Emory", "I0006": "Kaiser"}
SITE_ORDER = ["S0001", "I0006", "I0002"]
RULES = ["pi", "transfer", "oracle", "q_gt_pa"]

JOIN = ["bids_folder", "session"]
NON_FEATURE = {"dataset", "bids_folder", "session", "site", "site_name", "label", "age",
               "sex", "race", "ethnicity", "time_to_event", "time_to_last_visit",
               "ecg_channel", "eeg_channel", "rsp_channel", "spo2_channel", "eog_channel",
               "emg_channel", "has_resp_caisr", "report_seconds", "nk_seconds",
               "micro_seconds", "n_beats_total", "n_beats_clean", "arousal_fs",
               "n_epochs_scored", "duration_hours", "n_epochs"}


def load_with_keys(exports, cohort):
    def rk(name):
        df = pd.read_csv(os.path.join(exports, f"{name}_{cohort}.csv"),
                         dtype={"bids_folder": str, "session": str}, low_memory=False)
        for k in JOIN:
            df[k] = df[k].fillna("").astype(str).str.strip()
        return df.drop_duplicates(JOIN)

    base = rk("features")
    nk, rep, arch, micro = rk("nk_features"), rk("report_features"), rk("arch_features"), rk("micro_features")

    def pfx(df, p):
        cols = {c: f"{p}{c}" for c in df.columns if c not in JOIN and c not in NON_FEATURE}
        return df[JOIN + list(cols)].rename(columns=cols)

    m = base.merge(pfx(nk, "nk__"), on=JOIN, how="left", validate="one_to_one") \
            .merge(pfx(rep, "rep__"), on=JOIN, how="left", validate="one_to_one") \
            .merge(pfx(arch, "arch__"), on=JOIN, how="left", validate="one_to_one") \
            .merge(pfx(micro, "micro__"), on=JOIN, how="left", validate="one_to_one")
    lab = m["label"].map({True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0})
    keep = lab.isin([0, 1]).values
    m = m[keep].reset_index(drop=True)
    y = lab[keep].astype(int).values
    ages = np.asarray(m["age"].apply(lambda v: float(v) if str(v) not in ("", "nan") else np.nan))
    sites = m["site"].astype(str).values
    keys = list(zip(m["bids_folder"].values, m["session"].values))
    base_feats = [c for c in base.columns if c not in JOIN and c not in NON_FEATURE]
    feat_cols = base_feats + [c for c in m.columns if c.startswith(("nk__", "rep__", "arch__", "micro__"))]
    X = m[feat_cols].apply(pd.to_numeric, errors="coerce").astype(np.float64).values
    bmi_idx = feat_cols.index("bmi") if "bmi" in feat_cols else None
    return X, y, ages, sites, feat_cols, bmi_idx, keys


def load_spectral(path, keys):
    sp = pd.read_csv(path, dtype={"bids_folder": str, "session": str}, low_memory=False)
    for k in JOIN:
        sp[k] = sp[k].fillna("").astype(str).str.strip()
    sp = sp.drop_duplicates(JOIN).set_index(JOIN)
    cols = [c for c in sp.columns if c.startswith("sp__") and not c.startswith("sp__n_")]
    S = np.full((len(keys), len(cols)), np.nan)
    hit = 0
    for i, (b, s) in enumerate(keys):
        if (b, s) in sp.index:
            S[i] = pd.to_numeric(sp.loc[(b, s), cols], errors="coerce").values
            hit += 1
    return S, cols, hit


def _metric_of(col):
    core = col[len("sp__"):]
    for m in ("aper_exp", "aper_off", "spec_ent", "sef95", "med_freq"):
        if core.startswith(m + "_"):
            return m
    return None


def subset_indices(sp_cols):
    """Named spectral-column subsets to A/B against the champion (indices into sp_cols)."""
    is_std = lambda c: c.endswith("std")
    return {
        "full": list(range(len(sp_cols))),
        # drop the aperiodic OFFSET (an amplitude/montage term -> prime site-signal injector)
        "no_offset": [i for i, c in enumerate(sp_cols) if _metric_of(c) != "aper_off"],
        # pure shape: exponent + entropy + SEF95 stage-means only (no offset, no med_freq, no std)
        "shape_means": [i for i, c in enumerate(sp_cols)
                        if _metric_of(c) in ("aper_exp", "spec_ent", "sef95") and not is_std(c)],
    }


def run_arm(Xin, y, ages, sites, bmi_idx, names):
    protect = [bmi_idx] if bmi_idx is not None else []
    Xr = lv.site_rank(Xin, sites, protect=protect)
    pooled, folds, oof = lv.run_loso(Xr, y, ages, sites, fp, ev, names, model="hgb")
    rules = lv.decision_rules(oof, y, ages, ev)
    per_site = {str(s): {"n": n, "reward_pi": rw, "ac_auroc": ac, "auroc": au}
                for (s, n, rw, ac, au) in folds}
    return {"pooled": pooled, "rules": {r: float(rules[r]) for r in RULES}, "per_site": per_site}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", required=True)
    ap.add_argument("--spectral", required=True)
    ap.add_argument("--cohort", default="standard")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    X, y, ages, sites, feat_cols, bmi_idx, keys = load_with_keys(a.exports, a.cohort)
    # cross-check against the canonical loader (guarantees champion arm == levers champion)
    Xc, yc, *_ = lv.load_csv(a.exports, a.cohort)
    assert X.shape == Xc.shape and np.array_equal(y, yc), "row/label mismatch vs lv.load_csv"
    assert np.allclose(X, Xc, equal_nan=True), "champion X differs from lv.load_csv"

    S, sp_cols, hit = load_spectral(a.spectral, keys)
    n_all = np.isfinite(S).all(axis=1).sum()
    coverage = {"n_rows": len(keys), "spectral_matched": int(hit),
                "rows_all_spectral_finite": int(n_all), "n_spectral_cols": len(sp_cols)}

    groups = subset_indices(sp_cols)
    order = ["full", "no_offset", "shape_means"]
    champ = run_arm(X, y, ages, sites, bmi_idx, feat_cols)
    arms = {}
    for name in order:
        idx = groups[name]
        cols = [sp_cols[i] for i in idx]
        arm = run_arm(np.hstack([X, S[:, idx]]), y, ages, sites, bmi_idx, feat_cols + cols)
        arms[name] = {"n_cols": len(cols), "cols": cols, "arm": arm,
                      "delta_pooled": {k: arm["pooled"][k] - champ["pooled"][k] for k in arm["pooled"]},
                      "delta_rules": {r: arm["rules"][r] - champ["rules"][r] for r in RULES}}

    out = {"cohort": a.cohort, "n": int(len(y)), "pos": int(y.sum()),
           "n_champ_feats": len(feat_cols), "all_spectral_cols": sp_cols, "coverage": coverage,
           "champion": champ, "subsets": arms}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)

    def line(name, arm, ncols):
        p = arm["pooled"]; r = arm["rules"]
        return (f"{name:<20}{ncols:>6}{p['age_auroc']:>10.4f}{p['auroc']:>9.4f}{p['auprc']:>9.4f}"
                f"{r['pi']:>+9.4f}{r['transfer']:>+9.4f}{r['oracle']:>+9.4f}{r['q_gt_pa']:>+9.4f}")
    print(f"\n=== Spectral-shape subset A/B  (cohort={a.cohort}, n={len(y)}, pos={int(y.sum())}, "
          f"{hit}/{len(keys)} matched) ===")
    print(f"{'arm':<20}{'ncols':>6}{'AC-AUROC':>10}{'AUROC':>9}{'AUPRC':>9}{'pi':>9}{'transfer':>9}{'oracle':>9}{'q>pa':>9}")
    print(line("champion", champ, 0))
    for name in order:
        print(line("+" + name, arms[name]["arm"], arms[name]["n_cols"]))
    print("\nΔ vs champion (transfer = deployable lever):")
    for name in order:
        dp, dr = arms[name]["delta_pooled"], arms[name]["delta_rules"]
        print(f"  {name:<12} ΔAC-AUROC {dp['age_auroc']:+.4f}  ΔAUROC {dp['auroc']:+.4f}  "
              f"ΔAUPRC {dp['auprc']:+.4f}  Δtransfer {dr['transfer']:+.4f}  Δpi {dr['pi']:+.4f}")
    print("\nper-held-out-site (champion baseline):")
    for s in SITE_ORDER:
        if s in champ["per_site"]:
            c = champ["per_site"][s]
            print(f"  {SITE_NAMES.get(s,s):<7} n={c['n']:<4} AC-AUROC {c['ac_auroc']:.4f}  reward_pi {c['reward_pi']:+.4f}")
    print(f"\ncoverage: {coverage}")
    print(f"wrote {a.out}")
    print("DONE_SPECTRAL_AB")


if __name__ == "__main__":
    main()
