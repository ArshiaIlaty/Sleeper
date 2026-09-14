#!/usr/bin/env python3
"""Resampling A/B: champion (native fs) vs 200Hz-spliced (Emory 500->200 Hz), LOSO.

Only Emory (I0002) feature values differ between the two arms; rows / labels / sites are
identical -> a paired A/B. Both arms go through the IDENTICAL production stack (per-site rank ->
HGB LOSO -> the four decision rules), reusing levers_test, so any delta is attributable purely to
harmonizing the Emory 500 Hz subset to 200 Hz. Reports pooled metrics, the four reward rules, a
per-site breakdown (Emory is the only site that can move), and a diagnostic of how many Emory rows
the splice actually changed.

Usage:
  PYTHONPATH=<USP>:<repo>:<levers-dir> python3 resample_ab.py \
      --exports-dir <dir with {fam}_large.csv AND {fam}_large_200hz.csv> \
      --repo <repo> --levers-dir <dir> --cohort large
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

NON_FEATURE = {"dataset", "bids_folder", "session", "site", "site_name", "label", "age",
               "sex", "race", "ethnicity", "time_to_event", "time_to_last_visit",
               "ecg_channel", "eeg_channel", "rsp_channel", "spo2_channel", "eog_channel",
               "emg_channel", "has_resp_caisr", "report_seconds", "nk_seconds",
               "micro_seconds", "n_beats_total", "n_beats_clean", "arousal_fs",
               "n_epochs_scored", "duration_hours", "n_epochs"}
JOIN = ["bids_folder", "session"]


def load(exports, cohort, suffix):
    """levers_test.load_csv, parametrized by a filename suffix (so both arms load identically)."""
    import pandas as pd

    def rk(p):
        df = pd.read_csv(p, dtype={"bids_folder": str, "session": str}, low_memory=False)
        for k in JOIN:
            df[k] = df[k].fillna("").astype(str).str.strip()
        return df.drop_duplicates(JOIN)

    def path(fam):
        return os.path.join(exports, f"{fam}_{cohort}{suffix}.csv")

    base = rk(path("features")); arch = rk(path("arch_features")); rep = rk(path("report_features"))
    nk = rk(path("nk_features")); micro = rk(path("micro_features"))

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
    base_feats = [c for c in base.columns if c not in JOIN and c not in NON_FEATURE]
    feat_cols = base_feats + [c for c in m.columns if c.startswith(("nk__", "rep__", "arch__", "micro__"))]
    X = m[feat_cols].apply(pd.to_numeric, errors="coerce").astype(np.float64).values
    key = (m["bids_folder"] + "|" + m["session"]).values
    bmi_idx = feat_cols.index("bmi") if "bmi" in feat_cols else None
    return X, y, ages, sites, feat_cols, bmi_idx, key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports-dir", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--levers-dir", required=True)
    ap.add_argument("--cohort", default="large")
    a = ap.parse_args()
    for p in [a.repo, a.levers_dir]:
        if p and p not in sys.path:
            sys.path.insert(0, p)
    import feature_prep as fp
    import evaluate_model as ev
    import levers_test as lv

    Xc, yc, agc, stc, nmc, bmc, keyc = load(a.exports_dir, a.cohort, "")
    Xr, yr, agr, str_, nmr, bmr, keyr = load(a.exports_dir, a.cohort, "_200hz")
    assert nmc == nmr, "feature-column mismatch between arms"
    print(f"champion: n={len(yc)} feats={Xc.shape[1]} "
          f"sites={ {s: int((stc == s).sum()) for s in np.unique(stc)} }")
    print(f"200hz   : n={len(yr)} feats={Xr.shape[1]} "
          f"sites={ {s: int((str_ == s).sum()) for s in np.unique(str_)} }")

    # diagnostic: how many Emory rows actually changed (align by bids|session key)
    idx_r = {k: i for i, k in enumerate(keyr)}
    em = np.where(stc == "I0002")[0]
    changed, missing, ncells, tot_cells = 0, 0, 0, 0
    for i in em:
        j = idx_r.get(keyc[i])
        if j is None:
            missing += 1; continue
        d = ~np.isclose(np.nan_to_num(Xc[i], nan=-9e9), np.nan_to_num(Xr[j], nan=-9e9), atol=1e-9)
        if d.any():
            changed += 1; ncells += int(d.sum()); tot_cells += Xc.shape[1]
    print(f"Emory rows: {len(em)} | changed by splice: {changed} | unmatched: {missing} | "
          f"mean changed cells/changed-row: {ncells / max(changed,1):.1f}/{Xc.shape[1]}")

    print("\n" + "=" * 92)
    print(f"{'arm':<12}{'AC-AUROC':>10}{'AUROC':>9}{'AUPRC':>9}{'rwd@pi':>9}{'transfer':>10}"
          f"{'q>pa':>9}{'oracle':>9}")
    print("-" * 92)
    res = {}
    for tag, (X, y, ages, sites, names, bmi) in [
            ("champion", (Xc, yc, agc, stc, nmc, bmc)),
            ("200hz", (Xr, yr, agr, str_, nmr, bmr))]:
        protect = [bmi] if bmi is not None else []
        Xrk = lv.site_rank(X, sites, protect)
        pooled, folds, oof = lv.run_loso(Xrk, y, ages, sites, fp, ev, names, model="hgb")
        dr = lv.decision_rules(oof, y, ages, ev)
        res[tag] = (pooled, folds, dr)
        print(f"{tag:<12}{pooled['age_auroc']:>10.4f}{pooled['auroc']:>9.4f}{pooled['auprc']:>9.4f}"
              f"{dr['pi']:>+9.4f}{dr['transfer']:>+10.4f}{dr['q_gt_pa']:>+9.4f}{dr['oracle']:>+9.4f}")

    print("\nper-site (site: AC-AUROC / AUROC / reward@pi) — Emory is the only site that can move:")
    for tag in ("champion", "200hz"):
        print(f"  [{tag}]")
        for s, n, rwd, aca, au in res[tag][1]:
            print(f"     {lv.SITE_NAMES.get(s, s):<7} n={n:>4}  {aca:.4f} / {au:.4f} / {rwd:+.4f}")

    dac = res["200hz"][0]["age_auroc"] - res["champion"][0]["age_auroc"]
    dtr = res["200hz"][2]["transfer"] - res["champion"][2]["transfer"]
    print(f"\nDELTA (200hz - champion): AC-AUROC {dac:+.4f}   transfer-reward {dtr:+.4f}")
    print("\nDONE_RESAMPLE_AB")


if __name__ == "__main__":
    main()
