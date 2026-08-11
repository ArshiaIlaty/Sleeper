#!/usr/bin/env python3
"""Build local .npz feature caches for LOSO / 70-15-15 on the standard cohort.

Reproduces team_code's OWN feature matrix (preset `extract_all`: demographics +
autonomic ECG-HRV/SpO2 + CAISR) by running team_code.extract_all_features over the
local standard dataset -- this is the matrix that produced the ~0.168 leaderboard
model, and it includes the autonomic block that features_standard.csv omits. Then
it joins the new signal features (per-stage NeuroKit + clinical-report EEG spectral/
spindles/oxygenation/resp-events/REM density) by patient id and writes TWO caches:

  <out>/feature_matrix_local_baseline.npz  -- team_code extract_all only
  <out>/feature_matrix_local_plus.npz      -- extract_all + nk + report

Each cache carries X, y, ages, sites, pids, feature_names (same schema as the S3
cache) so scripts/cross_validate_s3.py and scripts/train_val_test_eval.py can run
on it unchanged via the local_cache shim.

Usage (on pdmle, as arshia_ilaty_physio26):
    python3 build_local_feature_cache.py \
        --data-root /data-temp/shared-physionet26-dataset/extracted \
        --exports   /data-temp/physio-viewer/exports \
        --repo      /data-temp/physio-viewer/bench/repo \
        --out       /data-temp/physio-viewer/bench/cache
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np


def _f(s):
    try:
        v = float(s)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def load_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# columns from the new CSVs that are identifiers / bookkeeping, never features
NON_FEATURE = {
    "dataset", "bids_folder", "session", "site", "site_name", "label", "age",
    "sex", "race", "ethnicity", "bmi", "time_to_event", "time_to_last_visit",
    "ecg_channel", "eeg_channel", "rsp_channel", "spo2_channel", "eog_channel",
    "has_resp_caisr", "report_seconds", "nk_seconds",
    "n_beats_total", "n_beats_clean",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--exports", required=True)
    ap.add_argument("--repo", required=True,
                    help="dir holding team_code.py + helpers (repo checkout)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    # import team_code from the repo checkout
    if args.repo not in sys.path:
        sys.path.insert(0, args.repo)
    from helper_code import HEADERS, DEMOGRAPHICS_FILE, find_patients, load_demographics, load_age, load_diagnoses
    import team_code as tc

    demo_file = os.path.join(args.data_root, DEMOGRAPHICS_FILE)
    records = find_patients(demo_file)
    if args.limit:
        records = records[:args.limit]

    # --- load the new-feature CSVs, keyed by bids_folder (session-1 standard) ---
    nk = {r.get("bids_folder", ""): r for r in load_csv(os.path.join(args.exports, "nk_features_standard.csv"))}
    rep = {r.get("bids_folder", ""): r for r in load_csv(os.path.join(args.exports, "report_features_standard.csv"))}
    nk_cols = [c for c in next(iter(nk.values())).keys() if c not in NON_FEATURE]
    rep_cols = [c for c in next(iter(rep.values())).keys() if c not in NON_FEATURE]

    X_base, X_plus, y, ages, sites, pids = [], [], [], [], [], []
    base_names = None
    plus_names = None
    n_nk_miss = n_rep_miss = 0

    from tqdm import tqdm
    for rec in tqdm(records, unit="rec"):
        pid = rec[HEADERS["bids_folder"]]
        sess = rec[HEADERS["session_id"]]
        try:
            label = load_diagnoses(demo_file, pid)
        except Exception:
            continue
        if label not in (0, 1):
            continue
        try:
            feats, names = tc.extract_all_features(
                rec, args.data_root, tc.DEFAULT_CSV_PATH, preset=tc.EXTRACT_CACHE_PRESET)
        except Exception as e:
            tqdm.write(f"  ! extract fail {pid}: {type(e).__name__}: {e}")
            continue
        feats = np.asarray(feats, dtype=np.float32).ravel()
        if base_names is None:
            base_names = list(names)

        # join new features by pid
        n = nk.get(pid)
        nk_vec = np.array([_f(n.get(c)) if n else np.nan for c in nk_cols], dtype=np.float32)
        if n is None:
            n_nk_miss += 1
        p = rep.get(pid)
        rep_vec = np.array([_f(p.get(c)) if p else np.nan for c in rep_cols], dtype=np.float32)
        if p is None:
            n_rep_miss += 1

        plus_vec = np.concatenate([feats, nk_vec, rep_vec]).astype(np.float32)
        if plus_names is None:
            plus_names = list(names) + ["nk__" + c for c in nk_cols] + ["rep__" + c for c in rep_cols]

        X_base.append(feats)
        X_plus.append(plus_vec)
        y.append(int(label))
        sites.append(rec[HEADERS["site_id"]])
        pids.append(pid)
        try:
            demo = load_demographics(demo_file, pid, sess)
            ages.append(float(load_age(demo)))
        except Exception:
            ages.append(float("nan"))

    # some recordings may miss a modality → pad base rows to common width
    base_w = max(len(r) for r in X_base)
    Xb = np.full((len(X_base), base_w), np.nan, dtype=np.float32)
    for i, r in enumerate(X_base):
        Xb[i, :len(r)] = r
    plus_w = max(len(r) for r in X_plus)
    Xp = np.full((len(X_plus), plus_w), np.nan, dtype=np.float32)
    for i, r in enumerate(X_plus):
        Xp[i, :len(r)] = r

    y = np.asarray(y, dtype=np.int8)
    ages = np.asarray(ages, dtype=np.float32)
    sites = np.asarray(sites)
    pids = np.asarray(pids)

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "feature_matrix_local_baseline.npz"),
                        X=Xb, y=y, ages=ages, sites=sites, pids=pids,
                        feature_names=np.asarray(base_names))
    np.savez_compressed(os.path.join(args.out, "feature_matrix_local_plus.npz"),
                        X=Xp, y=y, ages=ages, sites=sites, pids=pids,
                        feature_names=np.asarray(plus_names))

    print(f"\ncohort: {len(y)} recordings | CI+={int(y.sum())} CI-={int((1-y).sum())} "
          f"prevalence={y.mean():.3f}")
    print(f"sites: {dict(zip(*np.unique(sites, return_counts=True)))}")
    print(f"baseline features (team_code extract_all): {Xb.shape[1]}")
    print(f"plus features (+nk {len(nk_cols)} +report {len(rep_cols)}): {Xp.shape[1]}")
    print(f"nk rows missing: {n_nk_miss} | report rows missing: {n_rep_miss}")
    print(f"wrote caches to {args.out}")


if __name__ == "__main__":
    main()
