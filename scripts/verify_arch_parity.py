#!/usr/bin/env python3
"""Value-parity check: does team_code.extract_arch_features (the submission path)
produce the SAME 45 arch values as the validated arch_features_*.csv export that
the ablation trusted?

If yes, the ablation's measured +arch lift transfers to the submission unchanged
-- no need to re-run the 9h waveform LOSO to trust it. Compares on real records
by loading each record's CAISR annotation EDF through the exact team_code loader
(load_signal_data), calling extract_arch_features, and diffing against the CSV
row for the same (bids_folder, session).

Runs on the box (needs data + the vendored arch_features.py on PYTHONPATH).

Usage:
  PYTHONPATH=/data-temp/physio-viewer/bench/repo:<dir-with-team_code> \
  python3 verify_arch_parity.py \
    --data /data-temp/shared-physionet26-dataset/extracted \
    --arch-csv /data-temp/physio-viewer/exports/arch_features_standard.csv \
    --n 40
"""
from __future__ import annotations
import argparse, csv, os, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--arch-csv", required=True)
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    import team_code as tc
    import arch_features as af

    # index the validated CSV by (bids_folder, session)
    ref = {}
    with open(args.arch_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            ref[(r.get("bids_folder", ""), str(r.get("session", "")))] = r
    cols = list(af.ARCH_FEATURE_COLUMNS)
    print(f"ref rows: {len(ref)}  arch cols: {len(cols)}", flush=True)

    demo_file = os.path.join(args.data, tc.DEMOGRAPHICS_FILE)
    records = tc.find_patients(demo_file)
    print(f"records: {len(records)}", flush=True)

    checked = mism = missing = 0
    max_abs = 0.0
    worst = None
    for rec in records:
        if checked >= args.n:
            break
        pid = rec[tc.HEADERS["bids_folder"]]
        sid = rec[tc.HEADERS["site_id"]]
        sess = str(rec[tc.HEADERS["session_id"]])
        key = (pid, sess)
        if key not in ref:
            continue
        algo_file = tc._resolve_edf(
            os.path.join(args.data, tc.ALGORITHMIC_ANNOTATIONS_SUBFOLDER, sid),
            pid, sess, suffix="_caisr_annotations")
        if not (algo_file and os.path.exists(algo_file)):
            missing += 1
            continue
        algo, algo_fs = tc.load_signal_data(algo_file)
        arousal_fs = (algo_fs or {}).get("arousal_caisr")
        vec, names = tc.extract_arch_features(algo, arousal_fs)
        rrow = ref[key]
        checked += 1
        # compare each column: both NaN -> ok; else abs diff within tol
        rec_mism = 0
        for i, c in enumerate(cols):
            got = float(vec[i])
            raw = rrow.get(c, "")
            exp = float(raw) if raw not in ("", None) else np.nan
            if np.isnan(got) and np.isnan(exp):
                continue
            if np.isnan(got) != np.isnan(exp):
                rec_mism += 1
                continue
            d = abs(got - exp)
            if d > max_abs:
                max_abs, worst = d, (pid, c, got, exp)
            if d > 1e-3:
                rec_mism += 1
        if rec_mism:
            mism += 1
            if mism <= 5:
                print(f"  MISMATCH {pid} ({rec_mism} cols)", flush=True)

    print("\n=== PARITY SUMMARY ===", flush=True)
    print(f"checked={checked}  records_with_mismatch={mism}  no_caisr={missing}", flush=True)
    print(f"max abs diff across all cols/records: {max_abs:.2e}", flush=True)
    if worst:
        print(f"worst: {worst[0]} col={worst[1]} got={worst[2]} exp={worst[3]}", flush=True)
    print("PARITY_OK" if mism == 0 else "PARITY_FAIL", flush=True)


if __name__ == "__main__":
    main()
