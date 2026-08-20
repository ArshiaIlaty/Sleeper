#!/usr/bin/env python3
"""Value-parity check for the Track 2 in-container signal-feature block.

Does team_code.extract_signal_features (the submission path) reproduce the
validated offline exports (nk_features_*.csv + report_features_*.csv +
micro_features_*.csv) that the ablations trusted?

Compares on real records: for each record it loads the physio EDF + the CAISR
annotation EDF through the EXACT team_code loader (load_signal_data), calls
extract_signal_features, and diffs the 336-value vector against the three CSV
rows for the same (bids_folder, session), matched by block prefix
(nk__ / report__ / micro__ strip to the raw column name in each CSV).

Reports per-block: NaN-pattern agreement (the wiring check — a mismatch here is
a real bug) and, among cols both non-NaN, the max/median relative diff (the
value check). Small HRV drift is expected because the offline CSVs were built
with neurokit2 0.2.13 and the container pins 0.2.10; a WIRING bug shows up as
widespread NaN-pattern disagreement or whole blocks off, not a few % HRV drift.

Runs on the box (needs data + track2/ on PYTHONPATH).

Usage:
  PYTHONPATH=/data-temp/physio-viewer/bench/repo:<track2-dir> \
  python3 verify_sig_parity.py \
    --data /data-temp/shared-physionet26-dataset/extracted \
    --nk-csv    /data-temp/physio-viewer/exports/nk_features_standard.csv \
    --report-csv /data-temp/physio-viewer/exports/report_features_standard.csv \
    --micro-csv  /data-temp/physio-viewer/exports/micro_features_standard.csv \
    --n 20
"""
from __future__ import annotations
import argparse
import csv
import os

import numpy as np


def _index_csv(path):
    ref = {}
    if not path or not os.path.exists(path):
        return ref
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            ref[(r.get("bids_folder", ""), str(r.get("session", "")))] = r
    return ref


def _cell(row, col):
    raw = row.get(col, "") if row else ""
    if raw in ("", None):
        return np.nan
    try:
        return float(raw)
    except (TypeError, ValueError):
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--nk-csv", default="")
    ap.add_argument("--report-csv", default="")
    ap.add_argument("--micro-csv", default="")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--rel-tol", type=float, default=0.02,
                    help="relative-diff tolerance for a value 'match' (default 2%)")
    args = ap.parse_args()

    import team_code as tc
    import sig_features as sf

    refs = {
        "nk__": _index_csv(args.nk_csv),
        "report__": _index_csv(args.report_csv),
        "micro__": _index_csv(args.micro_csv),
    }
    cols = list(sf.SIGNAL_FEATURE_COLUMNS)
    print(f"sig cols: {len(cols)}  "
          f"nk_ref={len(refs['nk__'])} report_ref={len(refs['report__'])} "
          f"micro_ref={len(refs['micro__'])}", flush=True)

    demo_file = os.path.join(args.data, tc.DEMOGRAPHICS_FILE)
    records = tc.find_patients(demo_file)
    print(f"records: {len(records)}", flush=True)

    # per-block accumulators
    stats = {p: {"both_nan": 0, "both_val": 0, "nan_disagree": 0,
                 "val_mismatch": 0, "rel_diffs": [], "n_cols_checked": 0}
             for p in refs}
    checked = 0

    for rec in records:
        if checked >= args.n:
            break
        pid = rec[tc.HEADERS["bids_folder"]]
        sid = rec[tc.HEADERS["site_id"]]
        sess = str(rec[tc.HEADERS["session_id"]])
        key = (pid, sess)
        # need at least one ref block for this record
        if not any(key in refs[p] for p in refs):
            continue

        phys_file = tc._resolve_edf(
            os.path.join(args.data, tc.PHYSIOLOGICAL_DATA_SUBFOLDER, sid), pid, sess)
        algo_file = tc._resolve_edf(
            os.path.join(args.data, tc.ALGORITHMIC_ANNOTATIONS_SUBFOLDER, sid),
            pid, sess, suffix="_caisr_annotations")
        if not (phys_file and os.path.exists(phys_file)
                and algo_file and os.path.exists(algo_file)):
            continue

        phys, phys_fs = tc.load_signal_data(phys_file)
        algo, algo_fs = tc.load_signal_data(algo_file)
        vec, names = tc.extract_signal_features(phys, phys_fs, algo, algo_fs)
        del phys, algo
        checked += 1

        for i, full in enumerate(names):
            prefix = next((p for p in refs if full.startswith(p)), None)
            if prefix is None:
                continue
            row = refs[prefix].get(key)
            if row is None:
                continue
            raw_col = full[len(prefix):]
            got = float(vec[i])
            exp = _cell(row, raw_col)
            st = stats[prefix]
            st["n_cols_checked"] += 1
            gn, en = np.isnan(got), np.isnan(exp)
            if gn and en:
                st["both_nan"] += 1
            elif gn != en:
                st["nan_disagree"] += 1
            else:
                st["both_val"] += 1
                denom = max(abs(exp), 1e-9)
                rel = abs(got - exp) / denom
                st["rel_diffs"].append(rel)
                if rel > args.rel_tol and abs(got - exp) > 1e-6:
                    st["val_mismatch"] += 1

    print(f"\nchecked {checked} records\n", flush=True)
    ok = True
    for p in ("nk__", "report__", "micro__"):
        st = stats[p]
        if st["n_cols_checked"] == 0:
            print(f"[{p}] no ref rows matched — SKIPPED", flush=True)
            continue
        rd = np.asarray(st["rel_diffs"]) if st["rel_diffs"] else np.array([0.0])
        nan_rate = st["nan_disagree"] / st["n_cols_checked"]
        print(f"[{p}] cols/rec-checked={st['n_cols_checked']}  "
              f"both_nan={st['both_nan']}  both_val={st['both_val']}  "
              f"NaN-disagree={st['nan_disagree']} ({nan_rate:.1%})  "
              f"val-mismatch>{args.rel_tol:.0%}={st['val_mismatch']}", flush=True)
        print(f"      rel-diff (both non-NaN): median={np.median(rd):.2e} "
              f"p90={np.percentile(rd, 90):.2e} max={rd.max():.2e}", flush=True)
        # WIRING gate: NaN-pattern must broadly agree. A few % is fine (version
        # drift can null a marginal HRV cell); >10% signals a real wiring bug.
        if nan_rate > 0.10:
            ok = False
            print(f"      !! NaN-pattern disagreement {nan_rate:.1%} > 10% "
                  f"-> WIRING SUSPECT", flush=True)

    print("\nSIG_PARITY_OK" if ok else "\nSIG_PARITY_FAIL", flush=True)


if __name__ == "__main__":
    main()
