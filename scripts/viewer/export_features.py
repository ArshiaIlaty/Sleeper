"""Export a per-recording feature table (CSV) for model training.

Streams the CAISR annotation EDF + demographics for every recording in a dataset
and writes one wide row per recording: sleep macro-architecture, fragmentation /
transition dynamics, preprocessing deltas, event indices, demographics, and the
label. Uses the same `sources` abstraction as the viewer, so it runs against
either cohort:

    # standard (local) cohort -> eda/features_standard.csv
    python3 export_features.py --dataset standard --out eda/features_standard.csv

    # large (S3) cohort -> eda/features_large.csv  (streams small CAISR files)
    python3 export_features.py --dataset large --out eda/features_large.csv

Only the SMALL CAISR annotation files are read (a few hundred KB each), never the
170-460 MB physiological EDFs, so the whole large cohort completes in minutes.
Signal-derived autonomic features (ECG-HRV, SpO2) are intentionally out of scope
here — the modeling code (`team_code.py`) computes those from the raw waveforms.

Pure stdlib + numpy + edfio; no pandas (so it runs under any account, including
sudo). Resumable: pass --resume to skip recordings already present in --out.
"""
import os
import csv
import sys
import argparse

import numpy as np

from sources import get_dataset, REGISTRY
from preprocess import staging_report
from dynamics import (_recording_metrics, DYNAMIC_FIELDS_ORDER,
                      SECOND_ORDER_FIELDS_ORDER, second_order_vector)

EPOCH_SEC = 30.0
STAGE_CODES = {1: "N3", 2: "N2", 3: "N1", 4: "REM", 5: "Wake", 9: "Unknown"}
ASLEEP = {1, 2, 3, 4}
RESP_CODES = {1: "obstructive_apnea", 2: "central_apnea", 4: "hypopnea", 5: "RERA"}
LIMB_CODES = {1: "isolated_limb", 2: "periodic_limb"}
MIN_BOUT_EPOCHS = int(os.environ.get("CAISR_MIN_BOUT_EPOCHS", "2"))


def _event_rate(sig, hours, codes):
    if sig is None or len(sig) == 0 or not hours or hours <= 0:
        return None
    mask = np.isin(sig, list(codes) if not np.isscalar(codes) else [codes])
    n = int(np.count_nonzero(np.diff(mask.astype(int), prepend=0) == 1))
    return round(float(n / hours), 3)


def _psg_features(chans):
    """Sleep macro-architecture + event indices from CAISR channels.

    Mirrors scripts/eda/stats_sleep._psg_summary so the exported features match
    the cohort EDA and the viewer.
    """
    stage = chans.get("stage_caisr")
    resp = chans.get("resp_caisr")
    arousal = chans.get("arousal_caisr")
    limb = chans.get("limb_caisr")

    if resp is not None and len(resp) > 0:
        hours = len(resp) / 3600.0
    elif stage is not None and len(stage) > 0:
        hours = len(stage) * EPOCH_SEC / 3600.0
    else:
        hours = None

    d = {"duration_hours": round(hours, 3) if hours else None}
    tst_hours = None

    if stage is not None and len(stage) > 0:
        st = np.rint(stage).astype(int)
        n_epochs = st.size
        n_unknown = int(np.count_nonzero(st == 9))
        valid = st[st != 9]
        scored = valid.size
        d["n_epochs"] = int(n_epochs)
        d["pct_unknown_epochs"] = round(100.0 * n_unknown / n_epochs, 2) if n_epochs else None
        for code, name in STAGE_CODES.items():
            if code == 9:
                continue
            d[f"pct_{name}"] = round(100.0 * np.count_nonzero(valid == code) / scored, 2) if scored else None

        asleep_mask = np.isin(st, list(ASLEEP))
        n_asleep = int(np.count_nonzero(asleep_mask))
        tst_min = n_asleep * EPOCH_SEC / 60.0
        tst_hours = tst_min / 60.0
        d["tst_min"] = round(tst_min, 1)
        d["sleep_efficiency_pct"] = round(100.0 * n_asleep / n_epochs, 2) if n_epochs else None

        asleep_idx = np.where(asleep_mask)[0]
        onset = int(asleep_idx[0]) if asleep_idx.size else None
        d["sleep_latency_min"] = round(onset * EPOCH_SEC / 60.0, 1) if onset is not None else None
        if onset is not None:
            offset = int(asleep_idx[-1])
            waso = int(np.count_nonzero(st[onset:offset + 1] == 5))
            d["waso_min"] = round(waso * EPOCH_SEC / 60.0, 1)
            rem_idx = np.where(st == 4)[0]
            n3_idx = np.where(st == 1)[0]
            d["rem_latency_min"] = round((rem_idx[0] - onset) * EPOCH_SEC / 60.0, 1) if rem_idx.size else None
            d["n3_latency_min"] = round((n3_idx[0] - onset) * EPOCH_SEC / 60.0, 1) if n3_idx.size else None
        else:
            d["waso_min"] = d["rem_latency_min"] = d["n3_latency_min"] = None
        d["transitions_per_hr"] = round(float(np.count_nonzero(np.diff(valid) != 0)) / hours, 2) if hours else None

        counts = np.array([np.count_nonzero(valid == c) for c in (1, 2, 3, 4, 5)], float)
        p = counts / counts.sum() if counts.sum() > 0 else counts
        p = p[p > 0]
        d["stage_entropy"] = round(float(-np.sum(p * np.log(p)) / np.log(len(p))), 3) if p.size > 1 else 0.0

    per_hr = tst_hours if (tst_hours and tst_hours > 0) else hours
    d["ahi"] = _event_rate(resp, per_hr, (1, 2, 4))
    d["arousal_index"] = _event_rate(arousal, per_hr, (1,))
    d["plmi"] = _event_rate(limb, per_hr, (2,))
    for code, name in RESP_CODES.items():
        d[f"resp_{name}_idx"] = _event_rate(resp, per_hr, (code,))
    for code, name in LIMB_CODES.items():
        d[f"{name}_idx"] = _event_rate(limb, per_hr, (code,))
    return d


# Column order (stable across runs so colleagues get a consistent schema).
META_COLS = ["dataset", "bids_folder", "session", "site", "site_name", "label",
             "age", "sex", "race", "ethnicity", "bmi",
             "time_to_event", "time_to_last_visit"]
PSG_COLS = ["duration_hours", "n_epochs", "pct_unknown_epochs",
            "pct_Wake", "pct_N1", "pct_N2", "pct_N3", "pct_REM",
            "tst_min", "sleep_efficiency_pct", "sleep_latency_min", "waso_min",
            "rem_latency_min", "n3_latency_min", "transitions_per_hr", "stage_entropy",
            "ahi", "arousal_index", "plmi",
            "resp_obstructive_apnea_idx", "resp_central_apnea_idx",
            "resp_hypopnea_idx", "resp_RERA_idx",
            "isolated_limb_idx", "periodic_limb_idx"]
PREP_COLS = ["prep_epochs_changed", "prep_pct_changed",
             "prep_transitions_raw", "prep_transitions_clean", "prep_transitions_removed"]
DYN_COLS = list(DYNAMIC_FIELDS_ORDER)
# Fixed-length joint 2-step transition embedding (80 dims, same columns for every
# recording) — concatenable per-patient features for a linear probe / MLP.
SO2_COLS = list(SECOND_ORDER_FIELDS_ORDER)
ALL_COLS = META_COLS + PSG_COLS + PREP_COLS + DYN_COLS + SO2_COLS


def _demo_meta(ds, rec):
    site = rec.get("SiteID", "")
    return {
        "dataset": ds.key,
        "bids_folder": rec.get("BidsFolder", ""),
        "session": rec.get("SessionID", ""),
        "site": site,
        "site_name": {"S0001": "BIDMC", "I0002": "Emory", "I0006": "Kaiser"}.get(site, site),
        "label": rec.get("Cognitive_Impairment", ""),
        "age": rec.get("Age", ""), "sex": rec.get("Sex", ""),
        "race": rec.get("Race", ""), "ethnicity": rec.get("Ethnicity", ""),
        "bmi": rec.get("BMI", ""),
        "time_to_event": rec.get("Time_to_Event", ""),
        "time_to_last_visit": rec.get("Time_to_Last_Visit", ""),
    }


def _feature_row(ds, rec):
    """Compute one feature row for a demographics record; None if no CAISR."""
    bids = rec.get("BidsFolder", "")
    site, sess = rec.get("SiteID", ""), rec.get("SessionID", "1")
    edf = ds.open_caisr(site, bids, sess)
    if edf is None:
        return None
    chans = {s.label.strip(): np.asarray(s.data, float) for s in edf.signals}
    row = _demo_meta(ds, rec)
    row.update(_psg_features(chans))

    stage = chans.get("stage_caisr")
    if stage is not None and len(stage):
        rep = staging_report(np.rint(stage).astype(int), min_bout_epochs=MIN_BOUT_EPOCHS)
        s = rep["summary"]
        row["prep_epochs_changed"] = s["n_changed"]
        row["prep_pct_changed"] = s["pct_changed"]
        row["prep_transitions_raw"] = s["transitions_raw"]
        row["prep_transitions_clean"] = s["transitions_clean"]
        row["prep_transitions_removed"] = s["transitions_removed"]
        metrics, _ = _recording_metrics(np.rint(stage).astype(int))
        if metrics:
            for k in DYN_COLS:
                row[k] = metrics.get(k)
        # Fixed 80-dim joint 2-step transition embedding (raw staging, to match
        # the dynamics columns above). Always the same columns, so it is safe to
        # concatenate across recordings for a probe/MLP.
        row.update(second_order_vector(np.rint(stage).astype(int)))
    return row


def _load_done(path):
    """Return set of (bids, session) already written, for --resume."""
    done = set()
    if os.path.exists(path):
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                done.add((r.get("bids_folder", ""), r.get("session", "")))
    return done


def run(dataset_key, out_path, limit=None, resume=False, progress=None):
    ds = get_dataset(dataset_key)
    rows = ds.demographics()
    if limit:
        rows = rows[:limit]
    done = _load_done(out_path) if resume else set()
    mode = "a" if (resume and os.path.exists(out_path)) else "w"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    n_written = n_skip = n_nocaisr = n_err = 0
    with open(out_path, mode, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ALL_COLS, extrasaction="ignore")
        if mode == "w":
            w.writeheader()
        for i, rec in enumerate(rows):
            bids, sess = rec.get("BidsFolder", ""), rec.get("SessionID", "")
            if (bids, sess) in done:
                n_skip += 1
                continue
            try:
                row = _feature_row(ds, rec)
            except Exception as e:
                n_err += 1
                if progress:
                    progress(f"  ! {bids}: {type(e).__name__}: {e}")
                continue
            if row is None:
                n_nocaisr += 1
                continue
            w.writerow(row)
            n_written += 1
            if progress and (i + 1) % 50 == 0:
                fh.flush()
                progress(f"  {dataset_key}: {i+1}/{len(rows)} scanned, "
                         f"{n_written} written, {n_nocaisr} no-CAISR, {n_err} errors")
    summary = {"dataset": dataset_key, "out": out_path, "n_total": len(rows),
               "n_written": n_written, "n_skipped": n_skip,
               "n_no_caisr": n_nocaisr, "n_errors": n_err, "n_columns": len(ALL_COLS)}
    if progress:
        progress(f"DONE {summary}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="standard", choices=list(REGISTRY.keys()))
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--limit", type=int, default=None, help="cap recordings (debug)")
    ap.add_argument("--resume", action="store_true",
                    help="append, skipping recordings already in --out")
    args = ap.parse_args()
    summary = run(args.dataset, args.out, limit=args.limit, resume=args.resume,
                  progress=lambda m: print(m, file=sys.stderr, flush=True))
    print(",".join(ALL_COLS))
    print(f"Wrote {summary['n_written']} rows x {summary['n_columns']} cols to {args.out}")


if __name__ == "__main__":
    main()
