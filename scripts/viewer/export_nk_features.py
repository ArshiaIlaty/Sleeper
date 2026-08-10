"""Export per-sleep-stage NeuroKit2 features (CSV) for model training.

Companion to `export_features.py`. Where that script reads only the small CAISR
annotation files (sleep macro-architecture + event indices), THIS one reads the
big physiological EDFs and, per recording, extracts **per-stage** heart-rate
variability (ECG), EEG complexity, and respiratory rate/variability via
`nk_features.py`. The hypothesis: cognitive impairment is expressed less in a
night's average physiology than in how physiology is *modulated across sleep
stages* (blunted autonomic swing, reduced slow-wave EEG complexity), so every
feature is stage-resolved and the strongest features are cross-stage contrasts.

Only the channels actually needed (one ECG, one EEG, one respiratory-effort) are
decoded from each EDF — never the full montage — and the whole-night R-peak
detection + LINEAR HRV is what keeps it tractable (the O(n^2) nonlinear HRV /
Higuchi metrics are deliberately avoided; see nk_features.py). Still, this is far
heavier than the CAISR-only export (~5-10 s/recording locally, plus the S3
download for the large cohort), so run it in tmux / nohup for the large dataset.

    # standard (local) cohort
    python3 export_nk_features.py --dataset standard --out exports/nk_features_standard.csv

    # large (S3) cohort — downloads each physio EDF to the size-capped cache
    python3 export_nk_features.py --dataset large --out exports/nk_features_large.csv --resume

Pure stdlib + numpy + edfio + neurokit2 (no pandas here — csv module only — so it
runs under any data-capable account). Resumable: --resume skips recordings already
in --out. Merge with features_*.csv on (bids_folder, session) to train on the
union of sleep-architecture and per-stage physiology features.
"""
import os
import csv
import sys
import time
import argparse

import numpy as np
import edfio

from sources import get_dataset, REGISTRY
from app import channel_role, _caisr_stage_codes
import nk_features as nkf

# The physiological channel roles we need; we decode one channel per role group.
_ECG_ROLES = {"ecg"}
_EEG_ROLES = {"eeg"}
_RSP_ROLES = {"effort", "airflow"}
_EEG_PREFER = ["c3", "c4", "o1", "o2"]           # central > occipital for slow-wave


META_COLS = ["dataset", "bids_folder", "session", "site", "label",
             "ecg_channel", "eeg_channel", "rsp_channel",
             "n_beats_total", "n_beats_clean", "nk_seconds"]
NK_COLS = list(nkf.NK_FEATURE_COLUMNS)
ALL_COLS = META_COLS + NK_COLS


def _load_needed_channels(edf):
    """Decode only one channel per needed role from a lazily-opened EDF.

    Returns (channels, fss, roles) where channels holds just the selected ECG /
    EEG / respiratory channels' samples (the big montage is never fully decoded).
    """
    labels = [s.label.strip() for s in edf.signals]
    roles = {l: channel_role(l) for l in labels}
    ecg = nkf._pick_channel(labels, roles, _ECG_ROLES)
    eeg = nkf._pick_channel(labels, roles, _EEG_ROLES, prefer=_EEG_PREFER)
    rsp = nkf._pick_channel(labels, roles, _RSP_ROLES)
    keep = {c for c in (ecg, eeg, rsp) if c is not None}
    channels, fss = {}, {}
    for s in edf.signals:
        lab = s.label.strip()
        if lab in keep and lab not in channels:
            channels[lab] = np.asarray(s.data, float)   # decodes THIS channel only
            fss[lab] = float(s.sampling_frequency)
    return channels, fss, roles


def _feature_row(ds, rec):
    """Compute one NeuroKit feature row for a demographics record.

    Returns a dict, or None if the recording has no physio EDF or no staging.
    """
    bids = rec.get("BidsFolder", "")
    site, sess = rec.get("SiteID", ""), rec.get("SessionID", "1")

    codes, reason = _caisr_stage_codes(ds, rec, bids)
    if codes is None:
        return None
    try:
        f = ds.physio_path(site, bids, sess)
    except Exception:
        f = None
    if not f:
        return None

    edf = edfio.read_edf(f, lazy_load_data=True)
    channels, fss, roles = _load_needed_channels(edf)
    if not channels:
        return None

    t0 = time.time()
    nk_out = nkf.nk_stage_features(channels, fss, roles, codes)
    nk_sec = round(time.time() - t0, 2)

    used = nk_out.get("channels_used", {})
    ecg = nk_out.get("ecg") or {}
    row = {
        "dataset": ds.key, "bids_folder": bids, "session": sess,
        "site": site, "label": rec.get("Cognitive_Impairment", ""),
        "ecg_channel": used.get("ecg", ""),
        "eeg_channel": used.get("eeg", ""),
        "rsp_channel": used.get("rsp", ""),
        "n_beats_total": ecg.get("n_beats_total"),
        "n_beats_clean": ecg.get("n_beats_clean"),
        "nk_seconds": nk_sec,
    }
    row.update(nkf.flatten_features(nk_out))
    return row


def _load_done(path):
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

    n_written = n_skip = n_nofile = n_err = 0
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
                n_nofile += 1
                continue
            w.writerow(row)
            n_written += 1
            if progress and (i + 1) % 10 == 0:
                fh.flush()
                progress(f"  {dataset_key}: {i+1}/{len(rows)} scanned, "
                         f"{n_written} written, {n_nofile} no-file/staging, {n_err} errors")
    summary = {"dataset": dataset_key, "out": out_path, "n_total": len(rows),
               "n_written": n_written, "n_skipped": n_skip,
               "n_no_file": n_nofile, "n_errors": n_err, "n_columns": len(ALL_COLS)}
    if progress:
        progress(f"DONE {summary}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="standard", choices=list(REGISTRY.keys()))
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--limit", type=int, default=None, help="cap recordings (debug)")
    ap.add_argument("--resume", action="store_true",
                    help="append, skipping recordings already in --out")
    args = ap.parse_args()
    summary = run(args.dataset, args.out, limit=args.limit, resume=args.resume,
                  progress=lambda m: print(m, file=sys.stderr, flush=True))
    print(f"Wrote {summary['n_written']} rows x {summary['n_columns']} cols to {args.out}")


if __name__ == "__main__":
    main()
