"""Export the LONG per-epoch feature tables (the fully un-aggregated view).

Companion to the wide per-recording CSVs. Opens each physio EDF once, decodes one
central EEG channel, and writes TWO long tables into an output directory:

  <outdir>/epoch_eeg_<cohort>.csv      one row per (recording, stage, epoch):
      identifiers + per-epoch abs/rel band power, Theta/Alpha, Delta/Sigma,
      REM-slowing, and (left-joined on the capped subsample) sample/permutation
      entropy. `has_complexity`=1 marks the epochs that carry entropy.
  <outdir>/spindles_<cohort>.csv       one row per detected spindle:
      identifiers + stage, start_s, duration_s, amp_uv (+ per-recording threshold).

Both are keyed by (dataset, bids_folder, session) so they join to the wide CSVs and
to demographics. Row counts are large (~hundreds of epochs / tens of spindles per
recording), which is the point: this is for EDA, distribution plots, and
sequence/temporal models, not for the per-recording tabular model.

    python3 export_epoch_features.py --dataset standard --outdir exports/per_epoch
    python3 export_epoch_features.py --dataset large    --outdir exports/per_epoch --resume
    # skip the O(n^2) entropy pass (spectral + spindles only, much faster):
    python3 export_epoch_features.py --dataset standard --outdir exports/per_epoch --no-complexity

Pure stdlib + numpy + edfio (+ scipy/neurokit via the reused modules). Resumable at
the recording grain: --resume skips recordings already present in the EEG table.
--shard/--nshards stride-slice the cohort for parallel workers (each needs its own
PHYSIO_CACHE_DIR on the large cohort).
"""
import os
import csv
import sys
import time
import argparse

import numpy as np
import edfio

from sources import get_dataset, REGISTRY
from app import channel_role, _caisr_stage_codes, _NK_EEG_PREFER
import nk_features as nkf
import epoch_features as ef

_EEG_ROLES = {"eeg"}
BANDS = ef.BANDS

# ---- stable column layouts ----
ID_COLS = ["dataset", "bids_folder", "session", "site", "label", "eeg_channel"]

EEG_EPOCH_COLS = ID_COLS + ["stage", "epoch_index", "start_s"]
for _b in BANDS:
    EEG_EPOCH_COLS.append(f"abs_{_b}")
for _b in BANDS:
    EEG_EPOCH_COLS.append(f"rel_{_b}")
EEG_EPOCH_COLS += ["theta_alpha", "delta_sigma", "rem_slowing",
                   "has_complexity", "sampen", "permen"]

SPINDLE_COLS = ID_COLS + ["stage", "start_s", "duration_s", "amp_uv",
                          "threshold_uv"]


def _ids(ds, rec, bids, site, sess, eeg_ch):
    return {
        "dataset": ds.key, "bids_folder": bids, "session": sess, "site": site,
        "label": rec.get("Cognitive_Impairment", ""), "eeg_channel": eeg_ch or "",
    }


def _load_eeg(edf):
    labels = [s.label.strip() for s in edf.signals]
    roles = {l: channel_role(l) for l in labels}
    eeg = nkf._pick_channel(labels, roles, _EEG_ROLES, prefer=_NK_EEG_PREFER)
    if eeg is None:
        return None, None, None
    for s in edf.signals:
        if s.label.strip() == eeg:
            return eeg, np.asarray(s.data, float), float(s.sampling_frequency)
    return None, None, None


def _recording_rows(ds, rec, with_complexity=True):
    """Return (eeg_epoch_rows, spindle_rows) for one recording, or (None, None)."""
    bids = rec.get("BidsFolder", "")
    site, sess = rec.get("SiteID", ""), rec.get("SessionID", "1")
    codes, _ = _caisr_stage_codes(ds, rec, bids)
    if codes is None:
        return None, None
    try:
        f = ds.physio_path(site, bids, sess)
    except Exception:
        f = None
    if not f:
        return None, None

    edf = edfio.read_edf(f, lazy_load_data=True)
    eeg_ch, sig, fs = _load_eeg(edf)
    ids = _ids(ds, rec, bids, site, sess, eeg_ch)
    if eeg_ch is None:
        return [], []

    spectral = ef.eeg_spectral_epochs(sig, fs, codes)
    complexity = (ef.eeg_complexity_epochs(sig, fs, codes)
                  if with_complexity else {})
    eeg_rows = []
    for row in spectral:
        r = dict(ids)
        r.update(row)
        c = complexity.get(row["epoch_index"])
        if c:
            r["has_complexity"] = 1
            r["sampen"] = c["sampen"]
            r["permen"] = c["permen"]
        else:
            r["has_complexity"] = 0
            r["sampen"] = None
            r["permen"] = None
        eeg_rows.append(r)

    events, thr = ef.spindle_events(sig, fs, codes)
    spindle_rows = []
    for ev in events:
        r = dict(ids)
        r.update(ev)
        r["threshold_uv"] = thr
        spindle_rows.append(r)
    return eeg_rows, spindle_rows


def _done_recordings(path):
    """Set of (bids, session) already present in the EEG-epoch table."""
    done = set()
    if os.path.exists(path):
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                done.add((r.get("bids_folder", ""), r.get("session", "")))
    return done


def run(dataset_key, outdir, limit=None, resume=False, with_complexity=True,
        shard=0, nshards=1, progress=None):
    ds = get_dataset(dataset_key)
    rows = ds.demographics()
    if nshards > 1:
        rows = [r for i, r in enumerate(rows) if i % nshards == shard]
    if limit:
        rows = rows[:limit]

    os.makedirs(outdir, exist_ok=True)
    suffix = dataset_key if nshards == 1 else f"{dataset_key}_s{shard}"
    eeg_path = os.path.join(outdir, f"epoch_eeg_{suffix}.csv")
    spin_path = os.path.join(outdir, f"spindles_{suffix}.csv")

    done = _done_recordings(eeg_path) if resume else set()
    mode = "a" if (resume and os.path.exists(eeg_path)) else "w"

    n_rec = n_skip = n_nofile = n_err = 0
    n_eeg_rows = n_spin_rows = 0
    with open(eeg_path, mode, newline="") as efh, open(spin_path, mode, newline="") as sfh:
        ew = csv.DictWriter(efh, fieldnames=EEG_EPOCH_COLS, extrasaction="ignore")
        sw = csv.DictWriter(sfh, fieldnames=SPINDLE_COLS, extrasaction="ignore")
        if mode == "w":
            ew.writeheader()
            sw.writeheader()
        for i, rec in enumerate(rows):
            bids, sess = rec.get("BidsFolder", ""), rec.get("SessionID", "")
            if (bids, sess) in done:
                n_skip += 1
                continue
            try:
                eeg_rows, spin_rows = _recording_rows(ds, rec, with_complexity)
            except Exception as e:
                n_err += 1
                if progress:
                    progress(f"  ! {bids}: {type(e).__name__}: {e}")
                continue
            if eeg_rows is None:
                n_nofile += 1
                continue
            for r in eeg_rows:
                ew.writerow(r)
            for r in spin_rows:
                sw.writerow(r)
            n_rec += 1
            n_eeg_rows += len(eeg_rows)
            n_spin_rows += len(spin_rows)
            if progress and (i + 1) % 10 == 0:
                efh.flush(); sfh.flush()
                progress(f"  {dataset_key}: {i+1}/{len(rows)} scanned, "
                         f"{n_rec} recs -> {n_eeg_rows} epoch rows, "
                         f"{n_spin_rows} spindles, {n_nofile} no-file, {n_err} errors")
    summary = {"dataset": dataset_key, "eeg_table": eeg_path, "spindle_table": spin_path,
               "n_recordings": n_rec, "n_skipped": n_skip, "n_no_file": n_nofile,
               "n_errors": n_err, "n_epoch_rows": n_eeg_rows, "n_spindle_rows": n_spin_rows}
    if progress:
        progress(f"DONE {summary}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="standard", choices=list(REGISTRY.keys()))
    ap.add_argument("--outdir", required=True, help="output directory for the long CSVs")
    ap.add_argument("--limit", type=int, default=None, help="cap recordings (debug)")
    ap.add_argument("--resume", action="store_true",
                    help="append, skipping recordings already in the EEG table")
    ap.add_argument("--no-complexity", dest="with_complexity", action="store_false",
                    help="skip the O(n^2) per-epoch entropy pass")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    summary = run(args.dataset, args.outdir, limit=args.limit, resume=args.resume,
                  with_complexity=args.with_complexity,
                  shard=args.shard, nshards=args.nshards,
                  progress=lambda m: print(m, file=sys.stderr, flush=True))
    print(f"Wrote {summary['n_epoch_rows']} epoch rows + {summary['n_spindle_rows']} "
          f"spindles from {summary['n_recordings']} recordings to {args.outdir}")


if __name__ == "__main__":
    main()
