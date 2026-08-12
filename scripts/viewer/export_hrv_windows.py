"""Export NON-AVERAGED per-stage HRV — the long windowed-HRV table + its wide
dispersion companion, from a single ECG pass per recording.

`nk_features_*.csv` carries ONE HRV number per stage (the pooled mean). This adds
the un-aggregated view HRV needs (HRV is undefined per 30 s epoch, so the grain is
a short time window of beats):

  <outdir>/hrv_windows_<cohort>.csv     LONG · one row per (recording, stage, window)
      identifiers + stage, window_index, start_s, dur_s, n_beats, and the linear
      HRV metrics (hr_mean, sdnn, rmssd, pnn50, sdsd, cvnn, sd1/sd2/sd1sd2,
      lf/hf/lfhf/lfn/hfn/tp) for that ~120 s window.
  <outdir>/hrv_dispersion_<cohort>.csv  WIDE · one row per recording
      per-stage SPREAD of those per-window HRV metrics (SD/CV + p10/p50/p90/IQR for
      the flagship hr_mean/rmssd/sdnn/lfhf; SD/CV for the rest) — the non-avg wide
      companion that slots beside dispersion_features_<cohort>.csv.

Both key on (dataset, bids_folder, session) so they merge with the other CSVs and
demographics. Decodes only ONE ECG channel. R-peak detection + RR filtering +
per-window HRV all reuse `nk_features`/`hrv_windows`, so a window's HRV is defined
identically to the pooled per-stage HRV — only the aggregation grain differs.

    python3 export_hrv_windows.py --dataset standard --outdir exports/hrv
    python3 export_hrv_windows.py --dataset large    --outdir exports/hrv --resume
    # cheaper time-domain-only pass (skip hrv_frequency lf/hf/...):
    python3 export_hrv_windows.py --dataset standard --outdir exports/hrv --no-freq

Pure stdlib + numpy + edfio + neurokit2. Resumable at the recording grain
(--resume skips recordings already in the long table). --shard/--nshards
stride-slice the cohort for parallel workers (each needs its own PHYSIO_CACHE_DIR
on the large cohort).
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
import hrv_windows as hw

_ECG_ROLES = {"ecg"}

STAGE_TAGS = hw.STAGE_TAGS                          # wake n1 n2 n3 rem
DISP_FULL = hw.DISP_FULL                            # SD/CV + percentiles
DISP_COMPACT = hw.DISP_COMPACT                      # SD/CV only
STATS_FULL = ["sd", "cv", "p10", "p50", "p90", "iqr"]
STATS_COMPACT = ["sd", "cv"]

ID_COLS = ["dataset", "bids_folder", "session", "site", "label", "ecg_channel"]

# ---- LONG table columns ----
LONG_COLS = ID_COLS + ["stage", "window_index", "start_s", "dur_s", "n_beats"] \
    + list(hw.HRV_LONG_METRICS)

# ---- WIDE dispersion columns ----
META_COLS = ID_COLS + ["with_freq", "hrv_seconds"]


def _wide_cols():
    cols = list(META_COLS)
    for t in STAGE_TAGS:
        cols.append(f"hrv_{t}_n_windows")
        for m in DISP_FULL:
            cols += [f"hrv_{t}_{m}_{s}" for s in STATS_FULL]
        for m in DISP_COMPACT:
            cols += [f"hrv_{t}_{m}_{s}" for s in STATS_COMPACT]
    return cols


WIDE_COLS = _wide_cols()


def _ids(ds, rec, bids, site, sess, ecg_ch):
    return {"dataset": ds.key, "bids_folder": bids, "session": sess, "site": site,
            "label": rec.get("Cognitive_Impairment", ""), "ecg_channel": ecg_ch or ""}


def _load_ecg(edf):
    labels = [s.label.strip() for s in edf.signals]
    roles = {l: channel_role(l) for l in labels}
    ecg = nkf._pick_channel(labels, roles, _ECG_ROLES)
    if ecg is None:
        return None, None, None
    for s in edf.signals:
        if s.label.strip() == ecg:
            return ecg, np.asarray(s.data, float), float(s.sampling_frequency)
    return None, None, None


def _flatten_disp(res, ids, with_freq, hrv_sec):
    """Build the wide dispersion row from hrv_dispersion output."""
    row = dict(ids)
    row["with_freq"] = int(bool(with_freq))
    row["hrv_seconds"] = hrv_sec
    stages = (res or {}).get("stages") or {}
    for t in STAGE_TAGS:
        s = stages.get(t) or {}
        row[f"hrv_{t}_n_windows"] = s.get("n_windows")
        for m in DISP_FULL:
            d = s.get(m) or {}
            for st in STATS_FULL:
                row[f"hrv_{t}_{m}_{st}"] = d.get(st)
        for m in DISP_COMPACT:
            d = s.get(m) or {}
            for st in STATS_COMPACT:
                row[f"hrv_{t}_{m}_{st}"] = d.get(st)
    return row


def _recording(ds, rec, with_freq=True):
    """Return (long_rows, wide_row) for one recording, or (None, None)."""
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
    ecg_ch, sig, fs = _load_ecg(edf)
    ids = _ids(ds, rec, bids, site, sess, ecg_ch)
    if ecg_ch is None:
        return [], _flatten_disp({}, ids, with_freq, 0.0)

    t0 = time.time()
    win = hw.hrv_windows_by_stage(sig, fs, codes, do_freq=with_freq)
    # dispersion from the same windows (no second R-peak detection)
    by_stage = {}
    for r in win.get("windows", []):
        by_stage.setdefault(r["stage"], []).append(r)
    disp_stages = {}
    for t in STAGE_TAGS:
        wins = by_stage.get(t)
        if not wins:
            continue
        s = {"n_windows": len(wins)}
        for m in DISP_FULL:
            s[m] = hw._disp([x.get(m) for x in wins], nd=4, full=True)
        for m in DISP_COMPACT:
            s[m] = hw._disp([x.get(m) for x in wins], nd=4, full=False)
        disp_stages[t] = s
    hrv_sec = round(time.time() - t0, 2)

    long_rows = []
    for r in win.get("windows", []):
        row = dict(ids)
        row.update(r)
        long_rows.append(row)
    wide_row = _flatten_disp({"stages": disp_stages}, ids, with_freq, hrv_sec)
    return long_rows, wide_row


def _done_recordings(path):
    done = set()
    if os.path.exists(path):
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                done.add((r.get("bids_folder", ""), r.get("session", "")))
    return done


def run(dataset_key, outdir, limit=None, resume=False, with_freq=True,
        shard=0, nshards=1, progress=None):
    ds = get_dataset(dataset_key)
    rows = ds.demographics()
    if nshards > 1:
        rows = [r for i, r in enumerate(rows) if i % nshards == shard]
    if limit:
        rows = rows[:limit]

    os.makedirs(outdir, exist_ok=True)
    suffix = dataset_key if nshards == 1 else f"{dataset_key}_s{shard}"
    long_path = os.path.join(outdir, f"hrv_windows_{suffix}.csv")
    wide_path = os.path.join(outdir, f"hrv_dispersion_{suffix}.csv")

    done = _done_recordings(long_path) if resume else set()
    mode = "a" if (resume and os.path.exists(long_path)) else "w"

    n_rec = n_skip = n_nofile = n_err = n_win = 0
    with open(long_path, mode, newline="") as lfh, open(wide_path, mode, newline="") as wfh:
        lw = csv.DictWriter(lfh, fieldnames=LONG_COLS, extrasaction="ignore")
        ww = csv.DictWriter(wfh, fieldnames=WIDE_COLS, extrasaction="ignore")
        if mode == "w":
            lw.writeheader()
            ww.writeheader()
        for i, rec in enumerate(rows):
            bids, sess = rec.get("BidsFolder", ""), rec.get("SessionID", "")
            if (bids, sess) in done:
                n_skip += 1
                continue
            try:
                long_rows, wide_row = _recording(ds, rec, with_freq=with_freq)
            except Exception as e:
                n_err += 1
                if progress:
                    progress(f"  ! {bids}: {type(e).__name__}: {e}")
                continue
            if long_rows is None:
                n_nofile += 1
                continue
            for r in long_rows:
                lw.writerow(r)
            ww.writerow(wide_row)
            n_rec += 1
            n_win += len(long_rows)
            if progress and (i + 1) % 10 == 0:
                lfh.flush(); wfh.flush()
                progress(f"  {dataset_key}: {i+1}/{len(rows)} scanned, "
                         f"{n_rec} recs -> {n_win} HRV windows, "
                         f"{n_nofile} no-file, {n_err} errors")
    summary = {"dataset": dataset_key, "long_table": long_path, "wide_table": wide_path,
               "n_recordings": n_rec, "n_skipped": n_skip, "n_no_file": n_nofile,
               "n_errors": n_err, "n_windows": n_win,
               "n_long_cols": len(LONG_COLS), "n_wide_cols": len(WIDE_COLS)}
    if progress:
        progress(f"DONE {summary}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="standard", choices=list(REGISTRY.keys()))
    ap.add_argument("--outdir", default="exports/hrv", help="output directory")
    ap.add_argument("--limit", type=int, default=None, help="cap recordings (debug)")
    ap.add_argument("--resume", action="store_true",
                    help="append, skipping recordings already in the long table")
    ap.add_argument("--no-freq", dest="with_freq", action="store_false",
                    help="skip hrv_frequency (lf/hf/... left blank); cheaper")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    summary = run(args.dataset, args.outdir, limit=args.limit, resume=args.resume,
                  with_freq=args.with_freq, shard=args.shard, nshards=args.nshards,
                  progress=lambda m: print(m, file=sys.stderr, flush=True))
    print(f"Wrote {summary['n_windows']} HRV windows from {summary['n_recordings']} "
          f"recordings -> {summary['long_table']} ({summary['n_long_cols']} cols) + "
          f"{summary['wide_table']} ({summary['n_wide_cols']} cols)")


if __name__ == "__main__":
    main()
