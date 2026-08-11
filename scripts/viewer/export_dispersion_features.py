"""Export the WITHIN-STAGE DISPERSION features (the *non-avg* companion CSV).

The existing waveform CSVs (`nk_features_*.csv`, `report_features_*.csv`) give one
*mean* per stage for the per-epoch quantities (EEG band power / ratios, EEG
entropy, respiratory rate, spindle amplitude/duration). This exporter writes their
spread instead — SD, CV, and (for the flagship ratios/entropies/rates) p10/p50/p90
and IQR across the stage's epochs — one wide row per recording, keyed the same way
so it merges on `(bids_folder, session)` with the other three CSVs.

It decodes only ONE central EEG + ONE respiratory-effort channel from each physio
EDF (SpO2/EOG/ECG are not needed — those features have no epoch-averaged component
to expand). EEG complexity dispersion re-runs sample/permutation entropy per epoch,
so include/exclude it with --with-complexity (default on) since it dominates the
cost, exactly as in the NK export.

    # standard (local) cohort
    python3 export_dispersion_features.py --dataset standard \
        --out exports/dispersion_features_standard.csv

    # large (S3) cohort
    python3 export_dispersion_features.py --dataset large \
        --out exports/dispersion_features_large.csv --resume

Pure stdlib + numpy + edfio (+ scipy/neurokit via the reused modules). Resumable
(--resume skips recordings already in --out). Every feature degrades to an empty
cell rather than raising, so one bad recording never aborts the run.
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
import stage_dispersion as sd

_EEG_ROLES = {"eeg"}
_RSP_ROLES = {"effort", "airflow"}

STAGE_TAGS = sd.STAGE_TAGS                # wake n1 n2 n3 rem
SPINDLE_TAGS = sd.SPINDLE_TAGS            # n2 n3
BANDS = sd.BANDS


# ---- stable column layout (fixed order; missing values -> empty cells) -------
def _stat_cols(prefix, stats):
    return [f"{prefix}_{s}" for s in stats]


def _spectral_cols():
    cols = []
    for t in STAGE_TAGS:
        cols.append(f"eeg_{t}_n_epochs")
        for b in BANDS:                                   # compact: sd, cv
            cols += _stat_cols(f"eeg_{t}_rel_{b}", sd.STATS_COMPACT)
        for ratio in ("theta_alpha", "delta_sigma", "rem_slowing"):   # full stats
            cols += _stat_cols(f"eeg_{t}_{ratio}", sd.STATS_FULL)
    return cols


def _complexity_cols():
    cols = []
    for t in STAGE_TAGS:
        cols.append(f"eegc_{t}_n_epochs")
        for feat in ("sampen", "permen"):
            cols += _stat_cols(f"eegc_{t}_{feat}", sd.STATS_FULL)
    return cols


def _rsp_cols():
    cols = []
    for t in STAGE_TAGS:
        cols.append(f"rsp_{t}_n_epochs")
        cols += _stat_cols(f"rsp_{t}_rate", sd.STATS_FULL)
    return cols


def _spindle_cols():
    cols = []
    for t in SPINDLE_TAGS:
        cols.append(f"spindle_{t}_n_spindles")
        cols += _stat_cols(f"spindle_{t}_amp", sd.STATS_COMPACT)
        cols += _stat_cols(f"spindle_{t}_dur", sd.STATS_COMPACT)
    cols.append("spindle_threshold_uv")
    return cols


SPECTRAL_COLS = _spectral_cols()
COMPLEXITY_COLS = _complexity_cols()
RSP_COLS = _rsp_cols()
SPINDLE_COLS = _spindle_cols()

META_COLS = ["dataset", "bids_folder", "session", "site", "label",
             "eeg_channel", "rsp_channel", "with_complexity", "disp_seconds"]


def all_cols(with_complexity=True):
    feats = list(SPECTRAL_COLS)
    if with_complexity:
        feats += COMPLEXITY_COLS
    feats += RSP_COLS + SPINDLE_COLS
    return META_COLS + feats


# --------------------------------------------------------------------------- flatten
def _put_stats(row, prefix, disp, stats):
    """Write disp[stat] for stat in `stats` into row under prefix_<stat>."""
    disp = disp or {}
    for s in stats:
        row[f"{prefix}_{s}"] = disp.get(s)


def _flatten_spectral(res, row):
    stages = (res or {}).get("stages") or {}
    for t in STAGE_TAGS:
        s = stages.get(t) or {}
        row[f"eeg_{t}_n_epochs"] = s.get("n_epochs")
        for b in BANDS:
            _put_stats(row, f"eeg_{t}_rel_{b}", s.get(f"rel_{b}"), sd.STATS_COMPACT)
        for ratio in ("theta_alpha", "delta_sigma", "rem_slowing"):
            _put_stats(row, f"eeg_{t}_{ratio}", s.get(ratio), sd.STATS_FULL)


def _flatten_complexity(res, row):
    stages = (res or {}).get("stages") or {}
    for t in STAGE_TAGS:
        s = stages.get(t) or {}
        row[f"eegc_{t}_n_epochs"] = s.get("n_epochs")
        for feat in ("sampen", "permen"):
            _put_stats(row, f"eegc_{t}_{feat}", s.get(feat), sd.STATS_FULL)


def _flatten_rsp(res, row):
    stages = (res or {}).get("stages") or {}
    for t in STAGE_TAGS:
        s = stages.get(t) or {}
        row[f"rsp_{t}_n_epochs"] = s.get("n_epochs")
        _put_stats(row, f"rsp_{t}_rate", s, sd.STATS_FULL)


def _flatten_spindles(res, row):
    stages = (res or {}).get("stages") or {}
    for t in SPINDLE_TAGS:
        s = stages.get(t) or {}
        row[f"spindle_{t}_n_spindles"] = s.get("n_spindles")
        _put_stats(row, f"spindle_{t}_amp", s.get("amp"), sd.STATS_COMPACT)
        _put_stats(row, f"spindle_{t}_dur", s.get("dur"), sd.STATS_COMPACT)
    row["spindle_threshold_uv"] = (res or {}).get("threshold_uv")


# --------------------------------------------------------------------------- compute
def _load_needed_channels(edf):
    """Decode only one EEG (central-preferred) + one respiratory channel."""
    labels = [s.label.strip() for s in edf.signals]
    roles = {l: channel_role(l) for l in labels}
    eeg = nkf._pick_channel(labels, roles, _EEG_ROLES, prefer=_NK_EEG_PREFER)
    rsp = nkf._pick_channel(labels, roles, _RSP_ROLES)
    keep = {c for c in (eeg, rsp) if c is not None}
    channels, fss = {}, {}
    for s in edf.signals:
        lab = s.label.strip()
        if lab in keep and lab not in channels:
            channels[lab] = np.asarray(s.data, float)
            fss[lab] = float(s.sampling_frequency)
    return channels, fss, {"eeg": eeg, "rsp": rsp}


def _feature_row(ds, rec, with_complexity=True):
    bids = rec.get("BidsFolder", "")
    site, sess = rec.get("SiteID", ""), rec.get("SessionID", "1")

    codes, _ = _caisr_stage_codes(ds, rec, bids)
    if codes is None:
        return None
    try:
        f = ds.physio_path(site, bids, sess)
    except Exception:
        f = None
    if not f:
        return None

    edf = edfio.read_edf(f, lazy_load_data=True)
    channels, fss, picked = _load_needed_channels(edf)
    eeg_ch, rsp_ch = picked["eeg"], picked["rsp"]

    t0 = time.time()
    spectral = (sd.eeg_spectral_dispersion(channels[eeg_ch], fss.get(eeg_ch, 0), codes)
                if eeg_ch else {})
    spindle = (sd.spindle_dispersion(channels[eeg_ch], fss.get(eeg_ch, 0), codes)
               if eeg_ch else {})
    complexity = {}
    if with_complexity and eeg_ch:
        complexity = sd.eeg_complexity_dispersion(channels[eeg_ch], fss.get(eeg_ch, 0), codes)
    rsp = (sd.rsp_rate_dispersion(channels[rsp_ch], fss.get(rsp_ch, 0), codes)
           if rsp_ch else {})
    disp_sec = round(time.time() - t0, 2)

    row = {
        "dataset": ds.key, "bids_folder": bids, "session": sess, "site": site,
        "label": rec.get("Cognitive_Impairment", ""),
        "eeg_channel": eeg_ch or "", "rsp_channel": rsp_ch or "",
        "with_complexity": int(bool(with_complexity)), "disp_seconds": disp_sec,
    }
    _flatten_spectral(spectral, row)
    if with_complexity:
        _flatten_complexity(complexity, row)
    _flatten_rsp(rsp, row)
    _flatten_spindles(spindle, row)
    return row


def _load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                done.add((r.get("bids_folder", ""), r.get("session", "")))
    return done


def run(dataset_key, out_path, limit=None, resume=False, with_complexity=True,
        shard=0, nshards=1, progress=None):
    ds = get_dataset(dataset_key)
    rows = ds.demographics()
    if nshards > 1:
        rows = [r for i, r in enumerate(rows) if i % nshards == shard]
    if limit:
        rows = rows[:limit]
    done = _load_done(out_path) if resume else set()
    mode = "a" if (resume and os.path.exists(out_path)) else "w"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    cols = all_cols(with_complexity)

    n_written = n_skip = n_nofile = n_err = 0
    with open(out_path, mode, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        if mode == "w":
            w.writeheader()
        for i, rec in enumerate(rows):
            bids, sess = rec.get("BidsFolder", ""), rec.get("SessionID", "")
            if (bids, sess) in done:
                n_skip += 1
                continue
            try:
                row = _feature_row(ds, rec, with_complexity=with_complexity)
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
               "n_no_file": n_nofile, "n_errors": n_err, "n_columns": len(cols)}
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
    ap.add_argument("--no-complexity", dest="with_complexity", action="store_false",
                    help="skip EEG entropy dispersion (the expensive per-epoch pass)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    summary = run(args.dataset, args.out, limit=args.limit, resume=args.resume,
                  with_complexity=args.with_complexity,
                  shard=args.shard, nshards=args.nshards,
                  progress=lambda m: print(m, file=sys.stderr, flush=True))
    print(f"Wrote {summary['n_written']} rows x {summary['n_columns']} cols to {args.out}")


if __name__ == "__main__":
    main()
