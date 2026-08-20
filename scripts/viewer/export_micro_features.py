"""Export sleep-microstructure features (CSV): SO-spindle coupling, RSWA, CAP.

Batch-2 of the advanced-feature roadmap (ADVANCED_FEATURES_COVERAGE.md). Reads the
big physiological EDF once per recording and decodes only the two waveform channels
these features need — one central EEG and one chin (submental) EMG — plus the small
CAISR arousal stream, then runs:

  * **B3 SO-spindle coupling** (`eeg_coupling.so_spindle_coupling`) — how tightly
    spindles phase-lock to the slow-oscillation up-state (a memory-consolidation /
    cognitive-decline EEG biomarker). Central EEG.
  * **E2 RSWA** (`emg_atonia.rswa_features`) — REM-sleep-without-atonia: chin-EMG
    tone/phasic activity in REM, normalised to the recording's own atonia floor (a
    prodromal α-synuclein-neurodegeneration marker). Chin EMG.
  * **A2 CAP** (`cap_events.cap_features`) — cyclic alternating pattern rate +
    A1/A2/A3 subtypes: NREM instability. Central EEG (+ arousal stream for A3).

Like export_nk_features / export_report_features, decodes ONLY the needed channels
(never the full montage) from a lazily-opened EDF. Resumable (--resume) and
shardable (--shard/--nshards) for the large cohort, matching export_combined_features.
Merge with the other WIDE CSVs on (bids_folder, session).

    python3 export_micro_features.py --dataset standard --out exports/micro_features_standard.csv
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
import eeg_coupling as ecpl
import emg_atonia as ema
import cap_events as cap

_EEG_PREFER = ["c3", "c4", "o1", "o2"]


META_COLS = ["dataset", "bids_folder", "session", "site", "label",
             "eeg_channel", "emg_channel", "arousal_fs", "micro_seconds"]
MICRO_COLS = ecpl.coupling_columns() + ema.rswa_columns() + cap.cap_columns()
ALL_COLS = META_COLS + MICRO_COLS


def _load_needed_channels(edf):
    """Decode one central EEG + one chin EMG from a lazily-opened EDF (only these)."""
    labels = [s.label.strip() for s in edf.signals]
    roles = {l: channel_role(l) for l in labels}
    eeg = nkf._pick_channel(labels, roles, {"eeg"}, prefer=_EEG_PREFER)
    emg = nkf._pick_channel(labels, roles, {"chin_emg"})
    keep = {c for c in (eeg, emg) if c is not None}
    channels, fss = {}, {}
    for s in edf.signals:
        lab = s.label.strip()
        if lab in keep and lab not in channels:
            channels[lab] = np.asarray(s.data, float)      # decodes THIS channel only
            fss[lab] = float(s.sampling_frequency)
    return channels, fss, eeg, emg


def _arousal_stream(ds, site, bids, sess):
    """(arousal_codes, arousal_fs) from the small CAISR EDF, or (None, None)."""
    edf = ds.open_caisr(site, bids, sess)
    if edf is None:
        return None, None
    for s in edf.signals:
        if s.label.strip() == "arousal_caisr":
            return np.asarray(s.data, float), float(s.sampling_frequency)
    return None, None


def _feature_row(ds, rec):
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
    channels, fss, eeg_ch, emg_ch = _load_needed_channels(edf)
    if eeg_ch is None and emg_ch is None:
        return None
    ar, ar_fs = _arousal_stream(ds, site, bids, sess)

    t0 = time.time()
    row = {
        "dataset": ds.key, "bids_folder": bids, "session": sess,
        "site": site, "label": rec.get("Cognitive_Impairment", ""),
        "eeg_channel": eeg_ch or "", "emg_channel": emg_ch or "",
        "arousal_fs": round(ar_fs, 4) if ar_fs else "",
    }
    if eeg_ch is not None:
        eeg, efs = channels[eeg_ch], fss[eeg_ch]
        ecpl.flatten_coupling(ecpl.so_spindle_coupling(eeg, efs, codes), row)
        cap.flatten_cap(cap.cap_features(eeg, efs, codes, arousal_codes=ar, arousal_fs=ar_fs), row)
    else:
        ecpl.flatten_coupling({"ok": False}, row)
        cap.flatten_cap({"ok": False}, row)
    if emg_ch is not None:
        ema.flatten_rswa(ema.rswa_features(channels[emg_ch], fss[emg_ch], codes), row)
    else:
        ema.flatten_rswa({"ok": False}, row)
    row["micro_seconds"] = round(time.time() - t0, 2)
    return row


def _load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                done.add((r.get("bids_folder", ""), r.get("session", "")))
    return done


def run(dataset_key, out_path, limit=None, resume=False, progress=None, shard=0, nshards=1):
    ds = get_dataset(dataset_key)
    rows = ds.demographics()
    if limit:
        rows = rows[:limit]
    if nshards > 1:
        rows = [r for i, r in enumerate(rows) if i % nshards == shard]
    done = _load_done(out_path) if resume else set()
    mode = "a" if (resume and os.path.exists(out_path)) else "w"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    n_written = n_skip = n_nofile = n_err = 0
    with open(out_path, mode, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ALL_COLS, extrasaction="ignore")
        if mode == "w":
            w.writeheader()
        for i, rec in enumerate(rows):
            bids = rec.get("BidsFolder", "")
            # match the SessionID default used when the row is written (_feature_row
            # writes rec.get('SessionID','1')); using '' here would let a record with
            # no SessionID slip past --resume dedup and duplicate its row.
            sess = rec.get("SessionID", "")
            if (bids, sess) in done or (bids, sess or "1") in done:
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
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    summary = run(args.dataset, args.out, limit=args.limit, resume=args.resume,
                  shard=args.shard, nshards=args.nshards,
                  progress=lambda m: print(m, file=sys.stderr, flush=True))
    print(",".join(ALL_COLS))
    print(f"Wrote {summary['n_written']} rows x {summary['n_columns']} cols to {args.out}")


if __name__ == "__main__":
    main()
