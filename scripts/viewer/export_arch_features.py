"""Export sleep-architecture transition + microarousal features (CSV) for training.

Companion to `export_features.py`, and — like it — reads ONLY the small CAISR
annotation EDFs (a few hundred KB each), never the 170-460 MB physiological EDFs,
so the whole cohort completes in minutes. Per recording it writes one wide row of:

  * **F1 — stage-transition Markov model** (arch_dynamics.transition_features):
    the full 5x5 row-normalised transition-probability matrix + per-row conditional
    entropies + occupancy-weighted entropy rate and stability (self-transition)
    index. Our EDA says the discriminative axis is architecture *instability*, so
    this exposes it directly.
  * **A1 — microarousal distributions** (arch_dynamics.arousal_features): arousal
    event-duration and inter-arousal-interval distributions + per-stage arousal
    index, from the arousal_caisr stream. The stream's sampling rate is read from
    the EDF (not assumed), so durations/onsets are in real seconds.

Stage codes are used RAW (single-epoch spikes are genuine fragmentation), matching
export_features.py / dynamics.py so the counting is consistent with the rest of the
pipeline.

    # standard (local) cohort
    python3 export_arch_features.py --dataset standard --out exports/arch_features_standard.csv

    # large (S3) cohort
    python3 export_arch_features.py --dataset large --out exports/arch_features_large.csv --resume

Pure stdlib + numpy + edfio (no pandas — csv module only — so it runs under any
account). Resumable: --resume skips recordings already in --out. Merge with
features_*.csv / nk_features_*.csv on (bids_folder, session).
"""
import os
import csv
import sys
import argparse

import numpy as np

from sources import get_dataset, REGISTRY
import arch_dynamics as ad

META_COLS = ["dataset", "bids_folder", "session", "site", "label",
             "arousal_fs", "n_epochs_scored"]
ARCH_COLS = list(ad.ARCH_FEATURE_COLUMNS)
ALL_COLS = META_COLS + ARCH_COLS


def _feature_row(ds, rec):
    """Compute one architecture/arousal feature row; None if no CAISR file."""
    bids = rec.get("BidsFolder", "")
    site, sess = rec.get("SiteID", ""), rec.get("SessionID", "1")
    edf = ds.open_caisr(site, bids, sess)
    if edf is None:
        return None
    chans, fss = {}, {}
    for s in edf.signals:
        lab = s.label.strip()
        if lab not in chans:
            chans[lab] = np.asarray(s.data, float)
            fss[lab] = float(s.sampling_frequency)

    stage = chans.get("stage_caisr")
    arousal = chans.get("arousal_caisr")
    arousal_fs = fss.get("arousal_caisr")

    row = {
        "dataset": ds.key,
        "bids_folder": bids,
        "session": rec.get("SessionID", ""),
        "site": site,
        "label": rec.get("Cognitive_Impairment", ""),
        "arousal_fs": round(arousal_fs, 4) if arousal_fs else "",
        "n_epochs_scored": (int(np.count_nonzero(np.rint(stage).astype(int) != 9))
                            if stage is not None and len(stage) else ""),
    }
    row.update(ad.flatten_features(stage, arousal, arousal_fs))
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
    print(",".join(ALL_COLS))
    print(f"Wrote {summary['n_written']} rows x {summary['n_columns']} cols to {args.out}")


if __name__ == "__main__":
    main()
