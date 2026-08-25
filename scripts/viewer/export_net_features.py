"""Export multi-channel EEG network features (CSV): B1 connectivity + B2 microstates.

Batch-4 of the advanced-feature roadmap (ADVANCED_FEATURES_COVERAGE.md) — the
"heavy" multi-channel-EEG items, gated: these fold into feature_matrix_local_plus
only if the combined set beats the LOSO baseline. Unlike the other exporters this
decodes MULTIPLE EEG channels (the six standard scalp derivations F3,F4,C3,C4,O1,O2
present at all three sites), then runs:

  * **B1 connectivity** (`eeg_network.connectivity_features`) — coherence + wPLI
    per band, pooled NREM and REM.
  * **B2 microstates** (`eeg_network.microstate_features`) — permutation-invariant
    microstate dynamics over NREM.

Both re-reference to the common average of the decoded scalp channels, so the
montage-heterogeneous cohort is comparable across sites. Resumable (--resume) and
shardable (--shard/--nshards) for the large cohort; the multi-channel decode +
cross-spectra make this the slowest exporter, so shard it. Merge with the other
WIDE CSVs on (bids_folder, session).

    python3 export_net_features.py --dataset standard --out exports/net_features_standard.csv
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
import eeg_network as net

_MASTOID_CUES = ("m1", "m2")

META_COLS = ["dataset", "bids_folder", "session", "site", "label",
             "eeg_channels", "n_eeg", "eeg_fs", "net_seconds"]
NET_COLS = net.net_columns()
ALL_COLS = META_COLS + NET_COLS


def _pick_eeg_channels(edf):
    """Decode the standard scalp EEG derivations (F3,F4,C3,C4,O1,O2 etc.) from a
    lazily-opened EDF. Excludes pure mastoids (M1/M2) and non-EEG channels; decodes
    ONLY the selected channels, never the full montage. Returns (sigs, fss, labels)."""
    labels_all = [s.label.strip() for s in edf.signals]
    roles = {l: channel_role(l) for l in labels_all}
    keep = []
    for lab in labels_all:
        if roles.get(lab) != "eeg":
            continue
        low = lab.lower()
        if not net._is_scalp(lab):                 # skip generic 'eeg'/mastoid-only
            continue
        # a channel labelled purely as a mastoid (e.g. "M1") is a reference, not a site
        toks = low.replace("-", " ").split()
        if any(t in _MASTOID_CUES for t in toks) and not any(
                c in low for c in ("f3", "f4", "c3", "c4", "o1", "o2")):
            continue
        if lab not in keep:
            keep.append(lab)
    sigs, fss, used = [], [], []
    seen = set()
    for s in edf.signals:
        lab = s.label.strip()
        if lab in keep and lab not in seen:
            sigs.append(np.asarray(s.data, float))
            fss.append(float(s.sampling_frequency))
            used.append(lab)
            seen.add(lab)
    return sigs, fss, used


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
    sigs, fss, used = _pick_eeg_channels(edf)
    if len(sigs) < 2:
        return None

    t0 = time.time()
    row = {
        "dataset": ds.key, "bids_folder": bids, "session": sess,
        "site": site, "label": rec.get("Cognitive_Impairment", ""),
        "eeg_channels": "|".join(used), "n_eeg": len(used),
        "eeg_fs": round(float(fss[0]), 4) if fss else "",
    }
    net.flatten_connectivity(
        net.connectivity_features(sigs, fss, used, codes), row)
    net.flatten_microstate(
        net.microstate_features(sigs, fss, codes), row)
    row["net_seconds"] = round(time.time() - t0, 2)
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
            if progress and (i + 1) % 5 == 0:
                fh.flush()
                progress(f"  {dataset_key}[{shard}/{nshards}]: {i+1}/{len(rows)} scanned, "
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
