#!/usr/bin/env python3
"""Export directed cross-system coupling features (CSV), one row per recording.

A compact port of `causal_networks_physiology`'s Granger-G core to the Challenge-2026
pipeline (see scripts/featsel/README.md §2 for the scouting rationale). For each
recording we reconstruct three 1-Hz signals (breath rate, heart rate, EEG-alpha
envelope — `signals.build_signals`) and compute directed Granger G among them per sleep
stage (`coupling.coupling_features`), yielding a small block of interpretable coupling
scalars keyed by `bids_folder`. Feed the CSV straight to `coup_gate_ab.py` (LOSO fusion
gate) — nothing is folded into the champion unless it clears that gate.

Modeled on `export_net_features.py`: header-first channel picking (decode ONLY the ECG,
effort/airflow and central-EEG channels, never the full montage), CAISR staging via
`_caisr_stage_codes`, resumable (`--resume`), shardable (`--shard/--nshards`).

Run on pdmle as arshia_ilaty_physio26 (data-capable; NOT sudo), from the deployed
bundle so `sources`/`app`/`nk_features` import as top-level modules:

    cd /data-temp/physio-viewer
    python3 export_coupling_features.py --dataset standard \
        --out exports/coupling_features_standard.csv
"""
import os
import csv
import sys
import argparse

import numpy as np
import edfio

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
VIEWER = os.environ.get("VIEWER_DIR", os.path.abspath(os.path.join(HERE, "..", "viewer")))
if os.path.isdir(VIEWER):
    sys.path.insert(0, VIEWER)

from sources import get_dataset, REGISTRY          # noqa: E402
from app import channel_role, _caisr_stage_codes    # noqa: E402
from nk_features import _pick_channel               # noqa: E402
import coupling                                      # noqa: E402
from signals import build_signals                    # noqa: E402

# stage codes: 1=N3, 2=N2, 3=N1, 4=REM, 5=Wake, 9=Unknown
STAGE_GROUPS = {"nrem": {1, 2, 3}, "rem": {4}}
TIMESCALES = (1,)

META_COLS = ["dataset", "bids_folder", "session", "site", "label",
             "ecg_ch", "rsp_ch", "eeg_ch", "coup_seconds"]
FEAT_COLS = coupling.all_feature_names(STAGE_GROUPS, TIMESCALES)
ALL_COLS = META_COLS + FEAT_COLS


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
    labels = [s.label.strip() for s in edf.signals]
    roles = {l: channel_role(l) for l in labels}
    ecg_lab = _pick_channel(labels, roles, {"ecg"})
    rsp_lab = _pick_channel(labels, roles, {"effort", "airflow"})
    eeg_lab = _pick_channel(labels, roles, {"eeg"}, prefer=["c3", "c4", "o1", "o2"])
    need = {x for x in (ecg_lab, rsp_lab, eeg_lab) if x}
    if len(need) < 3:
        return None

    channels, fss = {}, {}
    for s in edf.signals:
        lab = s.label.strip()
        if lab in need and lab not in channels:
            channels[lab] = np.asarray(s.data, float)
            fss[lab] = float(s.sampling_frequency)
    roles2 = {lab: roles[lab] for lab in channels}

    sig = build_signals(channels, fss, roles2, codes, _pick_channel)
    if sig is None:
        return None
    feats = coupling.coupling_features(
        sig["breath"], sig["heart"], sig["eeg"], sig["stage"],
        STAGE_GROUPS, timescales=TIMESCALES)

    row = {
        "dataset": ds.key, "bids_folder": bids, "session": sess, "site": site,
        "label": rec.get("Cognitive_Impairment", ""),
        "ecg_ch": sig["ecg_ch"], "rsp_ch": sig["rsp_ch"], "eeg_ch": sig["eeg_ch"],
        "coup_seconds": sig["seconds"],
    }
    row.update({k: round(v, 6) for k, v in feats.items()})
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
