"""Export Group-3 dynamics/distribution features (CSV): D2 desat clustering,
D1 breath-interval distribution, E1 limb periodicity.

Batch-3 of the advanced-feature roadmap (ADVANCED_FEATURES_COVERAGE.md), gated:
these fold into feature_matrix_local_plus only if they beat the LOSO baseline.
Reuses streams we ALREADY decode -- one SpO2 + one effort/airflow channel from the
lazily-opened physio EDF, and the small `limb_caisr` CAISR stream -- so no new
signal decoding. Computes:

  * D2 desat clustering (`dyn_features.desat_cluster_features`) -- SpO2.
  * D1 breath-interval distribution (`dyn_features.breath_interval_features`) --
    effort/airflow, needs the CAISR stage codes to restrict to sleep epochs.
  * E1 limb periodicity (`dyn_features.limb_periodicity_features`) -- limb_caisr.

Same resumable/shardable structure as export_micro_features. Merge with the other
WIDE CSVs on (bids_folder, session).

    python3 export_dyn_features.py --dataset standard --out exports/dyn_features_standard.csv
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
import dyn_features as dyn

_SPO2_ROLES = {"spo2"}
_RSP_ROLES = {"effort", "airflow"}
_LIMB_CAISR = "limb_caisr"

META_COLS = ["dataset", "bids_folder", "session", "site", "label",
             "spo2_channel", "rsp_channel", "limb_fs", "dyn_seconds"]
DYN_COLS = dyn.dyn_columns()
ALL_COLS = META_COLS + DYN_COLS


def _load_needed_channels(edf):
    """Decode one SpO2 + one effort/airflow channel from a lazily-opened EDF (only
    these two -- never the full montage)."""
    labels = [s.label.strip() for s in edf.signals]
    roles = {l: channel_role(l) for l in labels}
    spo2 = nkf._pick_channel(labels, roles, _SPO2_ROLES)
    rsp = nkf._pick_channel(labels, roles, _RSP_ROLES)
    keep = {c for c in (spo2, rsp) if c is not None}
    channels, fss = {}, {}
    for s in edf.signals:
        lab = s.label.strip()
        if lab in keep and lab not in channels:
            channels[lab] = np.asarray(s.data, float)      # decodes THIS channel only
            fss[lab] = float(s.sampling_frequency)
    return channels, fss, spo2, rsp


def _limb_stream(ds, site, bids, sess):
    """(limb_codes, limb_fs) from the small CAISR EDF, or (None, None)."""
    edf = ds.open_caisr(site, bids, sess)
    if edf is None:
        return None, None
    for s in edf.signals:
        if s.label.strip() == _LIMB_CAISR:
            return np.asarray(s.data, float), float(s.sampling_frequency)
    return None, None


def _feature_row(ds, rec):
    bids = rec.get("BidsFolder", "")
    site, sess = rec.get("SiteID", ""), rec.get("SessionID", "1")
    codes, reason = _caisr_stage_codes(ds, rec, bids)
    # codes only needed for D1 (breath by sleep epoch); D2/E1 work without them.
    try:
        f = ds.physio_path(site, bids, sess)
    except Exception:
        f = None
    limb, limb_fs = _limb_stream(ds, site, bids, sess)

    channels, fss, spo2_ch, rsp_ch = {}, {}, None, None
    if f:
        edf = edfio.read_edf(f, lazy_load_data=True)
        channels, fss, spo2_ch, rsp_ch = _load_needed_channels(edf)

    # nothing usable at all -> skip the record entirely
    if spo2_ch is None and rsp_ch is None and limb is None:
        return None

    t0 = time.time()
    row = {
        "dataset": ds.key, "bids_folder": bids, "session": sess,
        "site": site, "label": rec.get("Cognitive_Impairment", ""),
        "spo2_channel": spo2_ch or "", "rsp_channel": rsp_ch or "",
        "limb_fs": round(limb_fs, 4) if limb_fs else "",
    }

    if spo2_ch is not None:
        dyn.flatten_desat_cluster(
            dyn.desat_cluster_features(channels[spo2_ch], fss[spo2_ch]), row)
    else:
        dyn.flatten_desat_cluster({"ok": False}, row)

    if rsp_ch is not None and codes is not None:
        dyn.flatten_breath_interval(
            dyn.breath_interval_features(channels[rsp_ch], fss[rsp_ch], codes), row)
    else:
        dyn.flatten_breath_interval({"ok": False}, row)

    dyn.flatten_limb_periodicity(
        dyn.limb_periodicity_features(limb, limb_fs), row)

    row["dyn_seconds"] = round(time.time() - t0, 2)
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
