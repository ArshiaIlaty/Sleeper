"""Export BOTH the per-stage NeuroKit CSV and the clinical-report CSV in one pass.

`export_nk_features.py` and `export_report_features.py` each read the big
physiological EDFs. Run separately over the large (S3) cohort they would download
every 170-460 MB file *twice* (~4 TB egress, 2x wall-clock), because the on-disk
cache is size-capped and evicts between passes. This driver opens each EDF **once**,
decodes the *union* of channels both need (ECG + central EEG + effort/airflow +
SpO2 + EOG) and the small `resp_caisr` stream, computes both feature sets, and
appends a row to each output CSV.

The two outputs are byte-for-byte schema-compatible with the standalone exporters
(same column lists, same flatten logic — imported, not re-implemented), so the
large-cohort CSVs merge with the standard-cohort ones and with `features_*.csv` on
`(bids_folder, session)`.

    # large (S3) cohort — the reason this driver exists
    python3 export_combined_features.py --dataset large \
        --nk-out exports/nk_features_large.csv \
        --report-out exports/report_features_large.csv --resume

    # standard (local) cohort works too (no double-download to save there, but handy)
    python3 export_combined_features.py --dataset standard \
        --nk-out exports/nk_features_standard.csv \
        --report-out exports/report_features_standard.csv --resume

Resumable per-file: with --resume, each output independently skips recordings it
already contains (never duplicates a row), and a recording is only skipped from
downloading entirely when BOTH outputs already have it. Every feature degrades to
an empty cell rather than raising, so one bad recording never aborts the run. Pure
stdlib + numpy + edfio + scipy + neurokit2 (csv module only; no pandas).
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
import eeg_spectral
import oxygenation
import resp_events
import clinical_report

# reuse the exact schemas + flatten helpers from the two standalone exporters
import export_nk_features as xnk
import export_report_features as xrep

# union of physio roles both exporters need (one channel each)
_ECG_ROLES = {"ecg"}
_EEG_ROLES = {"eeg"}
_RSP_ROLES = {"effort", "airflow"}
_SPO2_ROLES = {"spo2"}
_EOG_ROLES = {"eog"}
_RESP_CAISR = "resp_caisr"


def _load_union_channels(edf):
    """Decode only one channel per needed role (ECG / central EEG / effort /
    SpO2 / EOG) from a lazily-opened EDF. Returns (channels, fss, roles, picked)."""
    labels = [s.label.strip() for s in edf.signals]
    roles = {l: channel_role(l) for l in labels}
    picked = {
        "ecg": nkf._pick_channel(labels, roles, _ECG_ROLES),
        "eeg": nkf._pick_channel(labels, roles, _EEG_ROLES, prefer=_NK_EEG_PREFER),
        "rsp": nkf._pick_channel(labels, roles, _RSP_ROLES),
        "spo2": nkf._pick_channel(labels, roles, _SPO2_ROLES),
        "eog": nkf._pick_channel(labels, roles, _EOG_ROLES),
    }
    keep = {c for c in picked.values() if c is not None}
    channels, fss = {}, {}
    for s in edf.signals:
        lab = s.label.strip()
        if lab in keep and lab not in channels:
            channels[lab] = np.asarray(s.data, float)   # decodes THIS channel only
            fss[lab] = float(s.sampling_frequency)
    return channels, fss, roles, picked


def _nk_row(ds, rec, channels, fss, roles, codes):
    """Build the NeuroKit feature row (schema == export_nk_features.ALL_COLS)."""
    bids = rec.get("BidsFolder", "")
    site, sess = rec.get("SiteID", ""), rec.get("SessionID", "1")
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


def _report_row(ds, rec, channels, fss, roles, picked, resp_sig, codes):
    """Build the clinical-report feature row (schema == export_report_features.ALL_COLS)."""
    bids = rec.get("BidsFolder", "")
    site, sess = rec.get("SiteID", ""), rec.get("SessionID", "1")
    eeg_ch, spo2_ch, eog_ch = picked["eeg"], picked["spo2"], picked["eog"]
    _, tst_hours = clinical_report.sleep_quality(codes)

    t0 = time.time()
    band = (eeg_spectral.eeg_bandpower_by_stage(channels[eeg_ch], fss.get(eeg_ch, 0), codes)
            if eeg_ch else {})
    spind = (eeg_spectral.spindle_features_by_stage(channels[eeg_ch], fss.get(eeg_ch, 0), codes)
             if eeg_ch else {})
    oxy, spo2_norm = {}, None
    if spo2_ch:
        oxy = oxygenation.spo2_features(channels[spo2_ch], fss.get(spo2_ch, 0))
        spo2_norm, _, _ = oxygenation._normalise_spo2(channels[spo2_ch], fss.get(spo2_ch, 0))
    rev = {}
    if resp_sig is not None:
        rev = resp_events.resp_event_features(
            resp_sig, tst_hours=tst_hours,
            spo2=spo2_norm, spo2_fs=(fss.get(spo2_ch, 0) if spo2_ch else None))
    remd = (clinical_report.rem_eye_movement_index(channels[eog_ch], fss.get(eog_ch, 0), codes)
            if eog_ch else {})
    report_sec = round(time.time() - t0, 2)

    row = {
        "dataset": ds.key, "bids_folder": bids, "session": sess, "site": site,
        "label": rec.get("Cognitive_Impairment", ""),
        "eeg_channel": eeg_ch or "", "spo2_channel": spo2_ch or "",
        "eog_channel": eog_ch or "", "has_resp_caisr": int(resp_sig is not None),
        "report_seconds": report_sec,
    }
    xrep._flatten_spectral(band, row)
    xrep._flatten_spindles(spind, row)
    xrep._flatten_oxy(oxy, row)
    xrep._flatten_resp(rev, row)
    xrep._flatten_remd(remd, row)
    return row


def _open_writer(path, cols, resume):
    """Open a CSV for append-or-create and return (fh, DictWriter, done_set)."""
    done = xnk._load_done(path) if resume else set()
    mode = "a" if (resume and os.path.exists(path)) else "w"
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fh = open(path, mode, newline="")
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    if mode == "w":
        w.writeheader()
    return fh, w, done


def run(dataset_key, nk_out, report_out, limit=None, resume=False, progress=None,
        shard=0, nshards=1):
    ds = get_dataset(dataset_key)
    rows = ds.demographics()
    if limit:
        rows = rows[:limit]
    if nshards > 1:
        # stride-slice so each worker gets an interleaved (load-balanced) share;
        # ordering within a shard is preserved for stable --resume.
        rows = [r for i, r in enumerate(rows) if i % nshards == shard]

    nk_fh, nk_w, nk_done = _open_writer(nk_out, xnk.ALL_COLS, resume)
    rep_fh, rep_w, rep_done = _open_writer(report_out, xrep.ALL_COLS, resume)

    n_nk = n_rep = n_skip = n_nofile = n_err = 0
    try:
        for i, rec in enumerate(rows):
            bids, sess = rec.get("BidsFolder", ""), rec.get("SessionID", "")
            key = (bids, sess)
            need_nk = key not in nk_done
            need_rep = key not in rep_done
            if not need_nk and not need_rep:      # already in BOTH -> no download
                n_skip += 1
                continue
            try:
                codes, _ = _caisr_stage_codes(ds, rec, bids)
                if codes is None:
                    n_nofile += 1
                    continue
                site = rec.get("SiteID", "")
                try:
                    f = ds.physio_path(site, bids, sess)
                except Exception:
                    f = None
                if not f:
                    n_nofile += 1
                    continue

                resp_sig = None
                try:
                    cedf = ds.open_caisr(site, bids, sess)
                    if cedf is not None:
                        for s in cedf.signals:
                            if s.label.strip() == _RESP_CAISR:
                                resp_sig = np.asarray(s.data, float)
                                break
                except Exception:
                    resp_sig = None

                edf = edfio.read_edf(f, lazy_load_data=True)
                channels, fss, roles, picked = _load_union_channels(edf)

                if need_nk:
                    nk_w.writerow(_nk_row(ds, rec, channels, fss, roles, codes))
                    n_nk += 1
                if need_rep:
                    rep_w.writerow(_report_row(ds, rec, channels, fss, roles,
                                               picked, resp_sig, codes))
                    n_rep += 1
            except Exception as e:
                n_err += 1
                if progress:
                    progress(f"  ! {bids}: {type(e).__name__}: {e}")
                continue
            if progress and (i + 1) % 10 == 0:
                nk_fh.flush(); rep_fh.flush()
                progress(f"  {dataset_key}: {i+1}/{len(rows)} scanned, "
                         f"nk+{n_nk} report+{n_rep}, {n_skip} skipped, "
                         f"{n_nofile} no-file/staging, {n_err} errors")
    finally:
        nk_fh.close(); rep_fh.close()

    summary = {"dataset": dataset_key, "nk_out": nk_out, "report_out": report_out,
               "n_total": len(rows), "n_nk_written": n_nk, "n_report_written": n_rep,
               "n_skipped": n_skip, "n_no_file": n_nofile, "n_errors": n_err}
    if progress:
        progress(f"DONE {summary}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="large", choices=list(REGISTRY.keys()))
    ap.add_argument("--nk-out", required=True, help="NeuroKit per-stage CSV path")
    ap.add_argument("--report-out", required=True, help="clinical-report CSV path")
    ap.add_argument("--limit", type=int, default=None, help="cap recordings (debug)")
    ap.add_argument("--resume", action="store_true",
                    help="append, skipping recordings already present per-file")
    ap.add_argument("--shard", type=int, default=0,
                    help="this worker's index in [0, nshards) for parallel runs")
    ap.add_argument("--nshards", type=int, default=1,
                    help="total number of parallel workers (stride-slices the cohort)")
    args = ap.parse_args()
    summary = run(args.dataset, args.nk_out, args.report_out, limit=args.limit,
                  resume=args.resume, shard=args.shard, nshards=args.nshards,
                  progress=lambda m: print(m, file=sys.stderr, flush=True))
    print(f"nk={summary['n_nk_written']} report={summary['n_report_written']} rows; "
          f"skipped={summary['n_skipped']} no-file={summary['n_no_file']} "
          f"errors={summary['n_errors']}")


if __name__ == "__main__":
    main()
