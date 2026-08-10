"""Export the clinical-report signal features (CSV) for model training.

Third exporter in the family:

  * `export_features.py`      — CAISR-only sleep macro-architecture + event indices
                                (reads no waveforms; fast whole-cohort).
  * `export_nk_features.py`   — per-stage NeuroKit HRV / EEG complexity / resp rate
                                (reads ECG + one EEG + one effort channel).
  * `export_report_features.py` (THIS) — the *signal-derived clinical markers* that
                                until now only existed in the on-demand webapp
                                report: per-stage **EEG spectral** band power /
                                ratios / REM-slowing, **sleep spindles**, **SpO2
                                oxygenation + hypoxic burden**, **respiratory-event**
                                durations / recovery, and an EOG **REM-density**
                                proxy.

It decodes only ONE central EEG, ONE SpO2, and ONE EOG channel from each physio EDF
(never the full montage) plus the small `resp_caisr` annotation stream, and it
deliberately does NOT recompute ECG-HRV / EEG-complexity / respiratory rate — those
per-stage features already live in `nk_features_standard.csv` and this CSV is meant
to be *merged* with it (and with `features_*.csv`) on `(bids_folder, session)`.
Skipping the whole-night R-peak detection is what makes it the cheaper of the two
waveform exporters (~2-4 s/recording vs ~7-8 s).

    # standard (local) cohort
    python3 export_report_features.py --dataset standard --out exports/report_features_standard.csv

    # large (S3) cohort — downloads each physio EDF to the size-capped cache
    python3 export_report_features.py --dataset large --out exports/report_features_large.csv --resume

Pure stdlib + numpy + edfio + scipy (no pandas; csv module only, so it runs under
any data-capable account). Resumable: --resume skips recordings already in --out.
Every feature degrades to an empty cell rather than raising, so one bad recording
never aborts the run. SpO2 scale differs by site (Emory stores a 0-1 fraction) —
`oxygenation._normalise_spo2` handles that, so the numbers are comparable across
sites.
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

# Physio roles decoded (one channel each). No ECG / effort -> no R-peak / resp-rate
# pass; those stay in export_nk_features.py's CSV.
_EEG_ROLES = {"eeg"}
_SPO2_ROLES = {"spo2"}
_EOG_ROLES = {"eog"}
_RESP_CAISR = "resp_caisr"

# ---- stable column layout (fixed order; missing values become empty cells) ----
STAGE_TAGS = ["wake", "n1", "n2", "n3", "rem"]     # eeg_spectral POOL_TAG values
BANDS = ["delta", "theta", "alpha", "sigma", "beta"]
SPINDLE_TAGS = ["n2", "n3"]
RESP_TYPES = ["obstructive_apnea", "central_apnea", "hypopnea", "RERA"]


def _spectral_cols():
    cols = []
    for t in STAGE_TAGS:
        cols.append(f"eeg_{t}_n_epochs")
        cols += [f"eeg_{t}_abs_{b}" for b in BANDS]
        cols += [f"eeg_{t}_rel_{b}" for b in BANDS]
        cols += [f"eeg_{t}_theta_alpha", f"eeg_{t}_delta_sigma", f"eeg_{t}_rem_slowing"]
    cols += ["eeg_ctr_rem_slowing", "eeg_ctr_n3_delta_rel",
             "eeg_ctr_n3_delta_abs", "eeg_ctr_rem_n3_slowing_ratio"]
    return cols


def _spindle_cols():
    cols = []
    for t in SPINDLE_TAGS:
        cols += [f"spindle_{t}_{f}" for f in
                 ("minutes", "n_spindles", "density_per_min", "amp_mean", "dur_mean")]
    cols.append("spindle_threshold_uv")
    return cols


OXY_COLS = ["spo2_mean", "spo2_min", "spo2_p1", "t90_pct", "t90_min", "odi",
            "n_desat", "desat_depth_mean", "desat_depth_max",
            "hypoxic_burden", "hb_pctmin_total", "spo2_duration_hours"]


def _resp_cols():
    cols = []
    for t in RESP_TYPES:
        cols += [f"n_{t}", f"{t}_dur_mean_s", f"{t}_dur_max_s"]
    cols += ["n_apnea_hypopnea", "ahi", "rdi",
             "event_dur_mean_s", "event_dur_median_s", "event_dur_max_s",
             "resp_tst_hours", "post_event_overshoot", "resp_recovery_time_s"]
    return cols


REMD_COLS = ["remd_rem_min", "remd_n_movements", "remd_rem_density_index",
             "remd_threshold_uv"]

SPECTRAL_COLS = _spectral_cols()
SPINDLE_COLS = _spindle_cols()
RESP_COLS = _resp_cols()

META_COLS = ["dataset", "bids_folder", "session", "site", "label",
             "eeg_channel", "spo2_channel", "eog_channel", "has_resp_caisr",
             "report_seconds"]
FEATURE_COLS = SPECTRAL_COLS + SPINDLE_COLS + OXY_COLS + RESP_COLS + REMD_COLS
ALL_COLS = META_COLS + FEATURE_COLS


# --------------------------------------------------------------------------- flatten
def _flatten_spectral(band, row):
    stages = (band or {}).get("stages") or {}
    for t in STAGE_TAGS:
        s = stages.get(t) or {}
        row[f"eeg_{t}_n_epochs"] = s.get("n_epochs")
        for b in BANDS:
            row[f"eeg_{t}_abs_{b}"] = s.get(f"abs_{b}")
            row[f"eeg_{t}_rel_{b}"] = s.get(f"rel_{b}")
        row[f"eeg_{t}_theta_alpha"] = s.get("theta_alpha")
        row[f"eeg_{t}_delta_sigma"] = s.get("delta_sigma")
        row[f"eeg_{t}_rem_slowing"] = s.get("rem_slowing")
    c = (band or {}).get("contrasts") or {}
    row["eeg_ctr_rem_slowing"] = c.get("rem_slowing")
    row["eeg_ctr_n3_delta_rel"] = c.get("n3_delta_rel")
    row["eeg_ctr_n3_delta_abs"] = c.get("n3_delta_abs")
    row["eeg_ctr_rem_n3_slowing_ratio"] = c.get("rem_n3_slowing_ratio")


def _flatten_spindles(spind, row):
    stages = (spind or {}).get("stages") or {}
    for t in SPINDLE_TAGS:
        s = stages.get(t) or {}
        for f in ("minutes", "n_spindles", "density_per_min", "amp_mean", "dur_mean"):
            row[f"spindle_{t}_{f}"] = s.get(f)
    row["spindle_threshold_uv"] = (spind or {}).get("threshold_uv")


def _flatten_oxy(oxy, row):
    oxy = oxy or {}
    for k in OXY_COLS:
        if k == "spo2_duration_hours":
            row[k] = oxy.get("duration_hours")
        else:
            row[k] = oxy.get(k)


def _flatten_resp(rev, row):
    rev = rev or {}
    for t in RESP_TYPES:
        row[f"n_{t}"] = rev.get(f"n_{t}")
        row[f"{t}_dur_mean_s"] = rev.get(f"{t}_dur_mean_s")
        row[f"{t}_dur_max_s"] = rev.get(f"{t}_dur_max_s")
    row["n_apnea_hypopnea"] = rev.get("n_apnea_hypopnea")
    row["ahi"] = rev.get("ahi")
    row["rdi"] = rev.get("rdi")
    row["event_dur_mean_s"] = rev.get("event_dur_mean_s")
    row["event_dur_median_s"] = rev.get("event_dur_median_s")
    row["event_dur_max_s"] = rev.get("event_dur_max_s")
    row["resp_tst_hours"] = rev.get("tst_hours_used")
    row["post_event_overshoot"] = rev.get("post_event_overshoot")
    row["resp_recovery_time_s"] = rev.get("resp_recovery_time_s")


def _flatten_remd(remd, row):
    remd = remd or {}
    row["remd_rem_min"] = remd.get("rem_min")
    row["remd_n_movements"] = remd.get("n_movements")
    row["remd_rem_density_index"] = remd.get("rem_density_index")
    row["remd_threshold_uv"] = remd.get("threshold_uv")


# --------------------------------------------------------------------------- compute
def _load_needed_channels(edf):
    """Decode only one EEG (central-preferred), one SpO2, one EOG channel.

    Returns (channels {label: samples}, fss {label: Hz}, roles, picked{eeg/spo2/eog}).
    """
    labels = [s.label.strip() for s in edf.signals]
    roles = {l: channel_role(l) for l in labels}
    eeg = nkf._pick_channel(labels, roles, _EEG_ROLES, prefer=_NK_EEG_PREFER)
    spo2 = nkf._pick_channel(labels, roles, _SPO2_ROLES)
    eog = nkf._pick_channel(labels, roles, _EOG_ROLES)
    keep = {c for c in (eeg, spo2, eog) if c is not None}
    channels, fss = {}, {}
    for s in edf.signals:
        lab = s.label.strip()
        if lab in keep and lab not in channels:
            channels[lab] = np.asarray(s.data, float)   # decodes THIS channel only
            fss[lab] = float(s.sampling_frequency)
    return channels, fss, roles, {"eeg": eeg, "spo2": spo2, "eog": eog}


def _feature_row(ds, rec):
    """Compute one report-feature row, or None if no physio EDF / no staging."""
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

    # small CAISR resp channel (for the respiratory-event features)
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
    channels, fss, roles, picked = _load_needed_channels(edf)
    eeg_ch, spo2_ch, eog_ch = picked["eeg"], picked["spo2"], picked["eog"]

    # tst_hours (pure staging, cheap) — used for AHI/RDI denominators
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
    _flatten_spectral(band, row)
    _flatten_spindles(spind, row)
    _flatten_oxy(oxy, row)
    _flatten_resp(rev, row)
    _flatten_remd(remd, row)
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
