"""In-container signal-feature package (Track 2 full-feature submission).

Vendors the viewer's validated feature modules (NeuroKit2 per-stage HRV,
EEG spectral/spindles, oxygenation, respiratory events, SO-spindle coupling,
REM-atonia/RSWA, CAP) so they run at INFERENCE inside the frozen Challenge
container, instead of the offline CSV export pipeline they were built for.

The compute modules (nk_features / eeg_spectral / oxygenation / resp_events /
clinical_report / eeg_coupling / emg_atonia / cap_events) are copied verbatim
from scripts/viewer/ with their cross-imports made package-relative. This file
adds the two small helpers those modules' EXPORTERS relied on but that live in
app.py / preprocess.py (channel_role, smooth_stages) plus the exact per-record
recipe + fixed column layout, so the values match the validated *_features_*.csv
byte-for-byte (verified by scripts/verify_sig_parity.py).

`signal_feature_row(phys_data, phys_fs, algo_data, algo_fs)` returns
(values, column_names) as a FIXED-ORDER list; any missing stream / failed value
is NaN. It never raises — a bad recording degrades to an all-NaN block.
"""
from __future__ import annotations

import re

import numpy as np

from . import nk_features as nkf
from . import eeg_spectral
from . import oxygenation
from . import resp_events
from . import clinical_report
from . import eeg_coupling as ecpl
from . import emg_atonia as ema
from . import cap_events as cap

# --------------------------------------------------------------------------- #
# Vendored from scripts/viewer/app.py — coarse channel-role inference. The only
# app.py dependency the exporters used; kept byte-identical so channel PICKING
# matches the offline extraction exactly.
# --------------------------------------------------------------------------- #
_ROLE_CUES = [
    ("ecg", ["ekg", "ecg"]),
    ("spo2", ["spo2", "sao2"]),
    ("chin_emg", ["chin"]),
    ("limb_emg", ["leg", "lat", "rat", "lleg", "rleg", "plm"]),
    ("eog", ["e1", "e2", "eog", "loc", "roc"]),
    ("airflow", ["flow", "therm", "nasal", "ptaf", "cpap", "press", "npt", "cpres"]),
    ("effort", ["chest", "abd", "thor", "thorac", "abdomen"]),
    ("eeg", ["f3", "f4", "c3", "c4", "o1", "o2", "m1", "m2", "eeg", "fp", "cz", "pz"]),
]


def channel_role(label):
    """Infer a coarse channel role from its label (app.channel_role, verbatim)."""
    lab = label.lower()
    toks = set(re.split(r"[^a-z0-9]+", lab))
    for role, cues in _ROLE_CUES:
        for cue in cues:
            if cue in toks or (len(cue) > 2 and cue in lab):
                return role
    return "other"


# --------------------------------------------------------------------------- #
# Vendored from scripts/viewer/preprocess.py — the stage-smoothing the CAISR
# stage codes went through in the offline export (_caisr_stage_codes). Kept
# identical so per-stage features bin the same epochs.
# --------------------------------------------------------------------------- #
MIN_BOUT_EPOCHS = 2  # export default (CAISR_MIN_BOUT_EPOCHS unset -> "2")


def _segments(codes):
    """[(start, end_exclusive, code), ...] contiguous runs in `codes`."""
    segs = []
    if len(codes) == 0:
        return segs
    start = 0
    for i in range(1, len(codes)):
        if codes[i] != codes[i - 1]:
            segs.append((start, i, int(codes[start])))
            start = i
    segs.append((start, len(codes), int(codes[start])))
    return segs


def smooth_stages(codes, min_bout_epochs=MIN_BOUT_EPOCHS):
    """Merge implausibly short interior stage bouts into their longer neighbour.

    Verbatim from preprocess.smooth_stages: Unknown (9) is inert; returns the
    cleaned int array (the offline exporter used only clean, discarded the mask)."""
    out = np.asarray(codes, int).copy()
    if out.size < 3 or min_bout_epochs < 2:
        return out
    while True:
        segs = _segments(out)
        shortest = None
        for k in range(1, len(segs) - 1):
            s, e, code = segs[k]
            if code == 9:
                continue
            length = e - s
            if length >= min_bout_epochs:
                continue
            if segs[k - 1][2] == 9 and segs[k + 1][2] == 9:
                continue
            if shortest is None or length < shortest[1]:
                shortest = (k, length)
        if shortest is None:
            break
        k = shortest[0]
        s, e, _ = segs[k]
        ls, le, lcode = segs[k - 1]
        rs, re_, rcode = segs[k + 1]
        left_ok, right_ok = lcode != 9, rcode != 9
        if left_ok and right_ok:
            chosen = lcode if (le - ls) >= (re_ - rs) else rcode
        else:
            chosen = lcode if left_ok else rcode
        out[s:e] = chosen
    return out


def _stage_codes(algo_data):
    """Preprocessed per-epoch stage codes from the container's CAISR dict.

    Mirrors app._caisr_stage_codes: rint the raw stage_caisr channel then smooth.
    Returns None if there is no stage channel."""
    if not algo_data:
        return None
    stage = algo_data.get("stage_caisr")
    if stage is None or len(stage) == 0:
        return None
    return smooth_stages(np.rint(np.asarray(stage, float)).astype(int))


# --------------------------------------------------------------------------- #
# Channel-role sets + EEG preference (verbatim from the three exporters).
# --------------------------------------------------------------------------- #
_ECG_ROLES = {"ecg"}
_EEG_ROLES = {"eeg"}
_RSP_ROLES = {"effort", "airflow"}
_SPO2_ROLES = {"spo2"}
_EOG_ROLES = {"eog"}
_CHIN_ROLES = {"chin_emg"}
_EEG_PREFER = ["c3", "c4", "o1", "o2"]

# --------------------------------------------------------------------------- #
# Report-block column layout + flatten helpers (verbatim from
# export_report_features.py so names+order match report_features_*.csv).
# --------------------------------------------------------------------------- #
_STAGE_TAGS = ["wake", "n1", "n2", "n3", "rem"]
_BANDS = ["delta", "theta", "alpha", "sigma", "beta"]
_SPINDLE_TAGS = ["n2", "n3"]
_RESP_TYPES = ["obstructive_apnea", "central_apnea", "hypopnea", "RERA"]


def _spectral_cols():
    cols = []
    for t in _STAGE_TAGS:
        cols.append(f"eeg_{t}_n_epochs")
        cols += [f"eeg_{t}_abs_{b}" for b in _BANDS]
        cols += [f"eeg_{t}_rel_{b}" for b in _BANDS]
        cols += [f"eeg_{t}_theta_alpha", f"eeg_{t}_delta_sigma", f"eeg_{t}_rem_slowing"]
    cols += ["eeg_ctr_rem_slowing", "eeg_ctr_n3_delta_rel",
             "eeg_ctr_n3_delta_abs", "eeg_ctr_rem_n3_slowing_ratio"]
    return cols


def _spindle_cols():
    cols = []
    for t in _SPINDLE_TAGS:
        cols += [f"spindle_{t}_{f}" for f in
                 ("minutes", "n_spindles", "density_per_min", "amp_mean", "dur_mean")]
    cols.append("spindle_threshold_uv")
    return cols


_OXY_COLS = ["spo2_mean", "spo2_min", "spo2_p1", "t90_pct", "t90_min", "odi",
             "n_desat", "desat_depth_mean", "desat_depth_max",
             "hypoxic_burden", "hb_pctmin_total", "spo2_duration_hours"]


def _resp_cols():
    cols = []
    for t in _RESP_TYPES:
        cols += [f"n_{t}", f"{t}_dur_mean_s", f"{t}_dur_max_s"]
    cols += ["n_apnea_hypopnea", "ahi", "rdi",
             "event_dur_mean_s", "event_dur_median_s", "event_dur_max_s",
             "resp_tst_hours", "post_event_overshoot", "resp_recovery_time_s"]
    return cols


_REMD_COLS = ["remd_rem_min", "remd_n_movements", "remd_rem_density_index",
              "remd_threshold_uv"]

_SPECTRAL_COLS = _spectral_cols()
_SPINDLE_COLS = _spindle_cols()
_RESP_COLS = _resp_cols()
REPORT_COLUMNS = _SPECTRAL_COLS + _SPINDLE_COLS + _OXY_COLS + _RESP_COLS + _REMD_COLS


def _flatten_spectral(band, row):
    stages = (band or {}).get("stages") or {}
    for t in _STAGE_TAGS:
        s = stages.get(t) or {}
        row[f"eeg_{t}_n_epochs"] = s.get("n_epochs")
        for b in _BANDS:
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
    for t in _SPINDLE_TAGS:
        s = stages.get(t) or {}
        for f in ("minutes", "n_spindles", "density_per_min", "amp_mean", "dur_mean"):
            row[f"spindle_{t}_{f}"] = s.get(f)
    row["spindle_threshold_uv"] = (spind or {}).get("threshold_uv")


def _flatten_oxy(oxy, row):
    oxy = oxy or {}
    for k in _OXY_COLS:
        if k == "spo2_duration_hours":
            row[k] = oxy.get("duration_hours")
        else:
            row[k] = oxy.get(k)


def _flatten_resp(rev, row):
    rev = rev or {}
    for t in _RESP_TYPES:
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


# --------------------------------------------------------------------------- #
# Block column lists (fixed order). NK first (per-stage HRV/EEG/RSP), then
# report (spectral/spindle/oxy/resp/remd), then micro (coupling/RSWA/CAP).
# --------------------------------------------------------------------------- #
NK_COLUMNS = list(nkf.NK_FEATURE_COLUMNS)
MICRO_COLUMNS = ecpl.coupling_columns() + ema.rswa_columns() + cap.cap_columns()

SIGNAL_FEATURE_COLUMNS = (
    [f"nk__{c}" for c in NK_COLUMNS]
    + [f"report__{c}" for c in REPORT_COLUMNS]
    + [f"micro__{c}" for c in MICRO_COLUMNS]
)


def _to_float(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _nk_row(phys_data, phys_fs, roles, codes):
    """nk_features block -> {NK_COLUMNS: value}. nk_stage_features picks its own
    channels from the full dict (same _pick_channel used offline)."""
    try:
        nk_out = nkf.nk_stage_features(phys_data, phys_fs, roles, codes)
        return nkf.flatten_features(nk_out)
    except Exception:
        return {}


def _report_row(phys_data, phys_fs, roles, codes, algo_data, algo_fs):
    """report block -> {REPORT_COLUMNS: value}, reproducing export_report_features."""
    row = {}
    try:
        labels = list(phys_data.keys())
        eeg_ch = nkf._pick_channel(labels, roles, _EEG_ROLES, prefer=_EEG_PREFER)
        spo2_ch = nkf._pick_channel(labels, roles, _SPO2_ROLES)
        eog_ch = nkf._pick_channel(labels, roles, _EOG_ROLES)

        _, tst_hours = clinical_report.sleep_quality(codes)

        band = (eeg_spectral.eeg_bandpower_by_stage(
            phys_data[eeg_ch], phys_fs.get(eeg_ch, 0), codes) if eeg_ch else {})
        spind = (eeg_spectral.spindle_features_by_stage(
            phys_data[eeg_ch], phys_fs.get(eeg_ch, 0), codes) if eeg_ch else {})
        oxy, spo2_norm = {}, None
        if spo2_ch:
            oxy = oxygenation.spo2_features(phys_data[spo2_ch], phys_fs.get(spo2_ch, 0))
            spo2_norm, _, _ = oxygenation._normalise_spo2(
                phys_data[spo2_ch], phys_fs.get(spo2_ch, 0))
        rev = {}
        resp_sig = (algo_data or {}).get("resp_caisr")
        if resp_sig is not None:
            rev = resp_events.resp_event_features(
                resp_sig, tst_hours=tst_hours,
                spo2=spo2_norm, spo2_fs=(phys_fs.get(spo2_ch, 0) if spo2_ch else None))
        remd = (clinical_report.rem_eye_movement_index(
            phys_data[eog_ch], phys_fs.get(eog_ch, 0), codes) if eog_ch else {})

        _flatten_spectral(band, row)
        _flatten_spindles(spind, row)
        _flatten_oxy(oxy, row)
        _flatten_resp(rev, row)
        _flatten_remd(remd, row)
    except Exception:
        pass
    return row


def _micro_row(phys_data, phys_fs, roles, codes, algo_data, algo_fs):
    """micro block -> {MICRO_COLUMNS: value}, reproducing export_micro_features."""
    row = {}
    try:
        labels = list(phys_data.keys())
        eeg_ch = nkf._pick_channel(labels, roles, _EEG_ROLES, prefer=_EEG_PREFER)
        emg_ch = nkf._pick_channel(labels, roles, _CHIN_ROLES)
        ar = (algo_data or {}).get("arousal_caisr")
        ar_fs = (algo_fs or {}).get("arousal_caisr") if algo_fs else None

        if eeg_ch is not None:
            eeg, efs = phys_data[eeg_ch], phys_fs[eeg_ch]
            ecpl.flatten_coupling(ecpl.so_spindle_coupling(eeg, efs, codes), row)
            cap.flatten_cap(
                cap.cap_features(eeg, efs, codes, arousal_codes=ar, arousal_fs=ar_fs), row)
        else:
            ecpl.flatten_coupling({"ok": False}, row)
            cap.flatten_cap({"ok": False}, row)
        if emg_ch is not None:
            ema.flatten_rswa(ema.rswa_features(phys_data[emg_ch], phys_fs[emg_ch], codes), row)
        else:
            ema.flatten_rswa({"ok": False}, row)
    except Exception:
        pass
    return row


def signal_feature_row(phys_data, phys_fs, algo_data, algo_fs):
    """Full nk + report + micro feature vector for ONE recording.

    phys_data/phys_fs: physiological EDF channels (helper_code.load_signal_data,
    keys already lower-cased). algo_data/algo_fs: CAISR annotation channels.
    Returns (np.float32 vector in SIGNAL_FEATURE_COLUMNS order, column names).
    Any missing stream or failed value is NaN; never raises.
    """
    names = list(SIGNAL_FEATURE_COLUMNS)
    n = len(names)
    try:
        phys_data = phys_data or {}
        phys_fs = phys_fs or {}
        codes = _stage_codes(algo_data)
        if not phys_data or codes is None:
            return np.full(n, np.nan, dtype=np.float32), names
        roles = {lab: channel_role(lab) for lab in phys_data.keys()}

        nk = _nk_row(phys_data, phys_fs, roles, codes)
        rep = _report_row(phys_data, phys_fs, roles, codes, algo_data, algo_fs)
        mic = _micro_row(phys_data, phys_fs, roles, codes, algo_data, algo_fs)

        vals = []
        for c in NK_COLUMNS:
            vals.append(_to_float(nk.get(c)))
        for c in REPORT_COLUMNS:
            vals.append(_to_float(rep.get(c)))
        for c in MICRO_COLUMNS:
            vals.append(_to_float(mic.get(c)))
        return np.asarray(vals, dtype=np.float32), names
    except Exception:
        return np.full(n, np.nan, dtype=np.float32), names
