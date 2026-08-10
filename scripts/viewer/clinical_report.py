"""Assemble a per-patient clinical sleep report.

Pulls together every signal-derived feature the viewer computes and organises it
the way a clinician-facing summary would, into the tiers the team prioritised:

  Tier 1 (primary CI-relevant markers)
    1. REM EEG slowing        (eeg_spectral: (delta+theta)/(alpha+sigma+beta) in REM)
    2. N3 slow-wave activity  (eeg_spectral: relative delta power in N3)
    3. Spindle density        (eeg_spectral: N2 sigma-burst density /min)
    4. Hypoxic burden         (oxygenation: desat depth x duration, %*min/h)
    5. Sleep fragmentation    (arousal index + awakenings per hour of sleep)

  Tier 2 (supporting markers)
    1. Stage-specific HRV     (nk_features: RMSSD range across stages)
    2. NREM parasympathetic   (nk_features: RMSSD pooled over NREM)
    3. Respiratory instability(nk_features: RRV CV, pooled)
    4. REM density            (EOG rapid-eye-movement index within REM — a proxy)

  Clinical feature table: spindle density, slow-wave activity, REM slowing,
  hypoxic burden, stage-specific HRV, AHI, arousal index, PLMI.

  Sleep-quality metrics: efficiency, WASO, sleep latency, REM latency, N3 latency,
  number of awakenings, TST.

This module is the orchestrator: it receives the already-decoded physiological
channels + the CAISR annotation channels + the preprocessed staging, calls the
per-domain modules (nk_features / eeg_spectral / oxygenation / resp_events), and
returns one JSON-serialisable report dict. Heavy signal decoding stays in the
caller (app.py / an export CLI); nothing here reads files. Degrades field-by-field
to None rather than raising.
"""
import numpy as np

import nk_features
import eeg_spectral
import oxygenation
import resp_events

EPOCH_SEC = 30.0
ASLEEP = (1, 2, 3, 4)                          # everything but Wake / Unknown


def _finite(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _r(x, nd=3):
    v = _finite(x)
    return None if v is None else round(v, nd)


# --------------------------------------------------------------------------- sleep quality
def sleep_quality(stage_codes):
    """Sleep-architecture quality metrics from the (preprocessed) stage codes."""
    st = np.rint(np.asarray(stage_codes, float)).astype(int)
    out = {k: None for k in ("tst_min", "sleep_efficiency_pct", "sleep_latency_min",
                             "rem_latency_min", "n3_latency_min", "waso_min",
                             "n_awakenings", "awakenings_per_hr_sleep", "n_epochs")}
    if st.size == 0:
        return out
    n_epochs = st.size
    out["n_epochs"] = int(n_epochs)
    asleep = np.isin(st, ASLEEP)
    n_asleep = int(asleep.sum())
    tst_min = n_asleep * EPOCH_SEC / 60.0
    tst_hours = tst_min / 60.0
    out["tst_min"] = _r(tst_min, 1)
    out["sleep_efficiency_pct"] = _r(100.0 * n_asleep / n_epochs, 1) if n_epochs else None
    asleep_idx = np.where(asleep)[0]
    if asleep_idx.size:
        onset = int(asleep_idx[0])
        offset = int(asleep_idx[-1])
        out["sleep_latency_min"] = _r(onset * EPOCH_SEC / 60.0, 1)
        waso = int(np.count_nonzero(st[onset:offset + 1] == 5))
        out["waso_min"] = _r(waso * EPOCH_SEC / 60.0, 1)
        rem_idx = np.where(st == 4)[0]
        n3_idx = np.where(st == 1)[0]
        out["rem_latency_min"] = _r((rem_idx[0] - onset) * EPOCH_SEC / 60.0, 1) if rem_idx.size else None
        out["n3_latency_min"] = _r((n3_idx[0] - onset) * EPOCH_SEC / 60.0, 1) if n3_idx.size else None
        # awakenings: sleep -> wake transitions after onset
        awk = 0
        for i in range(onset, st.size - 1):
            if st[i] in ASLEEP and st[i + 1] == 5:
                awk += 1
        out["n_awakenings"] = int(awk)
        out["awakenings_per_hr_sleep"] = _r(awk / tst_hours, 2) if tst_hours > 0 else None
    return out, tst_hours if asleep_idx.size else None


# --------------------------------------------------------------------------- event indices
def _event_index(sig, codes, fs, hours):
    """Onsets of `codes` in a per-sample label stream, per hour."""
    if sig is None or len(sig) == 0 or not hours or hours <= 0 or not fs:
        return None
    s = np.rint(np.asarray(sig, float)).astype(int)
    mask = np.isin(s, list(codes))
    onsets = int(np.count_nonzero(np.diff(mask.astype(np.int8), prepend=0) == 1))
    return _r(onsets / hours, 2)


# --------------------------------------------------------------------------- REM density
def rem_eye_movement_index(eog, fs, stage_codes):
    """Rapid-eye-movement index: eye-movement deflections per minute of REM.

    A proxy for REM density from a single EOG channel: band-pass the EOG to the
    eye-movement band (0.5-5 Hz), and within REM epochs count threshold-crossing
    deflections (peaks above 2*robust-SD of the REM EOG), normalised per minute.
    Not a validated REM-density score, but tracks REM eye-movement intensity.
    """
    if not eeg_spectral._SCIPY_OK:
        return {"ok": False, "error": "scipy unavailable"}
    from scipy import signal as sp
    eog = np.asarray(eog, float)
    fs = float(fs)
    codes = np.rint(np.asarray(stage_codes, float)).astype(int)
    if fs <= 0 or eog.size == 0 or codes.size == 0:
        return {"ok": False, "error": "no usable EOG / staging"}
    spe = int(round(fs * EPOCH_SEC))
    if spe < 1:
        return {"ok": False, "error": "sampling rate too low"}
    n_ep = min(codes.size, eog.size // spe)
    rem_idx = np.where(codes[:n_ep] == 4)[0]
    rem_min = rem_idx.size * EPOCH_SEC / 60.0
    if rem_min < 2.0:
        return {"ok": False, "error": "insufficient REM sleep"}
    x = np.nan_to_num(eog[:n_ep * spe], nan=0.0)
    ny = 0.5 * fs
    hi = min(5.0, ny * 0.99)
    try:
        b, a = sp.butter(4, [0.5 / ny, hi / ny], btype="band")
        filt = sp.filtfilt(b, a, x)
    except Exception as e:
        return {"ok": False, "error": f"EOG filter failed: {type(e).__name__}"}
    # gather REM samples, set a robust amplitude threshold from them
    rem_mask = np.zeros(filt.size, bool)
    for e in rem_idx:
        rem_mask[e * spe:(e + 1) * spe] = True
    rem_sig = filt[rem_mask]
    if rem_sig.size < fs * 10:
        return {"ok": False, "error": "insufficient REM EOG"}
    mad = float(np.median(np.abs(rem_sig - np.median(rem_sig)))) or float(rem_sig.std())
    thr = 3.0 * 1.4826 * mad                    # ~3 robust SD
    if thr <= 0:
        return {"ok": False, "error": "flat EOG"}
    # count deflections within REM: peaks of |filt| above thr, min 0.15 s apart
    absr = np.abs(filt) * rem_mask
    peaks, _ = sp.find_peaks(absr, height=thr, distance=max(1, int(0.15 * fs)))
    n = int(peaks.size)
    return {"ok": True, "rem_min": _r(rem_min, 1), "n_movements": n,
            "rem_density_index": _r(n / rem_min, 2) if rem_min > 0 else None,
            "threshold_uv": _r(thr, 2)}


# --------------------------------------------------------------------------- assemble
def _item(key, label, value, unit="", note=None, worse=None):
    return {"key": key, "label": label, "value": value, "unit": unit,
            "note": note, "higher_is_worse": worse}


def build_report(phys_channels, phys_fss, roles, stage_codes,
                 caisr_chans=None, caisr_fss=None):
    """Assemble the full clinical report. See module docstring.

    phys_channels/phys_fss: decoded physiological channels {label: samples}/{label: Hz}.
    roles: {label: role} (app.channel_role). stage_codes: preprocessed epoch codes.
    caisr_chans/caisr_fss: CAISR annotation channels (resp_caisr / arousal_caisr /
    limb_caisr) + their sampling rates. Missing inputs degrade gracefully.
    """
    caisr_chans = caisr_chans or {}
    caisr_fss = caisr_fss or {}
    labels = list(phys_channels.keys())
    used = {}

    # --- pick channels ---
    ecg_ch = nk_features._pick_channel(labels, roles, {"ecg"})
    eeg_ch = nk_features._pick_channel(labels, roles, {"eeg"}, prefer=["c3", "c4", "o1", "o2"])
    rsp_ch = nk_features._pick_channel(labels, roles, {"effort", "airflow"})
    eog_ch = nk_features._pick_channel(labels, roles, {"eog"})
    spo2_ch = nk_features._pick_channel(labels, roles, {"spo2"})
    used = {"ecg": ecg_ch, "eeg": eeg_ch, "rsp": rsp_ch, "eog": eog_ch, "spo2": spo2_ch}

    # --- sleep quality + hours ---
    sq, tst_hours = sleep_quality(stage_codes)

    # --- per-domain computation ---
    hrv = (nk_features.ecg_hrv_by_stage(phys_channels[ecg_ch], phys_fss.get(ecg_ch, 0), stage_codes)
           if ecg_ch else {"ok": False})
    band = (eeg_spectral.eeg_bandpower_by_stage(phys_channels[eeg_ch], phys_fss.get(eeg_ch, 0), stage_codes)
            if eeg_ch else {"ok": False})
    spind = (eeg_spectral.spindle_features_by_stage(phys_channels[eeg_ch], phys_fss.get(eeg_ch, 0), stage_codes)
             if eeg_ch else {"ok": False})
    complexity = (nk_features.eeg_complexity_by_stage(phys_channels[eeg_ch], phys_fss.get(eeg_ch, 0), stage_codes)
                  if eeg_ch else {"ok": False})
    rspv = (nk_features.rsp_rate_by_stage(phys_channels[rsp_ch], phys_fss.get(rsp_ch, 0), stage_codes)
            if rsp_ch else {"ok": False})
    remd = (rem_eye_movement_index(phys_channels[eog_ch], phys_fss.get(eog_ch, 0), stage_codes)
            if eog_ch else {"ok": False})

    # oxygenation (scale-normalised inside the module)
    oxy = {k: None for k in ("hypoxic_burden", "odi", "t90_pct", "spo2_min", "spo2_mean")}
    spo2_norm = None
    if spo2_ch:
        oxy = oxygenation.spo2_features(phys_channels[spo2_ch], phys_fss.get(spo2_ch, 0))
        spo2_norm, spo2_fs_n, _ = oxygenation._normalise_spo2(
            phys_channels[spo2_ch], phys_fss.get(spo2_ch, 0))

    # respiratory events (+ recovery via SpO2)
    resp_sig = caisr_chans.get("resp_caisr")
    revents = {"ok": False}
    if resp_sig is not None:
        revents = resp_events.resp_event_features(
            resp_sig, tst_hours=tst_hours,
            spo2=spo2_norm, spo2_fs=(phys_fss.get(spo2_ch, 0) if spo2_ch else None))

    # event indices for the clinical table (arousal + PLMI from CAISR)
    hours_for_idx = tst_hours if (tst_hours and tst_hours > 0) else None
    if hours_for_idx is None and resp_sig is not None:
        hours_for_idx = len(resp_sig) / (float(caisr_fss.get("resp_caisr", 1.0)) * 3600.0)
    arousal_idx = _event_index(caisr_chans.get("arousal_caisr"), (1,),
                               caisr_fss.get("arousal_caisr"), hours_for_idx)
    plmi = _event_index(caisr_chans.get("limb_caisr"), (2,),
                        caisr_fss.get("limb_caisr"), hours_for_idx)
    ahi = revents.get("ahi") if revents.get("ok") else None

    # --- flagship scalars pulled from the domains ---
    rem_slowing = (band.get("contrasts") or {}).get("rem_slowing") if band.get("ok") else None
    n3_swa = (band.get("contrasts") or {}).get("n3_delta_rel") if band.get("ok") else None
    spindle_density = ((spind.get("stages") or {}).get("n2") or {}).get("density_per_min") if spind.get("ok") else None
    hypoxic_burden = oxy.get("hypoxic_burden")
    frag_index = None
    if arousal_idx is not None or sq.get("awakenings_per_hr_sleep") is not None:
        frag_index = _r((arousal_idx or 0) + (sq.get("awakenings_per_hr_sleep") or 0), 2)

    # stage-specific HRV summary: RMSSD range across the 5 stages + NREM RMSSD
    hrv_stage_range = (hrv.get("contrasts") or {}).get("stage_rmssd_cv") if hrv.get("ok") else None
    nrem_rmssd = ((hrv.get("stages") or {}).get("nrem") or {}).get("rmssd") if hrv.get("ok") else None
    resp_instability = ((rspv.get("stages") or {}).get("sleep") or {}).get("rrv_cv") if rspv.get("ok") else None
    if resp_instability is None and rspv.get("ok"):     # fall back to N2 if no pooled sleep tag
        resp_instability = ((rspv.get("stages") or {}).get("n2") or {}).get("rrv_cv")
    rem_density = remd.get("rem_density_index") if remd.get("ok") else None

    tier1 = [
        _item("rem_slowing", "REM EEG slowing", rem_slowing, "", worse=True,
              note="(delta+theta)/(alpha+sigma+beta) in REM; higher = more slowing"),
        _item("n3_swa", "N3 slow-wave activity", n3_swa, "rel", worse=False,
              note="relative delta power in deep sleep; lower = less restorative slow-wave"),
        _item("spindle_density", "Spindle density (N2)", spindle_density, "/min", worse=False,
              note="sigma-burst density in N2; reduced density tracks memory-consolidation deficits"),
        _item("hypoxic_burden", "Hypoxic burden", hypoxic_burden, "%·min/h", worse=True,
              note="desaturation depth × duration per hour; captures intermittent hypoxia dose"),
        _item("frag_index", "Sleep fragmentation index", frag_index, "/h", worse=True,
              note="arousals + awakenings per hour of sleep"),
    ]
    tier2 = [
        _item("hrv_stage_range", "Stage-specific HRV modulation", hrv_stage_range, "CV", worse=False,
              note="coefficient of variation of RMSSD across stages; small = blunted swing"),
        _item("nrem_parasympathetic", "NREM parasympathetic (RMSSD)", nrem_rmssd, "ms", worse=False,
              note="pooled RMSSD in NREM; vagal tone during deep sleep"),
        _item("resp_instability", "Respiratory instability", resp_instability, "CV", worse=True,
              note="breath-rate coefficient of variation during sleep"),
        _item("rem_density", "REM density (eye-movement index)", rem_density, "/min", worse=None,
              note="EOG rapid-eye-movement deflections per minute of REM (proxy)"),
    ]
    clinical = [
        _item("spindle_density", "Spindle density (N2)", spindle_density, "/min"),
        _item("n3_swa", "Slow-wave activity (N3 rel. delta)", n3_swa, "rel"),
        _item("rem_slowing", "REM slowing", rem_slowing, ""),
        _item("hypoxic_burden", "Hypoxic burden", hypoxic_burden, "%·min/h"),
        _item("nrem_rmssd", "Stage-specific HRV (NREM RMSSD)", nrem_rmssd, "ms"),
        _item("ahi", "AHI", ahi, "/h", worse=True),
        _item("arousal_index", "Arousal index", arousal_idx, "/h", worse=True),
        _item("plmi", "PLMI", plmi, "/h", worse=True),
    ]

    return {
        "ok": True,
        "channels_used": used,
        "sleep_quality": sq,
        "tier1": tier1,
        "tier2": tier2,
        "clinical": clinical,
        # full sub-module payloads for optional drill-down in the UI
        "detail": {
            "hrv": hrv, "bandpower": band, "spindles": spind,
            "complexity": complexity, "respiration": rspv,
            "oxygenation": oxy, "resp_events": revents, "rem_density": remd,
        },
    }
