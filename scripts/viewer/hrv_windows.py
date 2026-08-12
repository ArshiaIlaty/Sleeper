"""Non-averaged per-stage HRV — short-window HRV *within* each sleep stage.

`nk_features.ecg_hrv_by_stage` pools every same-stage RR interval into ONE HRV
number per stage (the MEAN view in `nk_features_*.csv`). That collapses the
within-stage variability, which is itself the cognitive-impairment signal (the
"blunted / more erratic autonomic modulation" thesis): two nights with the same
mean N2 RMSSD can differ sharply in how stable that RMSSD is minute to minute.

HRV can't be computed per 30 s epoch (SDNN/RMSSD/LF-HF need a window of many
beats), so the un-aggregated grain for HRV is a short **time window** (default
120 s of beats) rather than one epoch. This module:

  * `hrv_windows_by_stage` — detect R-peaks once, label each RR by the stage of its
    beat (dropping non-physiologic RR and RR that straddle a stage change, exactly
    like `ecg_hrv_by_stage`), then chunk each stage's RR series into consecutive
    ~`win_sec` windows and compute full linear HRV per window. Yields one dict per
    (stage, window) — the LONG, fully un-aggregated form.
  * `hrv_dispersion` — summarise those per-window HRV values into the spread across
    windows per stage (SD / CV / p10 / p50 / p90 / IQR), the NON-AVG wide companion
    that slots beside `dispersion_features_*.csv`.

Reuses `nk_features`' exact R-peak detection, RR filtering, and `_hrv_from_rr`
(same NeuroKit `hrv_time` / `hrv_frequency` definitions + closed-form Poincare), so
a window's HRV is defined identically to the pooled per-stage HRV — only the
aggregation grain differs. Every field degrades to None rather than raising.
"""
import warnings

import numpy as np

import nk_features as nkf

EPOCH_SEC = nkf.EPOCH_SEC
STAGE_ORDER = nkf.STAGE_ORDER                     # [5,3,2,1,4] wake,n1,n2,n3,rem
POOL_TAG = nkf.POOL_TAG
STAGE_TAGS = ["wake", "n1", "n2", "n3", "rem"]

# window sizing
HRV_WIN_SEC = 120.0          # target window length (seconds of RR)
HRV_MIN_WIN_BEATS = 30       # emit a window only with >= this many beats (time-domain floor)

# HRV metrics carried in the long table (order stable).
HRV_LONG_METRICS = ["hr_mean", "meannn", "sdnn", "rmssd", "pnn50", "sdsd", "cvnn",
                    "sd1", "sd2", "sd1sd2", "lf", "hf", "lfhf", "lfn", "hfn", "tp"]
# dispersion: FULL stats (SD/CV/percentiles) for the flagship metrics; COMPACT
# (SD/CV) for the rest — keeps the wide table's column count in check.
DISP_FULL = ["hr_mean", "rmssd", "sdnn", "lfhf"]
DISP_COMPACT = ["pnn50", "cvnn", "sd1sd2", "lf", "hf"]


def _finite(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _r(x, nd):
    v = _finite(x)
    return None if v is None else round(v, nd)


def _rpeaks_and_stages(ecg, fs, stage_codes):
    """Whole-night R-peak detection -> (rr_ms, beat_time_s, beat_stage), physiologic
    RR only, both endpoints in the *same* real stage. Shared with `ecg_hrv_by_stage`
    logic verbatim so the window HRV matches the pooled per-stage HRV definition."""
    ecg = np.asarray(ecg, float)
    fs = float(fs)
    codes = np.rint(np.asarray(stage_codes, float)).astype(int)
    if not nkf._NK_OK:
        return None, None, None, "neurokit2 unavailable"
    if fs <= 0 or ecg.size < int(10 * fs) or codes.size == 0:
        return None, None, None, "no usable ECG / staging"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clean = nkf.nk.ecg_clean(ecg, sampling_rate=fs)
            _, info = nkf.nk.ecg_peaks(clean, sampling_rate=fs)
        rpeaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=np.int64)
    except Exception as e:
        return None, None, None, f"peak detection failed: {type(e).__name__}"
    if rpeaks.size < nkf.MIN_BEATS_TIME + 1:
        return None, None, None, f"only {rpeaks.size} beats detected"

    rr_ms = np.diff(rpeaks) / fs * 1000.0
    beat_time = rpeaks[:-1] / fs                       # start time of each RR interval (s)
    ep = np.clip((rpeaks[:-1] / fs // EPOCH_SEC).astype(int), 0, codes.size - 1)
    ep_next = np.clip((rpeaks[1:] / fs // EPOCH_SEC).astype(int), 0, codes.size - 1)
    beat_stage = codes[ep]
    next_stage = codes[ep_next]
    good = ((rr_ms >= nkf.RR_MIN_MS) & (rr_ms <= nkf.RR_MAX_MS)
            & (beat_stage == next_stage) & (beat_stage != 9))
    return rr_ms[good], beat_time[good], beat_stage[good], None


def _chunk_by_time(rr_ms, win_ms):
    """Consecutive (start, end) index slices of `rr_ms`, each accumulating about
    `win_ms` of RR duration (the last chunk carries the remainder)."""
    chunks = []
    start = 0
    acc = 0.0
    for i, v in enumerate(rr_ms):
        acc += v
        if acc >= win_ms:
            chunks.append((start, i + 1))
            start = i + 1
            acc = 0.0
    if start < rr_ms.size:
        chunks.append((start, rr_ms.size))
    return chunks


def hrv_windows_by_stage(ecg, fs, stage_codes, win_sec=HRV_WIN_SEC,
                         min_beats=HRV_MIN_WIN_BEATS, do_freq=True):
    """Per-(stage, window) HRV — the long, un-aggregated form.

    Returns {"ok", "n_beats_total", "windows": [ {stage, window_index, start_s,
    dur_s, n_beats, <HRV metrics...>}, ... ]}. Windows with < `min_beats` beats are
    dropped (HRV undefined). `do_freq=False` skips `hrv_frequency` (lf/hf/... None)
    for a cheaper time-domain-only pass.
    """
    rr_ms, beat_time, beat_stage, err = _rpeaks_and_stages(ecg, fs, stage_codes)
    if err:
        return {"ok": False, "error": err, "windows": []}
    fs = float(fs)
    win_ms = win_sec * 1000.0
    windows = []
    for code in STAGE_ORDER:
        sel = beat_stage == code
        sel_rr = rr_ms[sel]
        sel_t = beat_time[sel]
        if sel_rr.size < min_beats:
            continue
        for wi, (a, b) in enumerate(_chunk_by_time(sel_rr, win_ms)):
            win_rr = sel_rr[a:b]
            if win_rr.size < min_beats:
                continue
            hrv = _hrv_window(win_rr, fs, do_freq=do_freq)
            row = {"stage": POOL_TAG[code], "window_index": wi,
                   "start_s": _r(float(sel_t[a]), 1),
                   "dur_s": _r(float(win_rr.sum() / 1000.0), 1),
                   "n_beats": int(win_rr.size)}
            row.update(hrv)
            windows.append(row)
    return {"ok": True, "n_beats_total": int(rr_ms.size), "windows": windows}


def _hrv_window(rr_ms, fs, do_freq=True):
    """Linear HRV for one window's RR series, restricted to HRV_LONG_METRICS.

    Thin wrapper over `nk_features._hrv_from_rr` (same NeuroKit definitions); with
    `do_freq=False` the frequency-domain features are forced to None without calling
    `hrv_frequency`, so a window costs only the cheap time-domain + Poincare pass.
    """
    if do_freq:
        full = nkf._hrv_from_rr(rr_ms, fs)
    else:
        saved = nkf.MIN_BEATS_FREQ
        nkf.MIN_BEATS_FREQ = np.inf                    # gate out the frequency pass
        try:
            full = nkf._hrv_from_rr(rr_ms, fs)
        finally:
            nkf.MIN_BEATS_FREQ = saved
    return {m: full.get(m) for m in HRV_LONG_METRICS}


# ------------------------------------------------------------------ dispersion
def _disp(vals, nd=4, full=False):
    """SD / CV (+ p10/p50/p90/IQR when full) of a list of per-window values.

    Mirrors `stage_dispersion._disp`: always returns n + mean; spread stats need
    >= 2 finite values."""
    v = np.asarray([x for x in (_finite(u) for u in vals) if x is not None], float)
    keys = (["sd", "cv", "p10", "p50", "p90", "iqr"] if full else ["sd", "cv"])
    out = {"n": int(v.size), "mean": _r(v.mean(), nd) if v.size else None}
    for k in keys:
        out[k] = None
    if v.size >= 2:
        m = float(v.mean())
        sd = float(v.std(ddof=1))
        m_all = {"sd": sd, "cv": (sd / abs(m)) if m != 0 else None,
                 "p10": float(np.percentile(v, 10)), "p50": float(np.percentile(v, 50)),
                 "p90": float(np.percentile(v, 90)),
                 "iqr": float(np.percentile(v, 75) - np.percentile(v, 25))}
        for k in keys:
            out[k] = _r(m_all[k], nd)
    return out


def hrv_dispersion(ecg, fs, stage_codes, win_sec=HRV_WIN_SEC, do_freq=True):
    """Per-stage dispersion of the windowed HRV metrics (the wide non-avg view).

    Returns {"ok", "stages": {tag: {n_windows, <metric>: {sd,cv,...}}}}. Computed
    from the same windows `hrv_windows_by_stage` emits, so the long table and this
    wide table are exactly consistent.
    """
    w = hrv_windows_by_stage(ecg, fs, stage_codes, win_sec=win_sec, do_freq=do_freq)
    if not w.get("ok"):
        return {"ok": False, "error": w.get("error"), "stages": {}}
    by_stage = {}
    for row in w["windows"]:
        by_stage.setdefault(row["stage"], []).append(row)
    stages = {}
    for tag in STAGE_TAGS:
        wins = by_stage.get(tag)
        if not wins:
            continue
        s = {"n_windows": len(wins)}
        for m in DISP_FULL:
            s[m] = _disp([x.get(m) for x in wins], nd=4, full=True)
        for m in DISP_COMPACT:
            s[m] = _disp([x.get(m) for x in wins], nd=4, full=False)
        stages[tag] = s
    return {"ok": True, "stages": stages}
