#!/usr/bin/env python3
"""Reconstruct the three 1-Hz signals the Granger-G coupling estimator needs, from a
recording's raw EDF channels — matching how the existing 436-feature pipeline reads the
same signals so the new block aligns with the rest.

  * **heart** — R-peaks via `nk.ecg_clean`+`nk.ecg_peaks` (identical to
    `nk_features.ecg_hrv_by_stage`), instantaneous HR at RR-midpoints (physiologic
    300-2000 ms), linearly interpolated onto a 1 Hz grid.
  * **breath** — `nk.rsp_clean`+`nk.rsp_rate` (identical to `rsp_rate_by_stage`),
    clipped to 4-40 bpm, block-averaged to 1 Hz.
  * **eeg** — alpha-band (8-12 Hz) Butterworth + Hilbert analytic envelope on a central
    scalp derivation, block-averaged to 1 Hz. (The source used O1 alpha; we prefer
    C3/C4/O1/O2 via the pipeline's `_pick_channel`, re-referenced as-is.)

All three are returned dense (NaN-free via interpolation) on a common timeline that
starts at recording second 0, so the per-second stage labels (`np.repeat(codes, 30)`)
line up. Requires neurokit2 + scipy (box only); kept separate from `coupling.py` so the
estimator stays unit-testable without those heavy deps.
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

EPOCH_SEC = 30
RR_MIN_S, RR_MAX_S = 0.30, 2.00          # physiologic RR (matches nk_features RR_*_MS)
BPM_MIN, BPM_MAX = 4.0, 40.0             # physiologic breathing rate (matches rsp_rate_by_stage)
ALPHA_BAND = (8.0, 12.0)                 # eeg_spectral alpha band


def _fill_nan(a: np.ndarray):
    """Linear-interpolate NaNs (edge-clamped). None if fewer than 2 finite points."""
    a = np.asarray(a, float)
    idx = np.arange(len(a))
    m = np.isfinite(a)
    if m.sum() < 2:
        return None
    return np.interp(idx, idx[m], a[m])


def _block_nanmean_1hz(x: np.ndarray, fs: float):
    """Average a signal sampled at `fs` into 1-second bins (nan-aware)."""
    k = max(1, int(round(fs)))
    n = (len(x) // k) * k
    if n == 0:
        return np.array([])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(np.asarray(x[:n], float).reshape(-1, k), axis=1)


def heart_1hz(ecg: np.ndarray, fs: float):
    """Instantaneous heart rate (bpm) on a 1 Hz grid from an ECG channel."""
    import neurokit2 as nk
    ecg = np.asarray(ecg, float)
    if fs <= 0 or ecg.size < int(10 * fs):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clean = nk.ecg_clean(ecg, sampling_rate=fs)
            _, info = nk.ecg_peaks(clean, sampling_rate=fs)
        rpeaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=np.int64)
    except Exception:
        return None
    if rpeaks.size < 12:
        return None
    rt = rpeaks / fs                                  # r-peak times (s)
    rr = np.diff(rt)                                  # RR intervals (s)
    good = (rr >= RR_MIN_S) & (rr <= RR_MAX_S)
    if good.sum() < 8:
        return None
    t_mid = (rt[1:] + rt[:-1]) / 2.0
    hr = 60.0 / rr                                    # bpm
    grid = np.arange(0, int(np.floor(rt[-1])) + 1)
    return np.interp(grid, t_mid[good], hr[good])     # edge-clamped -> dense


def breath_1hz(rsp: np.ndarray, fs: float):
    """Instantaneous breathing rate (bpm) on a 1 Hz grid from an effort/airflow channel."""
    import neurokit2 as nk
    rsp = np.asarray(rsp, float)
    if fs <= 0 or rsp.size < int(10 * fs):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clean = nk.rsp_clean(rsp, sampling_rate=fs)
            rate = np.asarray(nk.rsp_rate(clean, sampling_rate=fs), float)
    except Exception:
        return None
    rate = np.where(np.isfinite(rate) & (rate > BPM_MIN) & (rate < BPM_MAX), rate, np.nan)
    if np.isfinite(rate).sum() < int(10 * fs):
        return None
    return _fill_nan(_block_nanmean_1hz(rate, fs))


def eeg_alpha_1hz(eeg: np.ndarray, fs: float, band=ALPHA_BAND):
    """Alpha-band analytic-envelope amplitude on a 1 Hz grid from one EEG channel."""
    from scipy.signal import butter, filtfilt, hilbert
    eeg = np.asarray(eeg, float)
    if fs <= 0 or eeg.size < int(10 * fs):
        return None
    eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0)
    nyq = fs / 2.0
    lo, hi = band[0] / nyq, min(band[1] / nyq, 0.99)
    if not (0 < lo < hi < 1):
        return None
    try:
        b, a = butter(4, [lo, hi], btype="band")
        env = np.abs(hilbert(filtfilt(b, a, eeg)))
    except Exception:
        return None
    return _block_nanmean_1hz(env, fs)


def build_signals(channels: dict, fss: dict, roles: dict, codes, pick_channel):
    """Assemble the aligned 1-Hz (breath, heart, eeg, stage) arrays for one recording.

    channels/fss/roles: {label -> samples/Hz/role}; codes: per-epoch (30 s) stage codes;
    pick_channel: nk_features._pick_channel. Returns (dict of signals + chosen labels)
    or None if a required channel/signal is missing. `stage` is per-second.
    """
    labels = list(channels.keys())
    ecg_lab = pick_channel(labels, roles, {"ecg"})
    rsp_lab = pick_channel(labels, roles, {"effort", "airflow"})
    eeg_lab = pick_channel(labels, roles, {"eeg"}, prefer=["c3", "c4", "o1", "o2"])
    if not (ecg_lab and rsp_lab and eeg_lab):
        return None

    heart = heart_1hz(channels[ecg_lab], fss[ecg_lab])
    breath = breath_1hz(channels[rsp_lab], fss[rsp_lab])
    eeg = eeg_alpha_1hz(channels[eeg_lab], fss[eeg_lab])
    if heart is None or breath is None or eeg is None:
        return None

    codes = np.rint(np.asarray(codes, float)).astype(int)
    stage = np.repeat(codes, EPOCH_SEC)               # per-second stage labels
    T = min(len(heart), len(breath), len(eeg), len(stage))
    if T < 60:
        return None
    return {
        "breath": breath[:T], "heart": heart[:T], "eeg": eeg[:T], "stage": stage[:T],
        "ecg_ch": ecg_lab, "rsp_ch": rsp_lab, "eeg_ch": eeg_lab, "seconds": int(T),
    }
