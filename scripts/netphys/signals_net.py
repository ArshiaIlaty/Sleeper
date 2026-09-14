#!/usr/bin/env python3
"""Multi-node 1-Hz signal reconstruction for network-physiology (TDS) graphs.

Extends signals.py from 3 nodes to a full node set:
  * BRAIN — cortical-rhythm envelopes per scalp region x frequency band
    (delta/theta/alpha/sigma/beta on frontal / central / occipital derivations).
    Region x band gives both the intra-channel (cross-frequency) and inter-channel
    (cross-region) brain networks.
  * ORGANS — the key systems: heart (HR), respiration (RESP), chin muscle tone (EMG),
    ocular activity (EOG), blood-oxygen (SpO2).

Every node is a dense, NaN-free 1-Hz series on one timeline aligned to the per-second
stage codes, so `tds.compute_tds_networks` can build one network per sleep stage.
Reuses signals.py's heart_1hz / breath_1hz / block-mean helpers so nodes match how the
436-feature pipeline reads the same channels.

Channel picking is robust to montage heterogeneity: try the role vocabulary first
(`nk_features._pick_channel`), then fall back to case-insensitive label-substring match
so EMG/EOG/SpO2 are still found if `channel_role` does not categorise them. The driver
reports per-node coverage so a mis-guessed token is visible, never silent.
"""
from __future__ import annotations

import warnings

import numpy as np

from signals import heart_1hz, breath_1hz, _fill_nan, _block_nanmean_1hz, EPOCH_SEC

warnings.filterwarnings("ignore")

BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0),
         "sigma": (12.0, 16.0), "beta": (16.0, 30.0)}

# region -> (role set, prefer-substrings for _pick_channel / fallback label match)
REGIONS = {
    "F": ["f3", "f4", "fz", "fp1", "fp2", "f7", "f8"],
    "C": ["c3", "c4", "cz"],
    "O": ["o1", "o2", "oz"],
}
# organ node -> (role tokens, label substrings) for the robust picker
ORGAN_SPECS = {
    "HR":   ({"ecg"}, ["ecg", "ekg"]),
    "RESP": ({"effort", "airflow"}, ["effort", "flow", "thor", "abdo", "resp", "chest"]),
    "EMG":  ({"chin_emg"}, ["chin", "emg", "submental", "menton"]),
    "EOG":  ({"eog"}, ["eog", "e1", "e2", "loc", "roc"]),
    "SpO2": ({"spo2", "sao2", "oxygen", "sat", "spO2".lower()}, ["spo2", "sao2", "osat", "sat"]),
}


def _pick_any(labels, roles, roleset, prefer):
    """Role-based pick first; fall back to case-insensitive label-substring match."""
    from nk_features import _pick_channel  # noqa: local import so tds tests need no viewer
    lab = _pick_channel(labels, roles, roleset, prefer=prefer) if roleset else None
    if lab:
        return lab
    low = {l: l.lower() for l in labels}
    for sub in prefer:
        for l in labels:
            if sub in low[l]:
                return l
    return None


def band_env_1hz(x, fs, band):
    """Analytic-envelope amplitude of a frequency band, block-averaged to 1 Hz."""
    from scipy.signal import butter, filtfilt, hilbert
    x = np.nan_to_num(np.asarray(x, float), nan=0.0, posinf=0.0, neginf=0.0)
    if fs <= 0 or x.size < int(10 * fs):
        return None
    nyq = fs / 2.0
    lo, hi = band[0] / nyq, min(band[1] / nyq, 0.99)
    if not (0 < lo < hi < 1):
        return None
    try:
        b, a = butter(4, [lo, hi], btype="band")
        env = np.abs(hilbert(filtfilt(b, a, x)))
    except Exception:
        return None
    return _block_nanmean_1hz(env, fs)


def activity_1hz(x, fs, hp=None):
    """Generic muscle/ocular activity: optional highpass -> rectify -> 1-Hz block mean."""
    x = np.nan_to_num(np.asarray(x, float), nan=0.0, posinf=0.0, neginf=0.0)
    if fs <= 0 or x.size < int(10 * fs):
        return None
    if hp:
        from scipy.signal import butter, filtfilt
        w = hp / (fs / 2.0)
        if 0 < w < 1:
            try:
                b, a = butter(4, w, btype="high")
                x = filtfilt(b, a, x)
            except Exception:
                pass
    return _block_nanmean_1hz(np.abs(x), fs)


def level_1hz(x, fs):
    """A slowly-varying level channel (SpO2) block-averaged to 1 Hz, NaNs filled."""
    x = np.asarray(x, float)
    if fs <= 0 or x.size == 0:
        return None
    return _fill_nan(_block_nanmean_1hz(x, fs))


def _clean(a):
    """Ensure a dense finite 1-Hz node; None if unusable."""
    if a is None:
        return None
    a = np.asarray(a, float)
    if a.size < 60 or not np.isfinite(a).any():
        return None
    return _fill_nan(a)


def build_node_signals(channels, fss, roles, codes, regions=("C", "O"),
                       bands=("delta", "theta", "alpha", "sigma", "beta")):
    """Assemble the aligned 1-Hz node dict + per-second stage for one recording.

    channels/fss/roles: {label -> samples/Hz/role}. Returns
    {signals: {node -> 1Hz}, stage: per-second codes, meta: {node -> channel used},
     seconds: T} or None if fewer than 3 usable nodes.
    """
    labels = list(channels.keys())
    sig, meta = {}, {}

    for reg in regions:
        lab = _pick_any(labels, roles, {"eeg"}, REGIONS[reg])
        meta[f"eeg_{reg}"] = lab
        if not lab:
            continue
        for bd in bands:
            env = _clean(band_env_1hz(channels[lab], fss[lab], BANDS[bd]))
            if env is not None:
                sig[f"{reg}.{bd}"] = env

    for node, (roleset, subs) in ORGAN_SPECS.items():
        lab = _pick_any(labels, roles, roleset, subs)
        meta[node] = lab
        if not lab:
            continue
        if node == "HR":
            v = heart_1hz(channels[lab], fss[lab])
        elif node == "RESP":
            v = breath_1hz(channels[lab], fss[lab])
        elif node == "EMG":
            v = activity_1hz(channels[lab], fss[lab], hp=10.0)
        elif node == "EOG":
            v = activity_1hz(channels[lab], fss[lab], hp=0.3)
        else:  # SpO2
            v = level_1hz(channels[lab], fss[lab])
        v = _clean(v)
        if v is not None:
            sig[node] = v

    if len(sig) < 3:
        return None

    codes = np.rint(np.asarray(codes, float)).astype(int)
    stage = np.repeat(codes, EPOCH_SEC)
    T = min([len(v) for v in sig.values()] + [len(stage)])
    if T < 300:
        return None
    sig = {k: np.asarray(v, float)[:T] for k, v in sig.items()}
    return {"signals": sig, "stage": stage[:T], "meta": meta, "seconds": int(T)}
