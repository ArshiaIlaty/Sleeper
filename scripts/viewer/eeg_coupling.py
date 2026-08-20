"""Slow-oscillation–spindle coupling (B3) from one central EEG channel.

During healthy NREM sleep, sleep spindles (11-16 Hz sigma bursts) are
*phase-locked* to the up-state of the cortical slow oscillation (SO, ~0.5-1.25 Hz).
That coupling is the electrophysiological signature of hippocampal-neocortical
memory consolidation, and its **decline is one of the leading candidate EEG
biomarkers of aging and cognitive impairment** (Helfrich 2018; Winer 2019): older
/ impaired brains still produce spindles and SOs, but the spindle stops landing on
the SO up-state. So the discriminative quantity is not spindle *count* (we already
have that) but the *precision and phase* of the coupling.

Method (all reusing eeg_spectral's primitives so the spindle detection is
identical to `spindle_features_by_stage`):
  1. Detect spindles within each NREM stage (reuse eeg_spectral spindle detector).
  2. For each spindle, take its **peak time** = argmax of the sigma envelope.
  3. Filter the SAME channel in the SO band and take the **instantaneous SO phase**
     (angle of the analytic signal) at each spindle peak.
  4. Summarise the circular distribution of those phases:
       * **coupling strength** = mean resultant vector length (MVL, |mean e^{iφ}|)
         in [0,1] — 1 = every spindle at the same SO phase, 0 = uniform/no coupling.
         MVL is nearly independent of spindle count, unlike raw counts.
       * **preferred phase** = angle of that mean vector (radians; 0 = SO up-state
         peak), reported as cos/sin so it is usable as a linear model feature.
       * **modulation** = Rayleigh z (= n * MVL^2), the non-uniformity test statistic.

Reported for N2, N3, and pooled NREM. Everything is JSON-serialisable and degrades
to None rather than raising. Single central channel suffices (no multi-channel
decode), so this is a moderate-effort single-channel-EEG feature.
"""
import warnings

import numpy as np

import eeg_spectral as es

EPOCH_SEC = es.EPOCH_SEC
POOL_TAG = es.POOL_TAG

# Slow-oscillation band. The classic SO is ~0.5-1.25 Hz (Molle/Staresina); we use
# a slightly wider low edge to capture the full up-state without leaking into the
# 0.1-0.3 Hz drift that the AASM slow-wave definition excludes.
SO_BAND = (0.5, 1.25)
MIN_SPINDLES_FOR_COUPLING = 8          # below this the circular stats are too noisy
MIN_NREM_MIN = 5.0                     # need enough NREM to bother


def _finite(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _r(x, nd=4):
    v = _finite(x)
    return None if v is None else round(v, nd)


def _so_phase(x, fs):
    """Instantaneous slow-oscillation phase (radians) of `x`, via the same
    Butterworth->filtfilt->Hilbert idiom eeg_spectral uses for the spindle
    envelope — but taking the ANGLE of the analytic signal, not its amplitude.
    Phase 0 = SO positive peak (up-state)."""
    lo, hi = SO_BAND
    ny = 0.5 * fs
    hi = min(hi, ny * 0.99)
    if lo >= hi:
        return None
    b, a = es._sp_signal.butter(4, [lo / ny, hi / ny], btype="band")
    filt = es._sp_signal.filtfilt(b, a, x)
    return np.angle(es._sp_signal.hilbert(filt))


def _circ_stats(phases):
    """Circular summary of a set of angles (radians): mean resultant vector length
    (MVL in [0,1]), preferred angle, and Rayleigh z = n*MVL^2."""
    phases = np.asarray(phases, float)
    n = phases.size
    if n == 0:
        return None
    z = np.exp(1j * phases)
    mean_vec = z.mean()
    mvl = float(np.abs(mean_vec))
    ang = float(np.angle(mean_vec))
    return {"n": int(n), "mvl": mvl, "angle": ang, "rayleigh_z": float(n * mvl * mvl)}


def so_spindle_coupling(eeg, fs, stage_codes, k=es.SPINDLE_K):
    """SO-spindle coupling per NREM stage + pooled NREM from one EEG channel.

    Returns {"ok", "stages": {tag: {coupling_strength, preferred_phase_cos/sin,
    modulation, n_spindles}}}. Reuses eeg_spectral's spindle envelope + the same
    mean+k*SD threshold calibrated on N2+N3, so the N2 and N3 spindle sets match
    `spindle_features_by_stage` exactly; the `nrem` pool additionally includes any
    N1 spindles (that detector only scores N2/N3).
    """
    eeg = np.asarray(eeg, float)
    fs = float(fs)
    codes = np.rint(np.asarray(stage_codes, float)).astype(int)
    if not es._SCIPY_OK or fs <= 0 or eeg.size == 0 or codes.size == 0:
        return {"ok": False, "error": "no usable EEG / staging / scipy"}
    if fs < 2 * es.SPINDLE_BAND[1]:
        return {"ok": False, "error": "sampling rate too low for the spindle band"}
    spe, n_ep, codes = es._align_epochs(eeg.size, fs, codes)
    if spe is None or n_ep < 1:
        return {"ok": False, "error": "signal shorter than one epoch"}

    x = eeg[:n_ep * spe].astype(float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        sig_env = es._bandpass_env(x, fs, es.SPINDLE_BAND)     # sigma amplitude envelope
        so_phase = _so_phase(x, fs)                            # SO instantaneous phase
    except Exception as e:
        return {"ok": False, "error": f"filter failed: {type(e).__name__}"}
    if sig_env is None or so_phase is None:
        return {"ok": False, "error": "a band collapses at this sampling rate"}

    # per-recording spindle threshold, calibrated identically to
    # eeg_spectral.spindle_features_by_stage: mean + k*SD of the sigma envelope over
    # the pooled N2+N3 (codes 2,1) samples — NOT including N1 — so the spindles we
    # detect here are exactly the ones that detector reports.
    sample_stage = np.repeat(codes[:n_ep], spe)[:sig_env.size]
    calib_mask = np.isin(sample_stage, (2, 1))
    if calib_mask.sum() < fs * 60:
        return {"ok": False, "error": "insufficient N2+N3 to calibrate spindle threshold"}
    base = sig_env[calib_mask]
    thr = float(base.mean() + k * base.std())
    min_len = int(es.SPINDLE_MIN_S * fs)
    max_len = int(es.SPINDLE_MAX_S * fs)
    merge_gap = int(es.SPINDLE_MERGE_GAP_S * fs)

    def _phases_for_stage(stage_codes_wanted):
        mask = np.isin(sample_stage, stage_codes_wanted)
        above = (sig_env > thr) & mask
        bursts = es._detect_bursts(above, min_len, max_len, merge_gap=merge_gap)
        phases = []
        for a, b in bursts:
            peak = a + int(np.argmax(sig_env[a:b]))            # spindle amplitude peak
            phases.append(so_phase[peak])
        return phases

    def _summ(phases):
        cs = _circ_stats(phases)
        if cs is None or cs["n"] < MIN_SPINDLES_FOR_COUPLING:
            return None
        return {
            "coupling_strength": _r(cs["mvl"]),
            "preferred_phase_cos": _r(np.cos(cs["angle"])),
            "preferred_phase_sin": _r(np.sin(cs["angle"])),
            "modulation": _r(cs["rayleigh_z"], 3),
            "n_spindles": cs["n"],
        }

    stages = {}
    for code in (2, 1):                                        # N2, N3
        minutes = int(np.count_nonzero(codes[:n_ep] == code)) * EPOCH_SEC / 60.0
        if minutes < MIN_NREM_MIN:
            continue
        s = _summ(_phases_for_stage((code,)))
        if s is not None:
            stages[POOL_TAG[code]] = s
    s = _summ(_phases_for_stage(es.NREM_CODES))
    if s is not None:
        stages["nrem"] = s

    return {"ok": True, "so_band": list(SO_BAND), "stages": stages}


# ------------------------------------------------------------------ flat schema
_COUPLING_FEATS = ["coupling_strength", "preferred_phase_cos", "preferred_phase_sin",
                   "modulation", "n_spindles"]
_COUPLING_TAGS = ["n2", "n3", "nrem"]


def flatten_coupling(out, row=None):
    """Flatten so_spindle_coupling output into {col: value}; stable keys."""
    row = {} if row is None else row
    stages = out.get("stages", {}) if out.get("ok") else {}
    for tag in _COUPLING_TAGS:
        s = stages.get(tag) or {}
        for feat in _COUPLING_FEATS:
            row[f"couple_{tag}_{feat}"] = s.get(feat)
    return row


def coupling_columns():
    return [f"couple_{tag}_{feat}" for tag in _COUPLING_TAGS for feat in _COUPLING_FEATS]
