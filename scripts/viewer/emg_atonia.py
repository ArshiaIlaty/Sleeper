"""REM sleep without atonia (RSWA / E2) from the chin (submental) EMG channel.

Normal REM sleep is defined partly by **muscle atonia** — the chin EMG drops to
its lowest tone of the night. Loss of that atonia (**RSWA** — elevated tonic tone
and/or frequent phasic twitches during REM) is the core polysomnographic feature
of REM behavior disorder and one of the strongest early markers of prodromal
**α-synuclein neurodegeneration** (Parkinson's, DLB, MSA), which carries cognitive
decline. So RSWA is mechanistically on-target for a cognitive-impairment model.

Montage heterogeneity is the central difficulty: chin-EMG amplitude is not
comparable across sites (different electrodes, gain, referencing). Absolute µV is
therefore useless as a cross-site feature. Instead every metric is **normalised to
the recording's own atonia reference** — the low-percentile EMG level, which is the
patient's own physiological "floor":

  * **tonic tone ratio** — median REM EMG RMS / the recording's atonia floor. ~1 in
    a normally atonic REM; elevated in tonic RSWA.
  * **REM / NREM tone ratio** — REM EMG RMS / NREM EMG RMS. Healthy REM is the
    *quietest* stage (ratio < 1); RSWA pushes it toward / above 1.
  * **phasic burst density** — short (0.1-5 s) suprathreshold EMG bursts per minute
    of REM, threshold set from the atonia floor (a self-scaled version of the
    AASM/SINBAR phasic-activity metric). Elevated in phasic RSWA.
  * **tonic fraction** — fraction of REM analysis mini-epochs whose RMS exceeds the
    atonia floor by a set factor (a self-scaled tonic-density surrogate).

RMS is computed on short (0.5 s) windows of the rectified chin EMG. Reuses
eeg_spectral's burst detector so the phasic detection matches the spindle idiom.
Single channel, JSON-serialisable, degrades to None rather than raising.
"""
import numpy as np

import eeg_spectral as es

EPOCH_SEC = es.EPOCH_SEC
REM_CODE = 4
NREM_CODES = (3, 2, 1)

RMS_WIN_S = 0.5                    # mini-epoch for the RMS envelope
ATONIA_PCTILE = 20.0              # the recording's EMG "floor" = this percentile of REM RMS
TONIC_FACTOR = 2.0               # a mini-epoch is "tonic" if RMS > TONIC_FACTOR * floor
PHASIC_FACTOR = 4.0              # phasic burst threshold = PHASIC_FACTOR * floor
PHASIC_MIN_S, PHASIC_MAX_S = 0.1, 5.0     # SINBAR-style phasic duration bounds
MIN_REM_MIN = 5.0                # need enough REM to characterise atonia


def _finite(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _r(x, nd=4):
    v = _finite(x)
    return None if v is None else round(v, nd)


def _rms_envelope(x, fs, win_s=RMS_WIN_S):
    """Non-overlapping-window RMS of a (mean-removed, rectified) signal.
    Returns (rms_per_window, samples_per_window)."""
    x = np.nan_to_num(np.asarray(x, float), nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean()
    w = max(1, int(round(win_s * fs)))
    n_win = x.size // w
    if n_win < 1:
        return np.array([]), w
    seg = x[:n_win * w].reshape(n_win, w)
    rms = np.sqrt(np.mean(seg * seg, axis=1))
    return rms, w


def rswa_features(emg, fs, stage_codes):
    """RSWA / atonia metrics from one chin-EMG channel + per-epoch stage codes.

    Returns {"ok", "rem_min", "atonia_floor", plus the flat metrics}. All tone
    metrics are ratios to the recording's own atonia floor, so they are comparable
    across the site-heterogeneous montage.
    """
    emg = np.asarray(emg, float)
    fs = float(fs)
    codes = np.rint(np.asarray(stage_codes, float)).astype(int)
    if fs <= 0 or emg.size == 0 or codes.size == 0:
        return {"ok": False, "error": "no usable EMG / staging"}
    spe, n_ep, codes = es._align_epochs(emg.size, fs, codes)
    if spe is None or n_ep < 1:
        return {"ok": False, "error": "signal shorter than one epoch"}

    rem_min = int(np.count_nonzero(codes[:n_ep] == REM_CODE)) * EPOCH_SEC / 60.0
    if rem_min < MIN_REM_MIN:
        return {"ok": False, "error": f"only {rem_min:.1f} min REM"}

    x = emg[:n_ep * spe]
    rms, w = _rms_envelope(x, fs)
    if rms.size < 4:
        return {"ok": False, "error": "EMG too short for RMS envelope"}
    # map each RMS window to the stage of its first sample (window i = samples [i*w:(i+1)*w])
    win_idx = np.clip((np.arange(rms.size) * w) // spe, 0, n_ep - 1)
    win_stage = codes[:n_ep][win_idx]

    rem_rms = rms[win_stage == REM_CODE]
    nrem_rms = rms[np.isin(win_stage, NREM_CODES)]
    if rem_rms.size < 4:
        return {"ok": False, "error": "insufficient REM RMS windows"}

    # atonia floor: the patient's own quietest REM tone (robust low percentile)
    floor = float(np.percentile(rem_rms, ATONIA_PCTILE))
    if not np.isfinite(floor) or floor <= 0:
        floor = float(np.percentile(rem_rms[rem_rms > 0], ATONIA_PCTILE)) if np.any(rem_rms > 0) else 0.0
    if floor <= 0:
        return {"ok": False, "error": "degenerate atonia floor (flat EMG)"}

    rem_med = float(np.median(rem_rms))
    nrem_med = float(np.median(nrem_rms)) if nrem_rms.size else None

    # tonic fraction: REM windows exceeding TONIC_FACTOR * floor
    tonic_frac = float(np.mean(rem_rms > TONIC_FACTOR * floor))

    # phasic bursts: contiguous REM windows above PHASIC_FACTOR*floor, 0.1-5 s.
    # Work in RMS-window units; reuse the burst detector, converting durations to
    # window counts (win length = RMS_WIN_S).
    rem_mask_full = (win_stage == REM_CODE)
    above = (rms > PHASIC_FACTOR * floor) & rem_mask_full
    min_len = max(1, int(round(PHASIC_MIN_S / RMS_WIN_S)))
    max_len = max(min_len, int(round(PHASIC_MAX_S / RMS_WIN_S)))
    bursts = es._detect_bursts(above, min_len, max_len, merge_gap=0)
    phasic_per_min = len(bursts) / rem_min if rem_min > 0 else None

    out = {
        "ok": True,
        "rem_min": _r(rem_min, 1),
        "atonia_floor_uv": _r(floor, 3),
        "rswa_tonic_tone_ratio": _r(rem_med / floor),          # >1 = elevated tonic REM tone
        "rswa_rem_nrem_ratio": _r(rem_med / nrem_med) if nrem_med else None,  # >1 = REM not quietest
        "rswa_tonic_fraction": _r(tonic_frac),                 # fraction of REM windows tonic
        "rswa_phasic_per_min": _r(phasic_per_min, 3),          # phasic bursts / min REM
        "rswa_rem_rms_cv": _r(float(np.std(rem_rms) / rem_med)) if rem_med > 0 else None,  # burstiness
    }
    return out


# ------------------------------------------------------------------ flat schema
_RSWA_FEATS = ["rswa_tonic_tone_ratio", "rswa_rem_nrem_ratio", "rswa_tonic_fraction",
               "rswa_phasic_per_min", "rswa_rem_rms_cv"]


def flatten_rswa(out, row=None):
    row = {} if row is None else row
    ok = out.get("ok")
    for feat in _RSWA_FEATS:
        row[feat] = out.get(feat) if ok else None
    return row


def rswa_columns():
    return list(_RSWA_FEATS)
