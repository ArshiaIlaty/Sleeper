"""Per-sleep-stage EEG spectral features + sleep-spindle detection.

Extends the relative-band-power panel in `stage_signals.py` with the quantities
clinicians actually reason about for cognitive decline:

  * **Absolute & relative band power** per stage — delta (0.5-4 Hz), theta (4-8),
    alpha (8-12), sigma (12-16, the spindle band), beta (16-30). Absolute power
    is integrated PSD in uV^2; relative is the fraction of 0.5-30 Hz power.
  * **Band ratios** — Theta/Alpha and Delta/Sigma per stage. Ratios are
    scale-invariant (the total-power denominator cancels), so they are robust to
    the amplitude/impedance differences between sites that plague absolute power.
  * **REM-slowing metric** — (delta + theta) / (alpha + sigma + beta), the
    slow-to-fast power ratio. Computed per stage; its value *in REM* is a
    well-known marker of EEG slowing in neurodegeneration (higher = more slowing).
  * **Sleep spindles** — 11-16 Hz bursts detected on the EEG envelope; density
    (spindles/min), mean amplitude and duration, reported for N2 (primary) and N3.
    Reduced spindle density tracks memory consolidation deficits and cognitive
    impairment.

Cost control (measured on the box, no numba): band power is per-epoch Welch PSD
(~1 ms/epoch), capped at `MAX_BP_EPOCHS` epochs/stage sampled evenly. Spindle
detection filters the whole channel once (Butterworth + Hilbert envelope, an
O(n log n) FFT — a night at 200 Hz is ~1-2 s) then thresholds within each stage's
bouts, so the whole module runs in a few seconds regardless of montage.

Everything is JSON-serialisable and degrades to None rather than raising, so it is
safe to run unattended across a cohort. Staging is aligned to the signal by epoch
index (epoch i = samples [i*spe:(i+1)*spe], spe = round(fs*30)); the shorter of the
two lengths wins.
"""
import warnings

import numpy as np

try:
    from scipy import signal as _sp_signal
    _SCIPY_OK = True
except Exception:                              # pragma: no cover
    _sp_signal = None
    _SCIPY_OK = False

EPOCH_SEC = 30.0
STAGE_LABEL = {5: "Wake", 3: "N1", 2: "N2", 1: "N3", 4: "REM"}
STAGE_ORDER = [5, 3, 2, 1, 4]
POOL_TAG = {5: "wake", 3: "n1", 2: "n2", 1: "n3", 4: "rem"}
NREM_CODES = (3, 2, 1)

# EEG bands (Hz). sigma = spindle band.
BANDS = [("delta", 0.5, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 12.0),
         ("sigma", 12.0, 16.0), ("beta", 16.0, 30.0)]
_TOTAL_BAND = (0.5, 30.0)
MAX_BP_EPOCHS = 180                            # band-power epochs/stage (evenly sampled)

# Spindle detection (11-16 Hz sigma bursts), following the Molle/Klinzing family
# of envelope detectors: bandpass -> analytic amplitude -> smooth -> threshold ->
# merge near-adjacent crossings -> keep 0.5-3 s events.
SPINDLE_BAND = (11.0, 16.0)
SPINDLE_MIN_S, SPINDLE_MAX_S = 0.5, 3.0        # plausible spindle duration (AASM/literature)
SPINDLE_SMOOTH_S = 0.1                         # envelope moving-average window
SPINDLE_MERGE_GAP_S = 0.3                      # merge crossings separated by < this
SPINDLE_K = 2.5                                # threshold = mean + K*SD of the smoothed envelope
MIN_STAGE_MIN_FOR_SPINDLE = 2.0                # need >~2 min of a stage for a density estimate


# --------------------------------------------------------------------------- utils
def _finite(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _r(x, nd=3):
    v = _finite(x)
    return None if v is None else round(v, nd)


def _align_epochs(n_samples, fs, codes):
    spe = int(round(fs * EPOCH_SEC))
    if spe < 1:
        return None, 0, codes
    n_ep = min(len(codes), n_samples // spe)
    return spe, n_ep, codes[:n_ep]


def _ratio(a, b):
    """a/b as a rounded float, or None if either is missing / b<=0."""
    a, b = _finite(a), _finite(b)
    if a is None or b is None or b <= 0:
        return None
    return _r(a / b, 4)


# --------------------------------------------------------------------------- band power
def _epoch_band_powers(seg, fs):
    """(abs_dict uV^2, rel_dict fraction) for one epoch via Welch PSD, or (None,None)."""
    seg = seg[np.isfinite(seg)]
    if seg.size < int(fs * 2) or fs < 2 * _TOTAL_BAND[1]:
        return None, None
    seg = seg - seg.mean()
    nper = int(min(seg.size, max(fs * 2, 256)))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f, pxx = _sp_signal.welch(seg, fs=fs, nperseg=nper)
    except Exception:
        return None, None
    tot_mask = (f >= _TOTAL_BAND[0]) & (f < _TOTAL_BAND[1])
    total = float(np.trapz(pxx[tot_mask], f[tot_mask]))
    if not np.isfinite(total) or total <= 0:
        return None, None
    abs_bp, rel_bp = {}, {}
    for nm, lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        p = float(np.trapz(pxx[m], f[m])) if np.any(m) else 0.0
        abs_bp[nm] = p
        rel_bp[nm] = p / total
    return abs_bp, rel_bp


def _stage_spectral(abs_list, rel_list):
    """Summarise per-epoch band powers for one stage into a feature dict."""
    if not abs_list:
        return None
    absm = {nm: float(np.mean([d[nm] for d in abs_list])) for nm, _, _ in BANDS}
    relm = {nm: float(np.mean([d[nm] for d in rel_list])) for nm, _, _ in BANDS}
    fast = absm["alpha"] + absm["sigma"] + absm["beta"]
    slow = absm["delta"] + absm["theta"]
    out = {"n_epochs": len(abs_list)}
    for nm, _, _ in BANDS:
        out[f"abs_{nm}"] = _r(absm[nm], 3)
        out[f"rel_{nm}"] = _r(relm[nm], 4)
    out["theta_alpha"] = _ratio(absm["theta"], absm["alpha"])
    out["delta_sigma"] = _ratio(absm["delta"], absm["sigma"])
    out["rem_slowing"] = _ratio(slow, fast)     # (delta+theta)/(alpha+sigma+beta)
    return out


def eeg_bandpower_by_stage(eeg, fs, stage_codes, max_epochs=MAX_BP_EPOCHS):
    """Per-stage absolute/relative band power + Theta/Alpha, Delta/Sigma, REM-slowing.

    Returns {"ok", "stages": {tag: {...}}, "contrasts": {...}}. The flagship
    contrasts are `rem_slowing` in REM (EEG slowing marker) and the N3 slow-wave
    (delta) power fraction.
    """
    eeg = np.asarray(eeg, float)
    fs = float(fs)
    codes = np.rint(np.asarray(stage_codes, float)).astype(int)
    if not _SCIPY_OK or fs <= 0 or eeg.size == 0 or codes.size == 0:
        return {"ok": False, "error": "no usable EEG / staging / scipy"}
    spe, n_ep, codes = _align_epochs(eeg.size, fs, codes)
    if spe is None or n_ep < 1 or fs < 2 * _TOTAL_BAND[1]:
        return {"ok": False, "error": "signal too short or sampling rate too low"}

    stages = {}
    for code in STAGE_ORDER:
        idx = np.where(codes == code)[0]
        if idx.size == 0:
            continue
        if idx.size > max_epochs:
            idx = idx[np.linspace(0, idx.size - 1, max_epochs).astype(int)]
        abs_list, rel_list = [], []
        for e in idx:
            a, r = _epoch_band_powers(eeg[e * spe:(e + 1) * spe], fs)
            if a is not None:
                abs_list.append(a); rel_list.append(r)
        s = _stage_spectral(abs_list, rel_list)
        if s is not None:
            stages[POOL_TAG[code]] = s

    def g(tag, feat):
        return (stages.get(tag) or {}).get(feat)

    contrasts = {}
    contrasts["rem_slowing"] = g("rem", "rem_slowing")             # Tier-1: REM EEG slowing
    contrasts["n3_delta_rel"] = g("n3", "rel_delta")               # Tier-1: N3 slow-wave activity
    contrasts["n3_delta_abs"] = g("n3", "abs_delta")
    # how much the slow/fast ratio changes REM vs NREM deep sleep (blunted -> ~1)
    contrasts["rem_n3_slowing_ratio"] = _ratio(g("rem", "rem_slowing"), g("n3", "rem_slowing"))
    return {"ok": True, "stages": stages, "contrasts": contrasts}


# --------------------------------------------------------------------------- spindles
def _bandpass_env(x, fs, band, smooth_s=SPINDLE_SMOOTH_S):
    """Zero-phase Butterworth bandpass + Hilbert amplitude envelope of `x`,
    smoothed with a short moving average so the threshold crossings are spindle-
    scale bursts rather than per-sample noise."""
    lo, hi = band
    ny = 0.5 * fs
    hi = min(hi, ny * 0.99)
    if lo >= hi:
        return None
    b, a = _sp_signal.butter(4, [lo / ny, hi / ny], btype="band")
    filt = _sp_signal.filtfilt(b, a, x)
    env = np.abs(_sp_signal.hilbert(filt))
    w = max(1, int(round(smooth_s * fs)))
    if w > 1:
        kern = np.ones(w) / w
        env = np.convolve(env, kern, mode="same")
    return env


def _detect_bursts(env_mask, min_len, max_len, merge_gap=0):
    """Contiguous True runs in `env_mask`, first merging runs separated by a gap
    of < `merge_gap` samples, then keeping runs with length in [min_len, max_len].
    Returns a list of (start, end_exclusive)."""
    # raw runs
    runs = []
    n = env_mask.size
    i = 0
    while i < n:
        if env_mask[i]:
            j = i + 1
            while j < n and env_mask[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    # merge runs whose inter-run gap is small (a spindle can dip momentarily)
    if merge_gap > 0 and runs:
        merged = [runs[0]]
        for a, b in runs[1:]:
            if a - merged[-1][1] < merge_gap:
                merged[-1][1] = b
            else:
                merged.append([a, b])
        runs = merged
    return [(a, b) for a, b in runs if min_len <= (b - a) <= max_len]


def spindle_features_by_stage(eeg, fs, stage_codes, k=SPINDLE_K):
    """Sleep-spindle density / amplitude / duration in N2 and N3.

    Sigma-band (11-16 Hz) envelope thresholded at mean+k*SD over the pooled NREM
    (N2+N3) envelope; a spindle is a supra-threshold run of 0.4-2.5 s. Density is
    spindles per minute of that stage. Reduced N2 spindle density is an
    exploratory marker of impaired memory consolidation / cognitive decline.
    """
    eeg = np.asarray(eeg, float)
    fs = float(fs)
    codes = np.rint(np.asarray(stage_codes, float)).astype(int)
    if not _SCIPY_OK or fs <= 0 or eeg.size == 0 or codes.size == 0:
        return {"ok": False, "error": "no usable EEG / staging / scipy"}
    if fs < 2 * SPINDLE_BAND[1]:
        return {"ok": False, "error": "sampling rate too low for the spindle band"}
    spe, n_ep, codes = _align_epochs(eeg.size, fs, codes)
    if spe is None or n_ep < 1:
        return {"ok": False, "error": "signal shorter than one epoch"}

    x = eeg[:n_ep * spe].astype(float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        env = _bandpass_env(x, fs, SPINDLE_BAND)
    except Exception as e:
        return {"ok": False, "error": f"spindle filter failed: {type(e).__name__}"}
    if env is None:
        return {"ok": False, "error": "spindle band collapses at this sampling rate"}

    # per-recording amplitude threshold from the pooled NREM (N2+N3) envelope
    sample_stage = np.repeat(codes[:n_ep], spe)[:env.size]
    nrem_mask = np.isin(sample_stage, (2, 1))
    if nrem_mask.sum() < fs * 60:              # <1 min of NREM -> can't calibrate
        return {"ok": False, "error": "insufficient NREM to calibrate spindle threshold"}
    base = env[nrem_mask]
    thr = float(base.mean() + k * base.std())
    min_len = int(SPINDLE_MIN_S * fs)
    max_len = int(SPINDLE_MAX_S * fs)
    merge_gap = int(SPINDLE_MERGE_GAP_S * fs)

    stages = {}
    for code in (2, 1):                        # N2 (primary), N3
        idx = np.where(codes[:n_ep] == code)[0]
        minutes = idx.size * EPOCH_SEC / 60.0
        if minutes < MIN_STAGE_MIN_FOR_SPINDLE:
            continue
        stage_mask = np.isin(sample_stage, (code,))
        # detect only within contiguous runs of this stage (no cross-bout bursts)
        above = (env > thr) & stage_mask
        bursts = _detect_bursts(above, min_len, max_len, merge_gap=merge_gap)
        n = len(bursts)
        amps = [float(env[a:b].max()) for a, b in bursts] if bursts else []
        durs = [(b - a) / fs for a, b in bursts] if bursts else []
        stages[POOL_TAG[code]] = {
            "minutes": _r(minutes, 1),
            "n_spindles": int(n),
            "density_per_min": _r(n / minutes, 3) if minutes > 0 else None,
            "amp_mean": _r(np.mean(amps), 2) if amps else None,
            "dur_mean": _r(np.mean(durs), 3) if durs else None,
        }
    return {"ok": True, "threshold_uv": _r(thr, 2), "k": k, "stages": stages}
