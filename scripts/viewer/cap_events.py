"""Cyclic Alternating Pattern (CAP / A2) — an approximate automatic detector.

CAP is the EEG marker of **NREM sleep instability**: cyclic sequences of transient
activations (**A-phases**) standing out from the background (**B-phases**), each
phase 2-60 s, arranged in CAP cycles (A+B). The **CAP rate** (fraction of NREM
spent in CAP sequences) rises with age and is elevated in cognitive impairment and
poor sleep quality, and the **A-phase subtype balance** (A1 = synchronized slow
EEG, sleep-promoting; A3 = desynchronized fast/arousal-like) shifts toward A2/A3
with pathology. Because CAP is inherently a *dynamic* instability measure, it is on
the same axis our error analysis says the model is weak on.

⚠️ **This is a documented approximation, not clinical Terzano scoring.** Faithful
CAP scoring is a visual/expert task; robust automatic detectors (Barcaro, Ferri,
Mariani) use multi-band descriptor banks and validated thresholds. Here we use a
transparent, reproducible surrogate suitable as a *model feature*:

  1. Per non-overlapping 2 s window, band powers (delta/theta/alpha/sigma/beta) via
     rFFT (reusing eeg_spectral's band table). Total 0.5-30 Hz power = the window
     "activation" amplitude A(t).
  2. A-phase = NREM windows whose activation exceeds a LOCAL moving-median
     background by ≥ `ACT_FRAC` (Terzano's rule is a ≥1/3 amplitude rise over
     background; we use a noise-robust factor). Adjacent A-windows merge; A-phase
     duration kept to 2-60 s.
  3. Subtype from the A-phase spectral content: fast fraction
     F/(S+F) with S=delta+theta, F=alpha+sigma+beta → A1 (<`A1_MAX`), A2, A3
     (>`A3_MIN`). An A-phase overlapping a CAISR arousal is forced to A3 (arousals
     are A3 by definition).
  4. CAP sequence = ≥2 A-phases separated by B-phases (inter-A gaps) of 2-60 s.
     CAP rate = total CAP-sequence time / total NREM time.

Single central channel; reuses eeg_spectral's alignment/band constants. Optional
arousal stream (arousal_caisr) improves A3 detection. JSON-serialisable, degrades
to None rather than raising.
"""
import numpy as np

import eeg_spectral as es

EPOCH_SEC = es.EPOCH_SEC
NREM_CODES = (3, 2, 1)
BANDS = es.BANDS                          # (name, lo, hi) — reuse the exact band table
_TOTAL = es._TOTAL_BAND

WIN_S = 2.0                               # CAP descriptor window (Terzano phases are 2-60 s)
BASE_WIN_S = 100.0                        # moving-background window (spans a B-phase scale)
# A-phase activation must exceed background by this fraction. Terzano's visual rule
# is a ~1/3 rise; 0.5 proved too permissive on real data (CAP rate ~0.78, above the
# ~0.3-0.5 adult norm), so 0.75 is used — it lands the cohort CAP rate in the
# literature range while still detecting ample A-phases (calibrated on the box).
ACT_FRAC = 0.75
A_MIN_S, A_MAX_S = 2.0, 60.0             # A-phase duration bounds
B_MIN_S, B_MAX_S = 2.0, 60.0             # B-phase (inter-A gap) bounds for a CAP cycle
A1_MAX, A3_MIN = 0.30, 0.50              # fast-fraction cut points for A1 / A3
MIN_NREM_MIN = 10.0                       # need enough NREM for a CAP rate
MIN_A_PER_SEQ = 2                         # A-phases needed to call a CAP sequence


def _finite(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _r(x, nd=4):
    v = _finite(x)
    return None if v is None else round(v, nd)


def _moving_median(x, win):
    """Edge-padded centered moving median (vectorised via sliding windows)."""
    win = max(1, int(win) | 1)                       # force odd for symmetry
    if x.size <= win:
        return np.full_like(x, np.median(x) if x.size else 0.0)
    pad = win // 2
    xp = np.pad(x, pad, mode="edge")
    sw = np.lib.stride_tricks.sliding_window_view(xp, win)
    return np.median(sw, axis=1)


def _window_bandpowers(eeg, fs, wlen):
    """Band powers per non-overlapping `wlen`-sample window via rFFT.
    Returns (n_win, dict{band->power[n_win]}, total_power[n_win])."""
    n_win = eeg.size // wlen
    if n_win < 1:
        return 0, {}, None
    seg = eeg[:n_win * wlen].reshape(n_win, wlen).astype(float)
    seg = seg - seg.mean(axis=1, keepdims=True)
    freqs = np.fft.rfftfreq(wlen, d=1.0 / fs)
    psd = np.abs(np.fft.rfft(seg, axis=1)) ** 2       # (n_win, n_freq)
    band_pow = {}
    for nm, lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        band_pow[nm] = psd[:, m].sum(axis=1) if m.any() else np.zeros(n_win)
    tm = (freqs >= _TOTAL[0]) & (freqs < _TOTAL[1])
    total = psd[:, tm].sum(axis=1)
    return n_win, band_pow, total


def _arousal_seconds(arousal_codes, arousal_fs, n_sec):
    """Boolean per-second arousal mask (code 1) resampled to a 1 s axis, or None."""
    if arousal_codes is None or arousal_fs is None or arousal_fs <= 0:
        return None
    a = np.rint(np.asarray(arousal_codes, float)).astype(int)
    if a.size == 0:
        return None
    step = a.size / float(n_sec) if n_sec else 1.0
    idx = np.clip((np.arange(n_sec) * step).astype(int), 0, a.size - 1)
    return a[idx] == 1


def cap_features(eeg, fs, stage_codes, arousal_codes=None, arousal_fs=None):
    """Approximate CAP metrics from one EEG channel (+ optional arousal stream).

    Returns {"ok", "nrem_min", plus the flat CAP metrics}.
    """
    eeg = np.asarray(eeg, float)
    fs = float(fs)
    codes = np.rint(np.asarray(stage_codes, float)).astype(int)
    if not es._SCIPY_OK or fs <= 0 or eeg.size == 0 or codes.size == 0:
        return {"ok": False, "error": "no usable EEG / staging / scipy"}
    if fs < 2 * _TOTAL[1]:
        return {"ok": False, "error": "sampling rate too low for the band range"}
    spe, n_ep, codes = es._align_epochs(eeg.size, fs, codes)
    if spe is None or n_ep < 1:
        return {"ok": False, "error": "signal shorter than one epoch"}

    nrem_min = int(np.count_nonzero(np.isin(codes[:n_ep], NREM_CODES))) * EPOCH_SEC / 60.0
    if nrem_min < MIN_NREM_MIN:
        return {"ok": False, "error": f"only {nrem_min:.1f} min NREM"}

    x = np.nan_to_num(eeg[:n_ep * spe], nan=0.0, posinf=0.0, neginf=0.0)
    wlen = max(1, int(round(WIN_S * fs)))
    n_win, band_pow, total = _window_bandpowers(x, fs, wlen)
    if n_win < 5 or total is None:
        return {"ok": False, "error": "too few analysis windows"}

    # stage per window (window i -> epoch (i*wlen)//spe)
    win_ep = np.clip((np.arange(n_win) * wlen) // spe, 0, n_ep - 1)
    win_stage = codes[:n_ep][win_ep]
    is_nrem = np.isin(win_stage, NREM_CODES)
    if is_nrem.sum() < 5:
        return {"ok": False, "error": "insufficient NREM windows"}

    # local background from NREM windows only; activation = A/background
    base_win = max(3, int(round(BASE_WIN_S / WIN_S)))
    amp = np.sqrt(np.maximum(total, 0.0))
    nrem_amp = amp.copy()
    nrem_amp[~is_nrem] = np.nan
    # moving median over the amp series, but ignore non-NREM by filling with median
    fill = np.nanmedian(nrem_amp) if np.isfinite(np.nanmedian(nrem_amp)) else 0.0
    amp_filled = np.where(is_nrem, amp, fill)
    background = _moving_median(amp_filled, base_win)
    background = np.where(background > 0, background, fill if fill > 0 else 1.0)
    activation = amp / background

    # A-phase candidate windows (NREM + activation above threshold)
    a_win = is_nrem & (activation > (1.0 + ACT_FRAC))
    a_min_w = max(1, int(round(A_MIN_S / WIN_S)))
    a_max_w = max(a_min_w, int(round(A_MAX_S / WIN_S)))
    bursts = es._detect_bursts(a_win, a_min_w, a_max_w, merge_gap=0)   # (start_win, end_win)
    if not bursts:
        return {"ok": True, "nrem_min": _r(nrem_min, 1), "cap_rate": 0.0,
                "cap_time_min": 0.0, "cap_a_index": 0.0,
                "cap_a1_pct": None, "cap_a2_pct": None, "cap_a3_pct": None,
                "cap_a1_index": 0.0, "cap_a3_index": 0.0,
                "cap_mean_a_dur_s": None, "cap_mean_b_dur_s": None, "cap_n_sequences": 0}

    # subtype each A-phase: fast fraction F/(S+F); arousal overlap -> A3
    arousal_sec = _arousal_seconds(arousal_codes, arousal_fs, n_ep * int(round(EPOCH_SEC)))
    slow_bp = band_pow["delta"] + band_pow["theta"]
    fast_bp = band_pow["alpha"] + band_pow["sigma"] + band_pow["beta"]
    a_phases = []                                   # (start_win, end_win, subtype)
    for a, b in bursts:
        S = float(slow_bp[a:b].sum()); F = float(fast_bp[a:b].sum())
        fast_frac = F / (S + F) if (S + F) > 0 else 0.5
        sub = 1 if fast_frac < A1_MAX else (3 if fast_frac > A3_MIN else 2)
        if arousal_sec is not None:                 # arousal overlap forces A3
            s0 = int(a * wlen / fs); s1 = int(b * wlen / fs)
            if s1 > s0 and s1 <= arousal_sec.size and arousal_sec[s0:s1].any():
                sub = 3
        a_phases.append((a, b, sub))

    # CAP sequences: runs of >=MIN_A_PER_SEQ A-phases with B-gaps in [B_MIN,B_MAX]
    b_min_w = max(1, int(round(B_MIN_S / WIN_S)))
    b_max_w = max(b_min_w, int(round(B_MAX_S / WIN_S)))
    cap_time_w = 0
    n_seq = 0
    b_durs = []
    run = [a_phases[0]]
    def _flush(run):
        nonlocal cap_time_w, n_seq
        if len(run) >= MIN_A_PER_SEQ:
            n_seq += 1
            cap_time_w += run[-1][1] - run[0][0]     # first A start .. last A end
    for prev, cur in zip(a_phases[:-1], a_phases[1:]):
        gap = cur[0] - prev[1]                        # B-phase length in windows
        if b_min_w <= gap <= b_max_w:
            run.append(cur); b_durs.append(gap * WIN_S)
        else:
            _flush(run); run = [cur]
    _flush(run)

    a_durs = [(b - a) * WIN_S for a, b, _ in a_phases]
    n_a = len(a_phases)
    n_a1 = sum(1 for *_, s in a_phases if s == 1)
    n_a2 = sum(1 for *_, s in a_phases if s == 2)
    n_a3 = sum(1 for *_, s in a_phases if s == 3)
    cap_time_min = cap_time_w * WIN_S / 60.0

    return {
        "ok": True,
        "nrem_min": _r(nrem_min, 1),
        "cap_rate": _r(cap_time_min / nrem_min) if nrem_min > 0 else None,   # flagship
        "cap_time_min": _r(cap_time_min, 2),
        "cap_a_index": _r(n_a / nrem_min, 3),                                # A-phases / min NREM
        "cap_a1_pct": _r(100.0 * n_a1 / n_a, 1) if n_a else None,
        "cap_a2_pct": _r(100.0 * n_a2 / n_a, 1) if n_a else None,
        "cap_a3_pct": _r(100.0 * n_a3 / n_a, 1) if n_a else None,
        "cap_a1_index": _r(n_a1 / nrem_min, 3),
        "cap_a3_index": _r(n_a3 / nrem_min, 3),
        "cap_mean_a_dur_s": _r(np.mean(a_durs), 2) if a_durs else None,
        "cap_mean_b_dur_s": _r(np.mean(b_durs), 2) if b_durs else None,
        "cap_n_sequences": int(n_seq),
    }


# ------------------------------------------------------------------ flat schema
_CAP_FEATS = ["cap_rate", "cap_time_min", "cap_a_index", "cap_a1_pct", "cap_a2_pct",
              "cap_a3_pct", "cap_a1_index", "cap_a3_index", "cap_mean_a_dur_s",
              "cap_mean_b_dur_s", "cap_n_sequences"]


def flatten_cap(out, row=None):
    row = {} if row is None else row
    ok = out.get("ok")
    for feat in _CAP_FEATS:
        row[feat] = out.get(feat) if ok else None
    return row


def cap_columns():
    return list(_CAP_FEATS)
