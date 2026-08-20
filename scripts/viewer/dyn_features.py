"""Group-3 "dynamics / distribution-shape" features (advanced-feature roadmap D1/D2/E1).

Three cheap refinements built entirely on streams we already decode — no new
signal decoding. Each captures the *temporal-distribution shape* of events we
already count, which is the axis our error analysis says the model is weak on
(clustering / variability / periodicity rather than night-level means):

  * **D2 desaturation clustering** (`desat_cluster_features`) — from SpO2 we
    already have ODI / hypoxic burden (counts + depth). This adds the *temporal
    arrangement* of desaturations: inter-desaturation-interval (IDI) distribution
    (median / CV), a dispersion (Fano) index, and the fraction of desaturations
    that fall in a tight cluster. Clustered desaturations reflect unstable
    ventilatory control; dispersed ones a different phenotype.
  * **D1 breath-interval distribution** (`breath_interval_features`) — we have
    per-stage respiratory rate mean/SD (`nk__rsp_*`). This adds the *shape* of the
    breath-to-breath interval series over sleep: skewness, kurtosis, and sample
    entropy (regularity/complexity of breathing), which night-mean rate misses.
  * **E1 limb periodicity** (`limb_periodicity_features`) — we have PLMI (count).
    This adds the inter-movement-interval (IMI) distribution, the fraction of IMIs
    in the periodic PLM band (5-90 s), and a periodicity-spectrum peak from the
    movement-onset train (how rhythmic the limb movements are).

Every function returns a JSON-serialisable dict with an "ok" flag and degrades to
{"ok": False} rather than raising. `*_columns()` + `flatten_*` give the fixed CSV
schema, matching the idiom of oxygenation / resp_events / nk_features.
"""
import warnings

import numpy as np

try:
    from scipy.stats import skew as _sp_skew, kurtosis as _sp_kurtosis
    _SCIPY_STATS_OK = True
except Exception:                                   # pragma: no cover
    _SCIPY_STATS_OK = False

import oxygenation as oxy

try:
    import neurokit2 as nk
    _NK_OK = True
except Exception:                                   # pragma: no cover
    _NK_OK = False

EPOCH_SEC = 30.0
SLEEP_CODES = (4, 3, 2, 1)                           # REM + N1/N2/N3 (exclude wake=5)

# ------------------------------------------------------------------ shared utils
def _finite(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _r(x, nd=4):
    v = _finite(x)
    return None if v is None else round(v, nd)


def _skew(a):
    """Bias-corrected sample skewness; None if <3 points or degenerate."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size < 3 or np.std(a) == 0:
        return None
    if _SCIPY_STATS_OK:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _finite(_sp_skew(a, bias=False))
    m = a.mean(); s = a.std(ddof=1)
    n = a.size
    return _finite((n / ((n - 1) * (n - 2))) * np.sum(((a - m) / s) ** 3))


def _kurtosis(a):
    """Bias-corrected excess kurtosis; None if <4 points or degenerate."""
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size < 4 or np.std(a) == 0:
        return None
    if _SCIPY_STATS_OK:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _finite(_sp_kurtosis(a, fisher=True, bias=False))
    m = a.mean(); s = a.std(ddof=1); n = a.size
    g2 = np.sum(((a - m) / s) ** 4) * n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))
    g2 -= 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return _finite(g2)


def _sample_entropy(series, m=2, r_frac=0.2):
    """Sample entropy of a 1-D series (m=2). r = r_frac * std. O(n^2) but the
    input here is a short inter-event series (<~1000), so it stays cheap. None if
    too short / degenerate."""
    x = np.asarray(series, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < m + 2:
        return None
    sd = x.std()
    if sd == 0:
        return None
    r = r_frac * sd

    def _phi(mm):
        # count template matches within Chebyshev distance r (excluding self)
        templates = np.array([x[i:i + mm] for i in range(n - mm + 1)])
        count = 0
        for i in range(len(templates)):
            d = np.max(np.abs(templates - templates[i]), axis=1)
            count += np.sum(d <= r) - 1                # exclude self-match
        return count

    B = _phi(m)
    A = _phi(m + 1)
    if B <= 0 or A <= 0:
        return None
    return _r(-np.log(A / B), 4)


def _fano(intervals):
    """Index of dispersion (Fano factor = var/mean) of an interval series — >1 is
    over-dispersed (clustered), ~1 Poisson, <1 regular."""
    a = np.asarray(intervals, float)
    a = a[np.isfinite(a) & (a > 0)]
    if a.size < 3 or a.mean() == 0:
        return None
    return _r(a.var() / a.mean(), 4)


def _onset_seconds_runs(mask, fs):
    """Onset time (s) of each contiguous True run in a boolean event mask."""
    runs = oxy._runs_true(np.asarray(mask, bool))
    return np.array([a / fs for a, _b in runs], float)


# ============================================================= D2 desat clustering
_DESAT_FEATS = ["desat_idi_median_s", "desat_idi_cv", "desat_fano",
                "desat_cluster_frac", "desat_n_clusters", "desat_isolated_frac",
                "desat_rate_per_h", "desat_burstiness"]
DESAT_CLUSTER_GAP_S = 40.0                          # two desats <= this apart are "clustered"


def desat_cluster_features(spo2, fs):
    """Temporal-clustering metrics of SpO2 desaturations (reuses oxygenation's
    validated desat detector for the event onsets)."""
    out = {"ok": False}
    x, fs, n_valid = oxy._normalise_spo2(spo2, fs)
    if x is None or fs <= 0 or n_valid < int(fs * 60):
        return out
    filled = oxy._ffill(x)
    if filled is None:
        return out
    win = max(1, int(oxy.BASELINE_WIN_S * fs))
    baseline = oxy._moving_median(filled, win)
    drop = baseline - filled
    ev_mask = np.isfinite(x) & (drop >= oxy.DESAT_DROP)
    runs = oxy._runs_true(ev_mask)
    min_len = int(oxy.MIN_EVENT_S * fs)

    onsets = []
    for a, b in runs:
        if (b - a) < min_len:
            continue
        seg = drop[a:b]
        seg = seg[np.isfinite(seg) & (seg > 0)]
        if seg.size == 0:
            continue
        onsets.append(a / fs)
    onsets = np.asarray(onsets, float)
    hours = x.size / fs / 3600.0
    n = onsets.size
    if n < 3:
        return out

    idi = np.diff(np.sort(onsets))                  # inter-desaturation intervals (s)
    idi = idi[idi > 0]
    if idi.size < 2:
        return out
    med = float(np.median(idi))
    cv = float(np.std(idi) / np.mean(idi)) if np.mean(idi) > 0 else None

    # clusters: chain desats separated by <= DESAT_CLUSTER_GAP_S
    clustered = idi <= DESAT_CLUSTER_GAP_S
    n_clusters = 0
    in_cluster = False
    clustered_events = 0
    for c in clustered:
        if c:
            if not in_cluster:
                n_clusters += 1
                clustered_events += 2               # this gap links two events
                in_cluster = True
            else:
                clustered_events += 1
        else:
            in_cluster = False
    clustered_events = min(clustered_events, n)

    out = {
        "ok": True,
        "desat_idi_median_s": _r(med, 2),
        "desat_idi_cv": _r(cv, 4),
        "desat_fano": _fano(idi),
        "desat_cluster_frac": _r(clustered_events / n, 4),
        "desat_n_clusters": int(n_clusters),
        "desat_isolated_frac": _r(1.0 - clustered_events / n, 4),
        "desat_rate_per_h": _r(n / hours, 3) if hours > 0 else None,
        "desat_burstiness": _r((cv - 1) / (cv + 1), 4) if cv is not None else None,  # Goh-Barabasi B
    }
    return out


def flatten_desat_cluster(out, row=None):
    row = {} if row is None else row
    ok = out.get("ok")
    for f in _DESAT_FEATS:
        row[f] = out.get(f) if ok else None
    return row


def desat_cluster_columns():
    return list(_DESAT_FEATS)


# ========================================================= D1 breath-interval dist
_BREATH_FEATS = ["breath_n", "breath_ibi_median_s", "breath_ibi_cv",
                 "breath_ibi_skew", "breath_ibi_kurtosis", "breath_ibi_sampen",
                 "breath_rate_mean"]
_MIN_BREATHS = 30


def _sleep_signal(sig, fs, stage_codes):
    """Concatenate the signal over sleep epochs (exclude wake), as rsp_rate_by_stage
    does per stage. Returns (concatenated_signal, n_sleep_epochs)."""
    codes = np.rint(np.asarray(stage_codes, float)).astype(int)
    spe = int(round(EPOCH_SEC * fs))
    if spe < 1:
        return None, 0
    n_ep = min(len(codes), sig.size // spe)
    if n_ep < 1:
        return None, 0
    idx = np.where(np.isin(codes[:n_ep], SLEEP_CODES))[0]
    if idx.size < 4:
        return None, 0
    seg = np.concatenate([sig[e * spe:(e + 1) * spe] for e in idx])
    return np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0), int(idx.size)


def breath_interval_features(rsp, fs, stage_codes):
    """Distribution shape of the breath-to-breath interval series over sleep."""
    out = {"ok": False}
    rsp = np.asarray(rsp, float)
    fs = float(fs)
    if not _NK_OK or fs <= 0 or rsp.size == 0:
        return out
    seg, n_ep = _sleep_signal(rsp, fs, stage_codes)
    if seg is None:
        return out
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clean = nk.rsp_clean(seg, sampling_rate=fs)
            _sig, info = nk.rsp_peaks(clean, sampling_rate=fs)
        peaks = np.asarray(info.get("RSP_Peaks", []), float)
    except Exception:
        return out
    peaks = peaks[np.isfinite(peaks)]
    if peaks.size < _MIN_BREATHS:
        return out
    ibi = np.diff(np.sort(peaks)) / fs              # inter-breath interval (s)
    ibi = ibi[(ibi > 1.0) & (ibi < 15.0)]            # physiologic 4-60 bpm
    if ibi.size < _MIN_BREATHS - 1:
        return out
    mean_ibi = float(np.mean(ibi))
    out = {
        "ok": True,
        "breath_n": int(ibi.size + 1),
        "breath_ibi_median_s": _r(np.median(ibi), 3),
        "breath_ibi_cv": _r(np.std(ibi) / mean_ibi, 4) if mean_ibi > 0 else None,
        "breath_ibi_skew": _skew(ibi),
        "breath_ibi_kurtosis": _kurtosis(ibi),
        "breath_ibi_sampen": _sample_entropy(ibi),
        "breath_rate_mean": _r(60.0 / mean_ibi, 2) if mean_ibi > 0 else None,
    }
    return out


def flatten_breath_interval(out, row=None):
    row = {} if row is None else row
    ok = out.get("ok")
    for f in _BREATH_FEATS:
        row[f] = out.get(f) if ok else None
    return row


def breath_interval_columns():
    return list(_BREATH_FEATS)


# ============================================================= E1 limb periodicity
_LIMB_FEATS = ["limb_n_movements", "limb_imi_median_s", "limb_imi_cv",
               "limb_periodic_frac", "limb_fano", "limb_spec_peak_hz",
               "limb_spec_peak_frac", "limb_movements_per_h"]
LIMB_CODE = 2                                       # PLM code in limb_caisr (matches clinical_report)
PLM_IMI_LO, PLM_IMI_HI = 5.0, 90.0                  # AASM periodic inter-movement-interval band


def limb_periodicity_features(limb_codes, limb_fs):
    """Inter-movement-interval distribution + periodicity spectrum of the limb-
    movement onset train from the 2 Hz limb_caisr stream."""
    out = {"ok": False}
    if limb_codes is None or limb_fs is None or limb_fs <= 0:
        return out
    codes = np.rint(np.asarray(limb_codes, float)).astype(int)
    if codes.size == 0:
        return out
    fs = float(limb_fs)
    onsets = _onset_seconds_runs(codes == LIMB_CODE, fs)
    hours = codes.size / fs / 3600.0
    n = onsets.size
    if n < 4:
        return out

    imi = np.diff(np.sort(onsets))
    imi = imi[imi > 0]
    if imi.size < 3:
        return out
    mean_imi = float(np.mean(imi))
    periodic = np.mean((imi >= PLM_IMI_LO) & (imi <= PLM_IMI_HI))

    # periodicity spectrum: bin the onset train at 1 Hz, take the normalised PSD,
    # report the peak in the PLM band (1/90 .. 1/5 Hz) and its share of band power.
    peak_hz, peak_frac = None, None
    n_sec = int(np.ceil(codes.size / fs))
    if n_sec >= 32:
        train = np.zeros(n_sec, float)
        oi = np.clip(onsets.astype(int), 0, n_sec - 1)
        train[oi] = 1.0
        train -= train.mean()
        freqs = np.fft.rfftfreq(n_sec, d=1.0)
        psd = np.abs(np.fft.rfft(train)) ** 2
        band = (freqs >= 1.0 / PLM_IMI_HI) & (freqs <= 1.0 / PLM_IMI_LO)
        if band.any() and psd[1:].sum() > 0:
            bpsd = psd[band]
            peak_hz = _r(float(freqs[band][int(np.argmax(bpsd))]), 5)
            peak_frac = _r(float(bpsd.max() / psd[1:].sum()), 4)

    out = {
        "ok": True,
        "limb_n_movements": int(n),
        "limb_imi_median_s": _r(np.median(imi), 2),
        "limb_imi_cv": _r(np.std(imi) / mean_imi, 4) if mean_imi > 0 else None,
        "limb_periodic_frac": _r(periodic, 4),
        "limb_fano": _fano(imi),
        "limb_spec_peak_hz": peak_hz,
        "limb_spec_peak_frac": peak_frac,
        "limb_movements_per_h": _r(n / hours, 3) if hours > 0 else None,
    }
    return out


def flatten_limb_periodicity(out, row=None):
    row = {} if row is None else row
    ok = out.get("ok")
    for f in _LIMB_FEATS:
        row[f] = out.get(f) if ok else None
    return row


def limb_periodicity_columns():
    return list(_LIMB_FEATS)


# ------------------------------------------------------------------ combined schema
def dyn_columns():
    return desat_cluster_columns() + breath_interval_columns() + limb_periodicity_columns()
