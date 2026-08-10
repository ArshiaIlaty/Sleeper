"""SpO2 oxygenation features, including hypoxic burden.

Reads one pulse-oximetry channel from the physiological EDF and derives the
oxygenation markers used in sleep-apnea and cognitive-risk work:

  * **mean / min SpO2**, and **T90** — the fraction (and minutes) of recording
    spent below 90% saturation.
  * **ODI** — oxygen-desaturation index: distinct desaturations of >= 3% from a
    moving baseline, per hour.
  * **Hypoxic burden** — the flagship measure: the total area between a running
    baseline saturation and the desaturation curve, i.e. sum over desaturations of
    (depth x duration). Reported as %-minutes per hour of recording. It captures
    both how deep and how long the dips are — a better cognitive/cardiovascular
    risk marker than event counts alone.
  * **Desaturation depth stats** — mean / max drop (%) per event.

Site-robustness: SpO2 is stored differently per site — some as a 0-100 percentage,
some as a 0-1 fraction (Emory), sometimes at odd sampling rates and with 0/255
dropout spikes. `_normalise_spo2` auto-detects the scale from the physiologic
median, rescales a 0-1 signal to percent, and masks non-physiologic samples
(<=40% or >100%), so the same code produces comparable numbers across all sites.

Pure numpy + scipy-optional; degrades to None rather than raising.
"""
import numpy as np

VALID_LO, VALID_HI = 40.0, 100.0              # physiologic SpO2 percent bounds
T90_LEVEL = 90.0
DESAT_DROP = 3.0                              # % drop that defines a desaturation event
BASELINE_WIN_S = 120.0                        # running-baseline window (moving max/median)
MIN_EVENT_S = 5.0                             # ignore blips shorter than this
MAX_EVENT_S = 120.0                           # cap a single event's contribution


def _finite(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _r(x, nd=3):
    v = _finite(x)
    return None if v is None else round(v, nd)


def _normalise_spo2(spo2, fs):
    """Return (clean_percent_array, fs, n_valid) or (None, fs, 0).

    Auto-detects a 0-1 fraction encoding (median <= ~1.5) and rescales to percent,
    then keeps only physiologic samples (40-100%). Non-finite and dropout samples
    (0, 255, etc.) are removed rather than interpolated, so gaps don't invent
    saturation values.
    """
    x = np.asarray(spo2, float)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return None, fs, 0
    med = float(np.median(finite[finite > 0])) if np.any(finite > 0) else 0.0
    if 0 < med <= 1.5:                        # stored as a 0-1 fraction -> percent
        x = x * 100.0
    x = np.where(np.isfinite(x) & (x > VALID_LO) & (x <= VALID_HI), x, np.nan)
    n_valid = int(np.count_nonzero(np.isfinite(x)))
    return x, float(fs), n_valid


def _runs_true(mask):
    """(start, end_exclusive) runs of True in a 1-D bool array."""
    if mask.size == 0:
        return []
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return [(int(idx[i]), int(idx[i + 1])) for i in range(0, len(idx), 2)]


def spo2_features(spo2, fs):
    """Oxygenation feature dict from one SpO2 channel. JSON-serialisable; every
    field is None when SpO2 is absent/unusable."""
    keys = ["spo2_mean", "spo2_min", "spo2_p1", "t90_pct", "t90_min",
            "odi", "n_desat", "desat_depth_mean", "desat_depth_max",
            "hypoxic_burden", "hb_pctmin_total", "duration_hours"]
    out = {k: None for k in keys}
    x, fs, n_valid = _normalise_spo2(spo2, fs)
    if x is None or fs <= 0 or n_valid < int(fs * 60):     # <1 min of valid data
        return out

    valid = x[np.isfinite(x)]
    hours = x.size / fs / 3600.0
    out["duration_hours"] = _r(hours, 3)
    out["spo2_mean"] = _r(np.mean(valid), 2)
    out["spo2_min"] = _r(np.min(valid), 1)
    out["spo2_p1"] = _r(np.percentile(valid, 1), 1)
    out["t90_pct"] = _r(100.0 * np.mean(valid < T90_LEVEL), 2)
    out["t90_min"] = _r(np.count_nonzero(valid < T90_LEVEL) / fs / 60.0, 1)

    # Running baseline: moving median over BASELINE_WIN_S (robust to the dips
    # themselves), computed on a forward-filled copy so NaN gaps don't poison it.
    filled = _ffill(x)
    if filled is None:
        return out
    win = max(1, int(BASELINE_WIN_S * fs))
    baseline = _moving_median(filled, win)
    drop = baseline - filled                            # positive where below baseline

    # Desaturation events: contiguous stretches where the drop exceeds DESAT_DROP.
    ev_mask = np.isfinite(x) & (drop >= DESAT_DROP)
    runs = _runs_true(ev_mask)
    min_len = int(MIN_EVENT_S * fs)
    max_len = int(MAX_EVENT_S * fs)
    depths, burdens = [], []
    n_ev = 0
    for a, b in runs:
        if (b - a) < min_len:
            continue
        seg_drop = drop[a:min(b, a + max_len)]
        seg_drop = seg_drop[np.isfinite(seg_drop) & (seg_drop > 0)]
        if seg_drop.size == 0:
            continue
        n_ev += 1
        depths.append(float(seg_drop.max()))
        # area (%*s) between baseline and curve over the event, -> %*min
        burdens.append(float(np.trapz(seg_drop, dx=1.0 / fs) / 60.0))
    out["n_desat"] = int(n_ev)
    out["odi"] = _r(n_ev / hours, 2) if hours > 0 else None
    if depths:
        out["desat_depth_mean"] = _r(np.mean(depths), 2)
        out["desat_depth_max"] = _r(np.max(depths), 1)
    total_burden = float(np.sum(burdens)) if burdens else 0.0
    out["hb_pctmin_total"] = _r(total_burden, 2)
    out["hypoxic_burden"] = _r(total_burden / hours, 2) if hours > 0 else None  # %*min / h
    return out


def _ffill(x):
    """Forward-fill NaNs in a 1-D array (leading NaNs back-filled). None if all NaN."""
    x = np.asarray(x, float)
    mask = np.isfinite(x)
    if not mask.any():
        return None
    idx = np.where(mask, np.arange(x.size), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = x[idx]
    # back-fill any leading gap with the first valid value
    first = int(np.argmax(mask))
    if first > 0:
        filled[:first] = x[first]
    return filled


def _moving_median(x, win):
    """Approximate moving median via a strided percentile; falls back to moving
    mean for very long signals to stay cheap. Window is odd-centered."""
    n = x.size
    if win <= 1 or n == 0:
        return x.copy()
    # Downsample-based baseline: median within non-overlapping blocks, then
    # interpolate back to full length. O(n) and robust to dips, without an
    # O(n*win) sliding median.
    nb = max(1, n // win)
    edges = np.linspace(0, n, nb + 1, dtype=int)
    centers, meds = [], []
    for i in range(nb):
        a, b = edges[i], edges[i + 1]
        if b <= a:
            continue
        seg = x[a:b]
        centers.append((a + b) / 2.0)
        meds.append(float(np.median(seg)))
    if not centers:
        return x.copy()
    return np.interp(np.arange(n), np.array(centers), np.array(meds))
