"""Small numeric-summary helpers used across the EDA modules.

Kept dependency-light (numpy only) so the suite runs on the pdmle system Python
without matplotlib/h5py. All summaries are JSON-serialisable (plain floats/ints).
"""
import numpy as np


def _f(x):
    """Coerce to a JSON-safe float (NaN/inf -> None)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _num(x):
    """Coerce a single value to float, mapping anything non-numeric to NaN.

    Element-wise (unlike a bulk ``np.asarray(dtype=float)``) so a stray string
    such as 'unknown' becomes a missing value instead of crashing the summary.
    """
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def numeric_summary(values):
    """Distribution summary for a 1-D numeric array-like.

    Returns count of finite values, missing count, mean/std, and a spread of
    percentiles. Non-finite entries are excluded from the stats but counted as
    missing so coverage is auditable.
    """
    arr = np.array([_num(v) for v in values], dtype=float)
    n_total = arr.size
    finite = arr[np.isfinite(arr)]
    n = finite.size
    out = {
        "n_total": int(n_total),
        "n_valid": int(n),
        "n_missing": int(n_total - n),
        "missing_pct": _f(100.0 * (n_total - n) / n_total) if n_total else None,
    }
    if n == 0:
        for k in ("mean", "std", "min", "p1", "p5", "p25", "median", "p75", "p95", "p99", "max"):
            out[k] = None
        return out
    pcts = np.percentile(finite, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    out.update({
        "mean": _f(np.mean(finite)),
        "std": _f(np.std(finite)),
        "min": _f(pcts[0]),
        "p1": _f(pcts[1]),
        "p5": _f(pcts[2]),
        "p25": _f(pcts[3]),
        "median": _f(pcts[4]),
        "p75": _f(pcts[5]),
        "p95": _f(pcts[6]),
        "p99": _f(pcts[7]),
        "max": _f(pcts[8]),
    })
    return out


def value_counts(values, dropna=True):
    """Category counts as an ordered dict (descending by count)."""
    counts = {}
    for v in values:
        if dropna and (v is None or (isinstance(v, float) and np.isnan(v))):
            continue
        key = "<NA>" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def histogram(values, bins):
    """Return {bin_label: count} for the given bin edges (numeric)."""
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    counts, edges = np.histogram(arr, bins=bins)

    def _edge(x):
        # Show fractional edges (e.g. BMI 18.5) faithfully; keep integers clean
        # so distinct bin boundaries never collide into the same dict key.
        return f"{x:g}"

    out = {}
    for i, c in enumerate(counts):
        out[f"[{_edge(edges[i])},{_edge(edges[i+1])})"] = int(c)
    return out


def rate_and_ci(k, n):
    """Proportion with a Wilson 95% CI. Returns (rate, lo, hi) as percentages."""
    if n == 0:
        return None, None, None
    p = k / n
    z = 1.959963984540054
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return _f(100 * p), _f(100 * (centre - half)), _f(100 * (centre + half))
