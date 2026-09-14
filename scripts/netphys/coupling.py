#!/usr/bin/env python3
"""Directed cross-system coupling features — a compact port of the Granger-G core of
`causal_networks_physiology` (M. Günther), adapted to the Challenge-2026 pipeline.

The original sweeps 8 timescales over young/elderly/OSA groups and writes per-group
CSVs. We keep only the *estimator* and turn it into a small block of per-recording
features: directed Granger G-causality among three 1-Hz signals — **breath rate**,
**heart rate**, and an **EEG band envelope** — computed per sleep-stage group and
duration-weighted across that stage's continuous patches.

Granger G (Geweke 1982), as in the source:

    G(caused <- causing) = ln( sigma1 / sigma2 )

where sigma2 is the 1-step forecast error variance of `caused` in the **full** VAR
(all three signals) and sigma1 the same in the VAR with `causing` **removed**. G > 0
means `causing` improves prediction of `caused` (directed influence). Per patch we
z-normalise each signal and require ADF stationarity (regression="ct"), reducing the
VAR order 5->3 to find a stationary fit — mirroring the source's `checkStatCalcG`.

This module is signal-source-agnostic: it operates on numpy arrays + a per-sample stage
label. `extract_features.py` wires it to the EDF/CAISR pipeline; here we only need
numpy + statsmodels, so the estimator is unit-testable on synthetic coupled signals
(`python coupling.py` runs that self-test).
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore")

SIGNALS = ("breath", "heart", "eeg")            # fixed column order
PAIRS = [(c, g) for c in SIGNALS for g in SIGNALS if c != g]   # 6 ordered pairs
MAX_LAG = 5                                     # source model_lag_order
MIN_LAG = 3                                     # source stops reducing at order 3
ALPHA = 0.05                                    # ADF null-rejection level
MIN_PATCH = 5                                   # patch must exceed 5*timescale samples


def all_feature_names(stage_groups, timescales=(1,)) -> list:
    """Deterministic full list of columns `coupling_features` can emit — needed so the
    exporter can write a fixed CSV header (failed combos are written empty -> NaN)."""
    cols = []
    for gname in stage_groups:
        for ts in timescales:
            pref = f"cnp__g_ts{ts}_{gname}"
            for caused, causing in PAIRS:
                cols.append(f"{pref}_{caused}_from_{causing}")
            cols.append(f"{pref}_statfrac")
    return cols


def _block_average(x: np.ndarray, k: int) -> np.ndarray:
    """Resample a 1-Hz series to a coarser timescale by non-overlapping block mean."""
    if k <= 1:
        return np.asarray(x, dtype=float)
    n = (len(x) // k) * k
    if n == 0:
        return np.asarray(x[:0], dtype=float)
    return np.asarray(x[:n], dtype=float).reshape(-1, k).mean(axis=1)


def _znorm_cols(M: np.ndarray) -> np.ndarray:
    """Column-wise z-normalisation; returns None if any column is degenerate."""
    mu = M.mean(axis=0)
    sd = M.std(axis=0)
    if np.any(sd == 0) or not np.all(np.isfinite(sd)):
        return None
    return (M - mu) / sd


def _all_stationary(M: np.ndarray, order: int) -> bool:
    """ADF stationarity (trend="ct") on every column at fixed maxlag=order."""
    from statsmodels.tsa.stattools import adfuller
    for j in range(M.shape[1]):
        col = M[:, j]
        col = col[np.isfinite(col)]
        if len(col) < order + 5:
            return False
        try:
            p = adfuller(col, regression="ct", maxlag=order, autolag=None)[1]
        except Exception:
            return False
        if not (p < ALPHA):
            return False
    return True


def _g_matrix(M: np.ndarray, order: int) -> dict:
    """Directed G for all 6 ordered pairs of a 3-column (breath,heart,eeg) matrix.

    One full VAR gives sigma2 for every `caused`; three restricted VARs (dropping one
    signal each) give sigma1. Returns {(caused, causing): G}. Empty on fit failure.
    """
    from statsmodels.tsa.api import VAR
    out: dict = {}
    try:
        full = VAR(M).fit(maxlags=order, ic=None, trend="ct")
        sig_full = np.diag(np.asarray(full.forecast_cov(1)[0]))
    except Exception:
        return out
    idx = {name: i for i, name in enumerate(SIGNALS)}
    sig2 = {SIGNALS[i]: sig_full[i] for i in range(len(SIGNALS))}
    # restricted models: drop each signal once -> sigma1 for the pairs it "causes"
    sig1: dict = {}
    for drop_i, dropped in enumerate(SIGNALS):
        keep = [i for i in range(len(SIGNALS)) if i != drop_i]
        try:
            r = VAR(M[:, keep]).fit(maxlags=order, ic=None, trend="ct")
            rc = np.diag(np.asarray(r.forecast_cov(1)[0]))
        except Exception:
            continue
        for pos, i in enumerate(keep):
            sig1[(SIGNALS[i], dropped)] = rc[pos]
    for caused, causing in PAIRS:
        s2 = sig2.get(caused)
        s1 = sig1.get((caused, causing))
        if s1 is not None and s2 is not None and s1 > 0 and s2 > 0:
            out[(caused, causing)] = float(np.log(s1 / s2))
    return out


def _patch_g(patch: np.ndarray, timescale: int) -> dict:
    """G for one contiguous stage patch: block-average -> z-norm -> order-reduce ADF."""
    M = np.column_stack([_block_average(patch[:, j], timescale)
                         for j in range(patch.shape[1])])
    if len(M) <= MIN_PATCH * 1:            # need enough coarse samples
        return {}
    M = _znorm_cols(M)
    if M is None:
        return {}
    for order in range(MAX_LAG, MIN_LAG - 1, -1):
        if _all_stationary(M, order):
            return _g_matrix(M, order)
    return {}


def _contiguous_runs(mask: np.ndarray):
    """Yield (start, stop) index pairs of maximal True runs in a boolean array."""
    if mask.size == 0:
        return
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    for a, b in zip(idx[0::2], idx[1::2]):
        yield int(a), int(b)


def coupling_features(breath: np.ndarray, heart: np.ndarray, eeg: np.ndarray,
                      stage: np.ndarray, stage_groups: dict,
                      timescales=(1,)) -> dict:
    """Per-recording directed-coupling features.

    Parameters
    ----------
    breath, heart, eeg : 1-Hz float arrays, equal length (NaN allowed = missing second)
    stage              : same-length int/str array of the per-second sleep stage
    stage_groups       : {group_name: set(stage_values)} e.g. {"nrem": {1,2,3}, "rem": {4}}
    timescales         : block-average timescales in seconds to compute G at

    Returns
    -------
    dict feature_name -> value. For each (group, timescale) the 6 directed G values are
    duration-weighted across that group's stationary patches; plus a `_statfrac`
    coverage feature = fraction of the group's time that yielded a usable (stationary) G.
    Missing/failed combinations are omitted (join fills them NaN downstream).
    """
    n = min(len(breath), len(heart), len(eeg), len(stage))
    sig = np.column_stack([np.asarray(breath[:n], float),
                           np.asarray(heart[:n], float),
                           np.asarray(eeg[:n], float)])
    stage = np.asarray(stage[:n])
    feats: dict = {}
    for gname, gvals in stage_groups.items():
        gmask = np.isin(stage, list(gvals))
        for ts in timescales:
            acc = {p: [] for p in PAIRS}       # list of (G, weight)
            used = 0.0
            total = 0.0
            for a, b in _contiguous_runs(gmask):
                dur = b - a
                if dur <= MIN_PATCH * ts:
                    continue
                total += dur
                patch = sig[a:b]
                # a patch with any all-NaN column is unusable
                if np.any(~np.isfinite(patch).any(axis=0)):
                    continue
                g = _patch_g(patch, ts)
                if g:
                    used += dur
                    for p, val in g.items():
                        acc[p].append((val, dur))
            pref = f"cnp__g_ts{ts}_{gname}"
            for (caused, causing), vw in acc.items():
                if vw:
                    vals = np.array([v for v, _ in vw])
                    wts = np.array([w for _, w in vw], dtype=float)
                    feats[f"{pref}_{caused}_from_{causing}"] = float(
                        np.sum(vals * wts) / np.sum(wts))
            feats[f"{pref}_statfrac"] = float(used / total) if total > 0 else 0.0
    return feats


# --------------------------------------------------------------------------- #
# self-test: recover a known directed coupling  heart <- breath  (breath drives heart)
# --------------------------------------------------------------------------- #
def _self_test():
    rng = np.random.default_rng(0)
    n = 3000
    # breath: AR(1); heart: driven by lagged breath; eeg: independent AR(1)
    breath = np.zeros(n)
    heart = np.zeros(n)
    eeg = np.zeros(n)
    for t in range(2, n):
        breath[t] = 0.6 * breath[t - 1] + rng.normal(0, 1)
        heart[t] = 0.3 * heart[t - 1] + 0.7 * breath[t - 1] + rng.normal(0, 1)
        eeg[t] = 0.5 * eeg[t - 1] + rng.normal(0, 1)
    stage = np.full(n, 2)                     # all N2 -> one NREM patch
    feats = coupling_features(breath, heart, eeg, stage,
                              {"nrem": {1, 2, 3}}, timescales=(1,))
    g_h_from_b = feats.get("cnp__g_ts1_nrem_heart_from_breath")
    g_b_from_h = feats.get("cnp__g_ts1_nrem_breath_from_heart")
    print("self-test (breath -> heart should dominate):")
    for k in sorted(feats):
        print(f"  {k:42s} {feats[k]:+.4f}")
    assert g_h_from_b is not None and g_b_from_h is not None, "G not computed"
    assert g_h_from_b > g_b_from_h, "directional asymmetry not recovered"
    assert g_h_from_b > 0.1, "true coupling too weak / not detected"
    print("\nOK: recovered breath -> heart >> heart -> breath")


if __name__ == "__main__":
    _self_test()
