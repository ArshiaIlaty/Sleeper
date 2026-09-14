#!/usr/bin/env python3
"""Time-Delay Stability (TDS) — the network-physiology link estimator.

Port of the coupling method from Bashan, Bartsch, Kantelhardt, Havlin & Ivanov,
"Network physiology reveals relations between network topology and physiological
function" (Nat. Commun. 3:702, 2012). Unlike the directed Granger-G in `coupling.py`
(pairwise, VAR-based), TDS scales cleanly to many nodes and produces the classic
sleep-stage network *reconfiguration*: a link between two systems exists when the
time lag of their maximum cross-correlation stays stable over consecutive windows.

Method (per Bashan 2012, §Methods):
  * Slide a window of length ``L`` (s) with step ``step`` (s) across two 1-Hz signals.
  * In each window, z-normalise both signals and compute the cross-correlation for
    integer lags ``tau in [-tau_max, +tau_max]``; record ``tau0``, the lag of maximum
    |cross-correlation| (a signal that is flat in the window -> tau0 = NaN).
  * A window is *time-delay stable* if it lies inside a run of >= ``min_run`` consecutive
    windows whose ``tau0`` changes by <= ``tol`` between neighbours.
  * The link strength for a pair is the fraction of windows that are TDS. Split by the
    stage label at each window's centre to get one network per sleep stage.

`compute_tds_networks` takes a dict of aligned 1-Hz node signals + per-second stage
labels and returns, per stage, the symmetric ``n_nodes x n_nodes`` matrix of TDS link
strengths (plus the pooled matrix and per-stage window counts). Pure numpy — no heavy
deps — so ``python tds.py`` runs a self-test that recovers a coupling present in one
"stage" and absent in another.
"""
from __future__ import annotations

import numpy as np

# Defaults follow the Bashan-2012 sleep analysis scale for 1-Hz signals.
L_DEFAULT = 60          # window length (s)
STEP_DEFAULT = 30       # window step (s) -> 50% overlap
TAU_MAX_DEFAULT = 10    # max lag searched (s)
TOL_DEFAULT = 1         # lag may change by <= tol between neighbouring windows
MIN_RUN_DEFAULT = 4     # a TDS segment is >= this many consecutive stable windows


def _zscore_rows(M: np.ndarray):
    """Z-normalise each row; rows with ~0 variance become NaN (flat -> no coupling)."""
    mu = M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, keepdims=True)
    Z = (M - mu) / sd
    Z[(sd[:, 0] < 1e-9), :] = np.nan
    return Z


def _window_tau0(Z: np.ndarray, tau_max: int):
    """Lag of max |cross-correlation| for every node pair in one window.

    Z: (n_nodes, L) z-normalised window. Returns (n, n) float array of tau0 (signed
    lag, node_i vs node_j), NaN where either node is flat. tau0>0 means j leads i
    (i(t) best matches j(t+tau0)); the matrix is antisymmetric up to sign, but we only
    use |tau0| stability, so the sign convention does not affect link strength.
    """
    n, L = Z.shape
    lags = np.arange(-tau_max, tau_max + 1)
    # corr[k] = <Z_i(t) Z_j(t+lag_k)> over the valid overlap
    corr = np.full((len(lags), n, n), np.nan)
    for k, tau in enumerate(lags):
        if tau >= 0:
            A, B = Z[:, : L - tau], Z[:, tau:]
        else:
            A, B = Z[:, -tau:], Z[:, : L + tau]
        m = A.shape[1]
        if m <= 1:
            continue
        corr[k] = (A @ B.T) / m
    with np.errstate(invalid="ignore"):
        finite = np.isfinite(corr).all(axis=0)          # (n,n): pair usable in all lags
        idx = np.nanargmax(np.abs(np.where(np.isfinite(corr), corr, -np.inf)), axis=0)
    tau0 = lags[idx].astype(float)
    tau0[~finite] = np.nan
    return tau0


def _stable_mask(tau0_seq: np.ndarray, tol: int, min_run: int):
    """Boolean mask over windows: True where tau0 is in a TDS run.

    tau0_seq: (n_windows,) lag sequence for one pair. A window is stable if it belongs
    to a maximal run of >= min_run consecutive windows with |diff(tau0)| <= tol (NaN
    breaks a run).
    """
    W = len(tau0_seq)
    if W < min_run:
        return np.zeros(W, bool)
    d = np.abs(np.diff(tau0_seq))
    linked = np.isfinite(d) & (d <= tol)                # edge i<->i+1 is "stable"
    mask = np.zeros(W, bool)
    i = 0
    while i < len(linked):
        if not linked[i]:
            i += 1
            continue
        j = i
        while j < len(linked) and linked[j]:
            j += 1
        # edges i..j-1 stable -> windows i..j form a run of (j-i+1) windows
        if (j - i + 1) >= min_run:
            mask[i : j + 1] = True
        i = j + 1 if j > i else i + 1
    return mask


def compute_tds_networks(signals: dict, stage: np.ndarray, stage_groups: dict,
                         L=L_DEFAULT, step=STEP_DEFAULT, tau_max=TAU_MAX_DEFAULT,
                         tol=TOL_DEFAULT, min_run=MIN_RUN_DEFAULT):
    """TDS link-strength network per stage from a dict of aligned 1-Hz node signals.

    signals: ordered {node_name -> 1-Hz array}; all arrays + `stage` share a timeline.
    stage:   per-second stage codes. stage_groups: {name -> set(codes)} (e.g. per stage).
    Returns dict with:
      nodes: list of node names (insertion order)
      per_stage: {group_name -> (n,n) symmetric TDS strength in [0,1]}
      pooled:    (n,n) TDS strength over all windows
      n_windows: {group_name -> int}, total_windows: int
    """
    nodes = list(signals.keys())
    n = len(nodes)
    T = min(len(stage), min(len(signals[k]) for k in nodes))
    X = np.vstack([np.asarray(signals[k], float)[:T] for k in nodes])   # (n, T)
    stage = np.asarray(stage, float)[:T]

    starts = np.arange(0, T - L + 1, step)
    W = len(starts)
    if W < min_run:
        empty = {g: np.zeros((n, n)) for g in stage_groups}
        return {"nodes": nodes, "per_stage": empty, "pooled": np.zeros((n, n)),
                "n_windows": {g: 0 for g in stage_groups}, "total_windows": W}

    tau0 = np.full((W, n, n), np.nan)                    # lag per window per pair
    win_stage = np.empty(W, float)                       # stage at each window centre
    for wi, s0 in enumerate(starts):
        Z = _zscore_rows(X[:, s0 : s0 + L])
        tau0[wi] = _window_tau0(Z, tau_max)
        win_stage[wi] = stage[s0 + L // 2]

    # stability mask per pair over the window axis
    stable = np.zeros((W, n, n), bool)
    for i in range(n):
        for j in range(i + 1, n):
            m = _stable_mask(tau0[:, i, j], tol, min_run)
            stable[:, i, j] = m
            stable[:, j, i] = m

    def _strength(win_sel):
        cnt = win_sel.sum()
        if cnt == 0:
            return np.zeros((n, n)), 0
        return stable[win_sel].mean(axis=0), int(cnt)

    per_stage, n_windows = {}, {}
    for g, codes in stage_groups.items():
        sel = np.isin(win_stage, list(codes))
        mat, cnt = _strength(sel)
        np.fill_diagonal(mat, 0.0)
        per_stage[g] = mat
        n_windows[g] = cnt
    pooled, _ = _strength(np.ones(W, bool))
    np.fill_diagonal(pooled, 0.0)
    return {"nodes": nodes, "per_stage": per_stage, "pooled": pooled,
            "n_windows": n_windows, "total_windows": W}


def _self_test():
    """Recover a lagged coupling present in stage A and absent in stage B."""
    rng = np.random.default_rng(0)
    fs_sec = 1
    dur = 3600
    t = np.arange(dur)
    # driver: slow AR(1)-ish envelope
    drv = np.zeros(dur)
    for k in range(1, dur):
        drv[k] = 0.98 * drv[k - 1] + rng.normal(0, 0.3)
    lag = 3
    follower = np.zeros(dur)
    follower[lag:] = drv[:-lag]                          # follower lags driver by 3 s
    # stage A (first half): follower strongly driven; stage B (second half): independent
    half = dur // 2
    y = follower + rng.normal(0, 0.2, dur)
    y[half:] = np.cumsum(rng.normal(0, 0.3, dur - half))  # decoupled in 2nd half
    noise = np.cumsum(rng.normal(0, 0.3, dur))            # unrelated 3rd node
    stage = np.where(np.arange(dur) < half, 1, 2)         # 1 = A, 2 = B

    sig = {"driver": drv, "follower": y, "noise": noise}
    out = compute_tds_networks(sig, stage, {"A": {1}, "B": {2}},
                               L=60, step=30, tau_max=10, tol=1, min_run=4)
    ni = {nm: k for k, nm in enumerate(out["nodes"])}
    a = out["per_stage"]["A"][ni["driver"], ni["follower"]]
    b = out["per_stage"]["B"][ni["driver"], ni["follower"]]
    an = out["per_stage"]["A"][ni["driver"], ni["noise"]]
    print(f"windows: {out['n_windows']}  total={out['total_windows']}")
    print(f"driver<->follower  TDS  stageA={a:.3f}  stageB={b:.3f}   (expect A >> B)")
    print(f"driver<->noise     TDS  stageA={an:.3f}                  (expect low)")
    assert a > 0.5, f"coupled stage TDS too low: {a}"
    assert a > b + 0.3, f"stages not separated: A={a} B={b}"
    assert an < 0.3, f"spurious coupling to noise: {an}"
    print("SELF-TEST PASS")


if __name__ == "__main__":
    _self_test()
