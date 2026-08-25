"""Multi-channel EEG network features (advanced-feature roadmap B1 + B2).

The two "heavy" EEG items from ADVANCED_FEATURES_COVERAGE.md, both requiring the
full 6-derivation montage (F3,F4,C3,C4,O1,O2) rather than the single central
channel the other extractors use:

  * **B1 connectivity** (`connectivity_features`) — magnitude-squared coherence
    and weighted phase-lag index (wPLI) between EEG channel pairs, per band, for
    pooled NREM and REM. Coherence is the classic linear synchrony; wPLI is the
    volume-conduction / reference-robust phase-synchrony (only the imaginary part
    of the cross-spectrum), so the two are complementary. Reduced/reorganised
    sleep EEG connectivity is an exploratory marker of cortical disconnection in
    cognitive decline.
  * **B2 microstates** (`microstate_features`) — polarity-invariant modified
    k-means (K=4) on GFP-peak topographies over NREM, summarised with
    *label-permutation-invariant* dynamics only (GEV, mean state duration,
    transition rate, coverage entropy, GFP peak rate). Per-map coverage is NOT
    reported because cluster identity is arbitrary across recordings; only the
    global spatio-temporal-complexity summaries are comparable across subjects.

Montage robustness: the cohort is montage-heterogeneous (BIDMC/Emory mastoid-
referenced, Kaiser exposes raw + M1/M2). To make features comparable across
sites, both functions re-reference to the **common average** (CAR) of the
available scalp derivations before any computation — this puts every recording in
the same reference frame without needing to know the original one. Only the six
standard scalp derivations are used (mastoids excluded) so the channel set is
identical at all three sites.

Everything is JSON-serialisable and degrades to {"ok": False} rather than raising,
so it is safe to run unattended across a cohort. Cost is controlled by evenly
sub-sampling epochs per stage and vectorising the segment FFTs.
"""
import warnings

import numpy as np

try:
    from scipy import signal as _sp_signal
    _SCIPY_OK = True
except Exception:                                   # pragma: no cover
    _sp_signal = None
    _SCIPY_OK = False

EPOCH_SEC = 30.0
# stage codes (match eeg_spectral / dyn_features): 5=Wake 3=N1 2=N2 1=N3 4=REM
NREM_CODES = (3, 2, 1)
REM_CODES = (4,)

# EEG bands (Hz) — same panel as eeg_spectral. sigma = spindle band.
BANDS = [("delta", 0.5, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 12.0),
         ("sigma", 12.0, 16.0), ("beta", 16.0, 30.0)]
_BAND_NAMES = [b[0] for b in BANDS]

# scalp derivations we use (mastoids m1/m2 deliberately excluded — reference only)
_SCALP_CUES = ["f3", "f4", "c3", "c4", "o1", "o2",
               "cz", "pz", "fz", "fp1", "fp2", "f7", "f8", "p3", "p4", "t3", "t4"]

# cross-spectral estimation
SEG_SEC = 2.0                                       # segment length for coherence/wPLI
SEG_OVERLAP = 0.5                                   # 50% overlap (Welch-style)
MIN_SEGMENTS = 20                                   # below this the estimates are too noisy
MAX_CONN_EPOCHS = 160                               # cap epochs/stage (evenly sampled)
MIN_STAGE_EPOCHS = 10                               # need >= 5 min of the stage
MAX_SEGMENTS = 5000                                 # hard cap on FFT segments/stage

# microstates
MS_K = 4
MS_BAND = (2.0, 20.0)                               # broadband EEG for microstates
MS_MAX_NREM_EPOCHS = 240                            # cap NREM epochs used
MS_MAX_PEAKS = 3000                                 # GFP peaks used for clustering
MS_KMEANS_ITERS = 40
MS_MIN_DUR_MS = 30.0                                # merge sub-30 ms segments (standard)
MS_MIN_PEAKS = 100


# --------------------------------------------------------------------------- utils
def _finite(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _r(x, nd=4):
    v = _finite(x)
    return None if v is None else round(v, nd)


def _align(n_samples, fs, codes):
    spe = int(round(fs * EPOCH_SEC))
    if spe < 1:
        return None, 0, codes
    n_ep = min(len(codes), n_samples // spe)
    return spe, n_ep, codes[:n_ep]


def _is_scalp(label):
    lab = label.lower()
    return any(c in lab for c in _SCALP_CUES)


def _find_idx(labels, sub):
    """Index of the first label containing substring `sub` (lower-case), or None."""
    for i, lab in enumerate(labels):
        if sub in lab.lower():
            return i
    return None


def _car(sig2d):
    """Common-average re-reference: subtract the across-channel mean at each sample.
    NaNs are zeroed first so a single bad channel can't wipe the average."""
    x = np.nan_to_num(np.asarray(sig2d, float), nan=0.0, posinf=0.0, neginf=0.0)
    return x - x.mean(axis=0, keepdims=True)


def _sample_epochs(idx, cap):
    """Evenly sub-sample an epoch-index array to at most `cap`."""
    if idx.size > cap:
        return idx[np.linspace(0, idx.size - 1, cap).astype(int)]
    return idx


# ============================================================= B1 connectivity
def _seg_fft(x, seg_len, hop, win):
    """Vectorised per-segment rFFT of a 1-D signal -> (n_seg, nf) complex, or None.
    Segments are mean-removed and Hann-windowed (Welch-style)."""
    n = x.size
    if n < seg_len:
        return None
    starts = np.arange(0, n - seg_len + 1, hop)
    if starts.size == 0:
        return None
    if starts.size > MAX_SEGMENTS:                  # evenly thin to the cap
        starts = starts[np.linspace(0, starts.size - 1, MAX_SEGMENTS).astype(int)]
    idx = starts[:, None] + np.arange(seg_len)[None, :]
    segs = x[idx]
    segs = segs - segs.mean(axis=1, keepdims=True)
    segs = segs * win[None, :]
    return np.fft.rfft(segs, axis=1)


def _band_masks(freqs):
    return {nm: (freqs >= lo) & (freqs < hi) for nm, lo, hi in BANDS}


def _pairwise_conn(sig2d, fs):
    """Coherence + wPLI per band for every channel pair of a (n_ch, n_samp) block.

    Returns (coh, wpli, freqs_ok) where coh[(i,j)][band] and wpli[(i,j)][band] are
    scalars in [0,1], or None if there are too few segments.
    """
    n_ch, n = sig2d.shape
    seg_len = int(round(SEG_SEC * fs))
    if seg_len < 8 or fs < 2 * BANDS[-1][2]:
        return None, None
    hop = max(1, int(round(seg_len * (1.0 - SEG_OVERLAP))))
    win = np.hanning(seg_len)
    freqs = np.fft.rfftfreq(seg_len, d=1.0 / fs)
    bmask = _band_masks(freqs)

    # per-channel segment FFTs (aligned: identical starts for every channel)
    F = []
    n_seg = None
    for c in range(n_ch):
        Fc = _seg_fft(sig2d[c], seg_len, hop, win)
        if Fc is None:
            return None, None
        if n_seg is None:
            n_seg = Fc.shape[0]
        F.append(Fc[:n_seg])                         # guard equal segment count
    if n_seg is None or n_seg < MIN_SEGMENTS:
        return None, None

    Sxx = [np.mean(np.abs(Fc) ** 2, axis=0) for Fc in F]     # (nf,) per channel
    coh, wpli = {}, {}
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            cross = F[i] * np.conj(F[j])             # (n_seg, nf)
            Sxy = cross.mean(axis=0)                 # complex averaged cross-spectrum
            denom = Sxx[i] * Sxx[j]
            with np.errstate(divide="ignore", invalid="ignore"):
                msc = np.where(denom > 0, (np.abs(Sxy) ** 2) / denom, np.nan)
            im = cross.imag
            num = np.abs(im.mean(axis=0))
            den = np.mean(np.abs(im), axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                wp = np.where(den > 0, num / den, np.nan)
            cb, wb = {}, {}
            for nm in _BAND_NAMES:
                m = bmask[nm]
                cseg = msc[m][np.isfinite(msc[m])] if np.any(m) else np.array([])
                wseg = wp[m][np.isfinite(wp[m])] if np.any(m) else np.array([])
                cb[nm] = _finite(cseg.mean()) if cseg.size else None
                wb[nm] = _finite(wseg.mean()) if wseg.size else None
            coh[(i, j)] = cb
            wpli[(i, j)] = wb
    return coh, wpli


def _agg_band(pairdicts, pairs, band):
    """Mean over the given pairs of a per-pair per-band scalar."""
    vals = [pairdicts[p][band] for p in pairs
            if p in pairdicts and pairdicts[p].get(band) is not None]
    return _finite(np.mean(vals)) if vals else None


def _stage_connectivity(sig2d, fs, labels, codes, stage_codes_wanted):
    """Concatenate sampled epochs of the wanted stages, compute pairwise coherence/
    wPLI, and aggregate to global + region-pair summaries. Returns a dict or None."""
    spe, n_ep, codes = _align(sig2d.shape[1], fs, codes)
    if spe is None or n_ep < 1:
        return None
    idx = np.where(np.isin(codes[:n_ep], stage_codes_wanted))[0]
    if idx.size < MIN_STAGE_EPOCHS:
        return None
    idx = _sample_epochs(idx, MAX_CONN_EPOCHS)
    block = np.concatenate([sig2d[:, e * spe:(e + 1) * spe] for e in idx], axis=1)
    coh, wpli = _pairwise_conn(block, fs)
    if coh is None:
        return None

    n_ch = sig2d.shape[0]
    all_pairs = [(i, j) for i in range(n_ch) for j in range(i + 1, n_ch)]
    out = {}
    for nm in _BAND_NAMES:
        out[f"coh_{nm}"] = _r(_agg_band(coh, all_pairs, nm))
        out[f"wpli_{nm}"] = _r(_agg_band(wpli, all_pairs, nm))

    # region-pair contrasts (only if the channels are present)
    def _pair(a, b):
        ia, ib = _find_idx(labels, a), _find_idx(labels, b)
        if ia is None or ib is None:
            return None
        return (min(ia, ib), max(ia, ib))

    inter = [p for p in (_pair("f3", "f4"), _pair("c3", "c4"), _pair("o1", "o2"))
             if p is not None]
    antpost = [p for p in (_pair("f3", "o1"), _pair("f4", "o2")) if p is not None]
    out["coh_interhemi_alpha"] = _r(_agg_band(coh, inter, "alpha")) if inter else None
    out["coh_antpost_theta"] = _r(_agg_band(coh, antpost, "theta")) if antpost else None
    return out


def connectivity_features(eeg_channels, fss, labels, stage_codes):
    """B1: coherence + wPLI network features for pooled NREM and REM.

    `eeg_channels` is a list/array of 1-D EEG signals (the 6 scalp derivations),
    `fss` their sampling rates (must share a common fs), `labels` their channel
    names (for region-pair lookup), `stage_codes` the per-epoch CAISR staging.
    """
    if not _SCIPY_OK or eeg_channels is None or len(eeg_channels) < 2:
        return {"ok": False, "error": "need >=2 EEG channels + scipy"}
    fs_set = {round(float(f), 3) for f in fss}
    if len(fs_set) != 1:
        return {"ok": False, "error": f"EEG channels differ in fs: {fs_set}"}
    fs = float(list(fs_set)[0])
    if fs < 2 * BANDS[-1][2]:
        return {"ok": False, "error": "sampling rate too low for the beta band"}
    codes = np.rint(np.asarray(stage_codes, float)).astype(int)
    if codes.size == 0:
        return {"ok": False, "error": "no staging"}

    n = min(len(x) for x in eeg_channels)
    sig2d = _car(np.vstack([np.asarray(x, float)[:n] for x in eeg_channels]))

    stages = {}
    nrem = _stage_connectivity(sig2d, fs, labels, codes, NREM_CODES)
    if nrem is not None:
        stages["nrem"] = nrem
    rem = _stage_connectivity(sig2d, fs, labels, codes, REM_CODES)
    if rem is not None:
        stages["rem"] = rem
    if not stages:
        return {"ok": False, "error": "no stage had enough clean epochs"}
    return {"ok": True, "n_channels": len(eeg_channels), "stages": stages}


# --------------------------------------------------------- B1 flat schema
_CONN_FEATS = ([f"coh_{nm}" for nm in _BAND_NAMES]
               + [f"wpli_{nm}" for nm in _BAND_NAMES]
               + ["coh_interhemi_alpha", "coh_antpost_theta"])
_CONN_TAGS = ["nrem", "rem"]


def flatten_connectivity(out, row=None):
    row = {} if row is None else row
    stages = out.get("stages", {}) if out.get("ok") else {}
    for tag in _CONN_TAGS:
        s = stages.get(tag) or {}
        for feat in _CONN_FEATS:
            row[f"conn_{tag}_{feat}"] = s.get(feat)
    return row


def connectivity_columns():
    return [f"conn_{tag}_{feat}" for tag in _CONN_TAGS for feat in _CONN_FEATS]


# ============================================================= B2 microstates
def _bandpass(x, fs, band):
    lo, hi = band
    ny = 0.5 * fs
    hi = min(hi, ny * 0.99)
    if lo >= hi:
        return None
    b, a = _sp_signal.butter(4, [lo / ny, hi / ny], btype="band")
    return _sp_signal.filtfilt(b, a, x)


def _gfp_peaks(gfp):
    """Indices of local maxima of the global field power series."""
    if gfp.size < 3:
        return np.array([], int)
    left = gfp[1:-1] > gfp[:-2]
    right = gfp[1:-1] >= gfp[2:]
    return np.where(left & right)[0] + 1


def _ms_kmeans(V, gfp_pk, k=MS_K, n_iter=MS_KMEANS_ITERS, seed=0):
    """Polarity-invariant modified k-means on unit-norm topographies V (n, n_ch).
    Returns (maps (k, n_ch), labels (n,), corr (n,)). Cluster assignment maximises
    |correlation|; each map is updated to the first eigenvector of its members'
    scatter matrix (Pascual-Marqui 1995)."""
    n = V.shape[0]
    rng = np.random.RandomState(seed)
    maps = V[rng.choice(n, k, replace=False)].copy()
    maps /= (np.linalg.norm(maps, axis=1, keepdims=True) + 1e-12)
    labels = np.zeros(n, int)
    for _ in range(n_iter):
        C = V @ maps.T                               # (n, k) correlations (V, maps unit-norm)
        new_labels = np.argmax(np.abs(C), axis=1)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for kk in range(k):
            Vk = V[labels == kk]
            if Vk.shape[0] == 0:
                continue
            S = Vk.T @ Vk
            w, U = np.linalg.eigh(S)
            maps[kk] = U[:, -1]
        maps /= (np.linalg.norm(maps, axis=1, keepdims=True) + 1e-12)
    C = V @ maps.T
    labels = np.argmax(np.abs(C), axis=1)
    corr = np.abs(C[np.arange(n), labels])
    return maps, labels, corr


def _merge_short(labels, min_len):
    """Relabel runs shorter than `min_len` samples to the preceding run's label."""
    if min_len <= 1 or labels.size == 0:
        return labels
    out = labels.copy()
    i = 0
    n = out.size
    while i < n:
        j = i + 1
        while j < n and out[j] == out[i]:
            j += 1
        if (j - i) < min_len and i > 0:
            out[i:j] = out[i - 1]
        i = j
    return out


def microstate_features(eeg_channels, fss, stage_codes):
    """B2: permutation-invariant EEG-microstate dynamics over NREM.

    Returns GEV, GFP peak rate/mean, mean state duration, transition rate,
    coverage entropy, effective #states, and mean backfit correlation.
    """
    if not _SCIPY_OK or eeg_channels is None or len(eeg_channels) < MS_K:
        return {"ok": False, "error": f"need >={MS_K} EEG channels + scipy"}
    fs_set = {round(float(f), 3) for f in fss}
    if len(fs_set) != 1:
        return {"ok": False, "error": f"EEG channels differ in fs: {fs_set}"}
    fs = float(list(fs_set)[0])
    codes = np.rint(np.asarray(stage_codes, float)).astype(int)
    if codes.size == 0 or fs <= 0:
        return {"ok": False, "error": "no staging"}

    n = min(len(x) for x in eeg_channels)
    spe, n_ep, codes = _align(n, fs, codes)
    if spe is None or n_ep < 1:
        return {"ok": False, "error": "signal shorter than one epoch"}
    idx = np.where(np.isin(codes[:n_ep], NREM_CODES))[0]
    if idx.size < MIN_STAGE_EPOCHS:
        return {"ok": False, "error": "insufficient NREM"}
    idx = _sample_epochs(idx, MS_MAX_NREM_EPOCHS)

    # CAR then band-pass each channel, then keep only the sampled-NREM samples
    car = _car(np.vstack([np.asarray(x, float)[:n_ep * spe] for x in eeg_channels]))
    try:
        filt = np.vstack([_bandpass(car[c], fs, MS_BAND) for c in range(car.shape[0])])
    except Exception as e:
        return {"ok": False, "error": f"bandpass failed: {type(e).__name__}"}
    if filt is None or not np.all(np.isfinite(filt)):
        filt = np.nan_to_num(filt, nan=0.0, posinf=0.0, neginf=0.0)
    keep = np.concatenate([np.arange(e * spe, (e + 1) * spe) for e in idx])
    data = filt[:, keep]                             # (n_ch, n_keep)
    n_ch, n_keep = data.shape
    if n_keep < int(fs * 60):
        return {"ok": False, "error": "too few NREM samples"}
    dur_sec = n_keep / fs

    gfp = data.std(axis=0)
    peaks = _gfp_peaks(gfp)
    peaks = peaks[gfp[peaks] > 0]
    n_peaks_raw = int(peaks.size)                    # rate uses the RAW count, not the cap
    if peaks.size < MS_MIN_PEAKS:
        return {"ok": False, "error": "too few GFP peaks"}
    if peaks.size > MS_MAX_PEAKS:                     # subsample only for the k-means fit
        peaks = peaks[np.linspace(0, peaks.size - 1, MS_MAX_PEAKS).astype(int)]

    # unit-norm topographies at GFP peaks -> cluster
    Vpk = data[:, peaks].T                           # (n_pk, n_ch)
    Vpk = Vpk - Vpk.mean(axis=1, keepdims=True)      # average-reference each map
    norms = np.linalg.norm(Vpk, axis=1, keepdims=True)
    Vpk = Vpk / (norms + 1e-12)
    maps, pk_labels, pk_corr = _ms_kmeans(Vpk, gfp[peaks])

    # GEV on the peak set: sum(gfp^2 * corr^2) / sum(gfp^2)
    g2 = gfp[peaks] ** 2
    gev = _finite(np.sum(g2 * pk_corr ** 2) / np.sum(g2)) if np.sum(g2) > 0 else None

    # backfit every kept sample -> label train (for durations/transitions/coverage)
    Vall = data.T - data.T.mean(axis=1, keepdims=True)
    Vall = Vall / (np.linalg.norm(Vall, axis=1, keepdims=True) + 1e-12)
    Call = Vall @ maps.T
    all_labels = np.argmax(np.abs(Call), axis=1)
    all_corr = np.abs(Call[np.arange(n_keep), all_labels])
    all_labels = _merge_short(all_labels, int(round(MS_MIN_DUR_MS / 1000.0 * fs)))

    # run-length segmentation of the smoothed label train
    change = np.nonzero(np.diff(all_labels))[0] + 1
    seg_bounds = np.concatenate([[0], change, [n_keep]])
    seg_len = np.diff(seg_bounds)
    n_seg = seg_len.size
    mean_dur_ms = _finite(np.mean(seg_len) / fs * 1000.0)
    trans_rate = _finite((n_seg - 1) / dur_sec) if dur_sec > 0 else None

    cov = np.bincount(all_labels, minlength=MS_K).astype(float)
    p = cov / cov.sum() if cov.sum() > 0 else cov
    nz = p[p > 0]
    entropy = _finite(-np.sum(nz * np.log(nz)))
    n_eff = _finite(np.exp(entropy)) if entropy is not None else None

    return {
        "ok": True,
        "n_channels": n_ch,
        "ms_gev": _r(gev),
        "ms_gfp_peak_rate": _r(n_peaks_raw / dur_sec, 4),
        "ms_gfp_mean": _r(float(np.mean(gfp)), 4),
        "ms_mean_dur_ms": _r(mean_dur_ms, 2),
        "ms_trans_rate": _r(trans_rate, 4),
        "ms_coverage_entropy": _r(entropy),
        "ms_n_eff": _r(n_eff),
        "ms_mean_corr": _r(float(np.mean(all_corr)), 4),
    }


# --------------------------------------------------------- B2 flat schema
_MS_FEATS = ["ms_gev", "ms_gfp_peak_rate", "ms_gfp_mean", "ms_mean_dur_ms",
             "ms_trans_rate", "ms_coverage_entropy", "ms_n_eff", "ms_mean_corr"]


def flatten_microstate(out, row=None):
    row = {} if row is None else row
    ok = out.get("ok")
    for f in _MS_FEATS:
        row[f] = out.get(f) if ok else None
    return row


def microstate_columns():
    return list(_MS_FEATS)


# ------------------------------------------------------------------ combined schema
def net_columns():
    return connectivity_columns() + microstate_columns()
