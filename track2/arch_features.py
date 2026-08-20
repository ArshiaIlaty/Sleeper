"""Sleep-architecture transition + microarousal features for the submission.

VENDORED, SELF-CONTAINED copy of scripts/viewer/arch_dynamics.py (+ the three
symbols it needs from scripts/viewer/dynamics.py) so it imports inside the
frozen submission container, where `scripts/` is dropped by .dockerignore.
Pure numpy, CAISR-annotation-only (no waveform decode, no neurokit2) -> cheap
enough to run per-record inside the Challenge inference time limit.

Two feature families (see the EDA / ablation writeups):
  * F1 transition-Markov model  -- 25 row-normalised P(next|cur) + per-row
    conditional entropies + occupancy-weighted entropy rate + stability index.
    Needs only the per-epoch stage_caisr stream.
  * A1 microarousal distributions -- arousal duration + inter-arousal-interval
    stats + per-stage arousal index. Needs arousal_caisr + its sampling rate.

Ablation on the plus cache (autonomic already in the core) showed this block
adds cross-site signal ON TOP OF the shipped features: LOSO AC-AUROC +0.034,
reward@pi +0.145 -> +0.233. `feature_vector()` returns a fixed-order 45-vector
(NaN for any missing modality) so the training/inference schema is stable.

Keep in sync with scripts/viewer/arch_dynamics.py + dynamics.py if either
changes; this is a deliberate duplicate (the container can't see scripts/).
"""
import numpy as np

EPOCH_SEC = 30.0
STAGE_ORDER = [5, 3, 2, 1, 4]                       # Wake, N1, N2, N3, REM
IDX = {code: i for i, code in enumerate(STAGE_ORDER)}
POOL_TAG = {5: "wake", 3: "n1", 2: "n2", 1: "n3", 4: "rem"}
TAG_ORDER = [POOL_TAG[c] for c in STAGE_ORDER]      # wake, n1, n2, n3, rem
AROUSAL_CODE = 1                                     # code 1 == arousal in arousal_caisr


# ------------------------------------------------ inlined from dynamics.py
def transition_matrix(seq):
    """5x5 count matrix over consecutive epoch pairs (codes, Unknown removed)."""
    m = np.zeros((5, 5), dtype=float)
    for a, b in zip(seq[:-1], seq[1:]):
        if a in IDX and b in IDX:
            m[IDX[a], IDX[b]] += 1.0
    return m


def _row_normalise(counts):
    """Row-normalise counts to probabilities; empty rows stay all-zero."""
    out = np.zeros_like(counts)
    rs = counts.sum(axis=1)
    nz = rs > 0
    out[nz] = counts[nz] / rs[nz, None]
    return out


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


def _runs(mask):
    """(start, end_exclusive) index runs of True in a boolean array."""
    if mask.size == 0:
        return []
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return [(int(idx[i]), int(idx[i + 1])) for i in range(0, len(idx), 2)]


def _row_entropy(p):
    """Shannon entropy of a probability row, normalised to [0,1] by log(k) where
    k = number of reachable states (>1). None for an empty/degenerate row."""
    p = p[p > 0]
    if p.size < 2:
        return 0.0 if p.size == 1 else None          # 1 reachable state -> 0 disorder
    h = float(-np.sum(p * np.log(p)))
    return h / np.log(len(STAGE_ORDER))              # normalise by log(5) for comparability


# --------------------------------------------------------------------------- F1
def transition_features(stage_codes):
    """Full transition-probability matrix + entropies from per-epoch stage codes."""
    if stage_codes is None or len(stage_codes) == 0:
        return {"ok": False, "error": "no stage stream"}
    st = np.rint(np.asarray(stage_codes, float)).astype(int)
    seq = st[st != 9]                                # drop Unknown, like dynamics.py
    if seq.size < 2:
        return {"ok": False, "error": "fewer than 2 scored epochs"}

    counts = transition_matrix(seq)                  # 5x5, STAGE_ORDER rows/cols
    probs = _row_normalise(counts)
    row_tot = counts.sum(axis=1)                     # transitions leaving each state
    occ = row_tot / row_tot.sum() if row_tot.sum() > 0 else np.zeros_like(row_tot)

    out = {"ok": True}
    for i, ftag in enumerate(TAG_ORDER):
        for j, ttag in enumerate(TAG_ORDER):
            out[f"trans_p_{ftag}_{ttag}"] = _r(probs[i, j]) if row_tot[i] > 0 else None

    ent = np.full(len(STAGE_ORDER), np.nan)
    for i, ftag in enumerate(TAG_ORDER):
        h = _row_entropy(probs[i]) if row_tot[i] > 0 else None
        out[f"trans_entropy_{ftag}"] = _r(h)
        if h is not None:
            ent[i] = h
    m = np.isfinite(ent) & (occ > 0)
    out["trans_entropy_rate"] = _r(float(np.sum(occ[m] * ent[m]) / occ[m].sum())) if m.any() else None
    diag = np.diag(probs)
    md = (row_tot > 0)
    out["trans_stability_index"] = _r(float(np.sum(occ[md] * diag[md]) / occ[md].sum())) if md.any() else None
    return out


# --------------------------------------------------------------------------- A1
def arousal_features(arousal_codes, arousal_fs, stage_codes=None):
    """Microarousal duration + inter-arousal-interval distributions, and per-stage
    arousal index, from the arousal_caisr stream."""
    if arousal_codes is None or len(arousal_codes) == 0:
        return {"ok": False, "error": "no arousal stream"}
    fs = _finite(arousal_fs)
    if not fs or fs <= 0:
        return {"ok": False, "error": "no arousal sampling rate"}
    codes = np.rint(np.asarray(arousal_codes, float)).astype(int)

    events = _runs(codes == AROUSAL_CODE)            # (start_sample, end_sample)
    out = {"ok": True, "n_arousals": len(events)}
    dkeys = ["arousal_dur_mean_s", "arousal_dur_median_s", "arousal_dur_sd_s",
             "arousal_dur_max_s", "arousal_iai_mean_s", "arousal_iai_median_s",
             "arousal_iai_cv"]
    for k in dkeys:
        out[k] = None
    for tag in TAG_ORDER:
        out[f"arousal_idx_{tag}"] = None

    if events:
        durs = np.array([(e - s) / fs for s, e in events])   # seconds
        out["arousal_dur_mean_s"] = _r(np.mean(durs), 2)
        out["arousal_dur_median_s"] = _r(np.median(durs), 2)
        out["arousal_dur_sd_s"] = _r(np.std(durs), 2) if durs.size > 1 else None
        out["arousal_dur_max_s"] = _r(np.max(durs), 1)
    if len(events) >= 3:
        onsets = np.array([s / fs for s, _ in events])       # onset seconds
        iai = np.diff(onsets)                                 # inter-arousal intervals
        out["arousal_iai_mean_s"] = _r(np.mean(iai), 2)
        out["arousal_iai_median_s"] = _r(np.median(iai), 2)
        out["arousal_iai_cv"] = _r(float(np.std(iai) / np.mean(iai))) if np.mean(iai) > 0 else None

    # per-stage arousal index: arousals whose onset falls in each stage, per hour
    # spent in that stage. Maps each onset (seconds) to its 30 s stage epoch.
    if stage_codes is not None and len(stage_codes) and events:
        st = np.rint(np.asarray(stage_codes, float)).astype(int)
        onset_ep = np.array([int((s / fs) // EPOCH_SEC) for s, _ in events])
        onset_ep = onset_ep[(onset_ep >= 0) & (onset_ep < st.size)]
        stage_of_onset = st[onset_ep]
        for code, tag in POOL_TAG.items():
            n_ev = int(np.count_nonzero(stage_of_onset == code))
            stage_hours = float(np.count_nonzero(st == code)) * EPOCH_SEC / 3600.0
            out[f"arousal_idx_{tag}"] = _r(n_ev / stage_hours, 2) if stage_hours > 0 else None
    return out


# ------------------------------------------------------------------ flat schema
def flatten_features(stage_codes, arousal_codes, arousal_fs):
    """One wide {col: value} row combining F1 + A1. Missing modality -> empty."""
    row = {}
    tf = transition_features(stage_codes)
    if tf.get("ok"):
        row.update({k: v for k, v in tf.items() if k != "ok"})
    af = arousal_features(arousal_codes, arousal_fs, stage_codes)
    if af.get("ok"):
        row.update({k: v for k, v in af.items() if k != "ok"})
    return row


def _build_columns():
    """Stable column list so the feature schema is fixed even for recordings
    missing a stream (missing values -> NaN)."""
    cols = []
    for ftag in TAG_ORDER:
        for ttag in TAG_ORDER:
            cols.append(f"trans_p_{ftag}_{ttag}")
    for ftag in TAG_ORDER:
        cols.append(f"trans_entropy_{ftag}")
    cols += ["trans_entropy_rate", "trans_stability_index"]
    cols += ["n_arousals", "arousal_dur_mean_s", "arousal_dur_median_s",
             "arousal_dur_sd_s", "arousal_dur_max_s", "arousal_iai_mean_s",
             "arousal_iai_median_s", "arousal_iai_cv"]
    cols += [f"arousal_idx_{tag}" for tag in TAG_ORDER]
    return cols


ARCH_FEATURE_COLUMNS = _build_columns()   # 25 + 5 + 2 + 8 + 5 = 45 features


def feature_vector(algo_data, arousal_fs):
    """Fixed-order (ARCH_FEATURE_COLUMNS) float32 arch feature vector for one
    recording, from the loaded CAISR annotation dict + arousal sampling rate.
    Any missing/failed value -> NaN. Never raises: degrades to an all-NaN row so
    the caller's schema stays stable regardless of stream availability."""
    names = list(ARCH_FEATURE_COLUMNS)
    try:
        stage = algo_data.get("stage_caisr") if algo_data else None
        arousal = algo_data.get("arousal_caisr") if algo_data else None
        row = flatten_features(stage, arousal, arousal_fs)
    except Exception:
        row = {}
    vec = np.array([row.get(c, np.nan) if row.get(c, None) is not None else np.nan
                    for c in names], dtype=np.float32)
    return vec, names
