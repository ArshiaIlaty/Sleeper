"""Sleep-architecture transition Markov features + microarousal distributions.

Two CAISR-annotation-only feature families (no waveform decode, so they are cheap
enough to run over the whole cohort in minutes):

  * **F1 — stage-transition Markov model.** The full 5x5 row-normalised
    transition-probability matrix P(next stage | current stage) from the CAISR
    stage stream, plus the information-theoretic summaries that are the actual
    signal: per-row conditional entropy (how unpredictable the next stage is from
    each stage), the occupancy-weighted **entropy rate** (overall architecture
    disorder), and the occupancy-weighted **stability index** (mean self-transition
    probability). Our EDA found the discriminative axis is architecture
    *instability*, not night averages — this exposes that axis as per-recording
    features. Complements dynamics.py, which gives per-hour *rates* of six named
    transitions but not the normalised probabilities or their entropies.

  * **A1 — microarousal distributions.** Beyond the single `arousal_index`
    (events/hour) we already export, this derives the **event-duration
    distribution** (mean/median/sd/max seconds), the **inter-arousal-interval
    distribution** (mean/median/CV of the gaps between successive arousals — a
    high CV means clustered/periodic arousal, a marker of unstable sleep), and the
    **per-stage arousal index** (arousals per hour spent in each stage). The
    arousal stream's sampling rate is read from the EDF (do NOT assume 1/2 Hz — it
    varies), so durations and onsets are in real seconds.

Reuses dynamics.transition_matrix / _row_normalise so the counting matches the
per-patient viewer and the cohort EDA exactly. Every value is JSON-serialisable
and every failure mode degrades to None rather than raising, so it is safe to run
across a whole cohort unattended.
"""
import numpy as np

from dynamics import transition_matrix, _row_normalise, STAGE_ORDER

EPOCH_SEC = 30.0
# Stage code -> lower-case tag used in feature-column names (CAISR coding).
POOL_TAG = {5: "wake", 3: "n1", 2: "n2", 1: "n3", 4: "rem"}
TAG_ORDER = [POOL_TAG[c] for c in STAGE_ORDER]      # wake, n1, n2, n3, rem
AROUSAL_CODE = 1                                     # code 1 == arousal in arousal_caisr


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
    """Full transition-probability matrix + entropies from per-epoch stage codes.

    Returns {"ok", plus flat keys}:
      trans_p_{from}_{to}     -- 25 row-normalised transition probabilities
      trans_entropy_{from}    -- per-row conditional entropy (normalised [0,1])
      trans_entropy_rate      -- occupancy-weighted mean row entropy
      trans_stability_index   -- occupancy-weighted mean self-transition prob
    Uses raw codes (single-epoch spikes ARE fragmentation), matching dynamics.py.
    """
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
    # occupancy-weighted overall disorder + persistence, over rows we could measure
    m = np.isfinite(ent) & (occ > 0)
    out["trans_entropy_rate"] = _r(float(np.sum(occ[m] * ent[m]) / occ[m].sum())) if m.any() else None
    diag = np.diag(probs)
    md = (row_tot > 0)
    out["trans_stability_index"] = _r(float(np.sum(occ[md] * diag[md]) / occ[md].sum())) if md.any() else None
    return out


# --------------------------------------------------------------------------- A1
def arousal_features(arousal_codes, arousal_fs, stage_codes=None):
    """Microarousal duration + inter-arousal-interval distributions, and per-stage
    arousal index, from the arousal_caisr stream.

    arousal_codes: int label stream (code 1 == arousal).
    arousal_fs:    the stream's sampling rate in Hz (read from the EDF, not assumed).
    stage_codes:   optional per-epoch (30 s) stage codes, for the per-stage index.
    """
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
    """Stable column list so the CSV header is fixed even for recordings missing a
    stream (missing values -> empty)."""
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


ARCH_FEATURE_COLUMNS = _build_columns()
