"""Per-patient sleep-stage dynamics for the viewer.

Computes, on demand from a single patient's CAISR stage channel, the same
transition matrix + fragmentation ("spikes") + bout statistics that the cohort
EDA produces (scripts/eda/stats_transitions.py) — and pairs every number with
the cohort baseline so a reader sees where this patient sits relative to the
whole release.

Self-contained on purpose: the viewer is deployed as a standalone bundle
without the `eda` package, so the algorithm is duplicated here rather than
imported. The COHORT_* baselines are the pooled means measured over all 1090
CAISR recordings (eda/transitions.json); keep them in sync if the EDA rerun
changes them.
"""
import numpy as np

EPOCH_SEC = 30.0                       # stage epochs are 30 s
STAGE_ORDER = [5, 3, 2, 1, 4]          # Wake, N1, N2, N3, REM (matrix row/col order)
STAGE_LABEL = {5: "Wake", 3: "N1", 2: "N2", 1: "N3", 4: "REM"}
STAGE_NAMES = [STAGE_LABEL[c] for c in STAGE_ORDER]
ASLEEP = {1, 2, 3, 4}                  # everything but Wake
IDX = {code: i for i, code in enumerate(STAGE_ORDER)}
_NAME2CODE = {"Wake": 5, "N1": 3, "N2": 2, "N3": 1, "REM": 4}

# Clinically salient directed transitions tracked as per-hour-of-sleep rates.
NAMED_TRANSITIONS = [
    ("N2", "N3", "Deepening (N2→N3)"),
    ("N3", "N2", "Lightening (N3→N2)"),
    ("N2", "REM", "Into REM (N2→REM)"),
    ("REM", "Wake", "REM→Wake"),
    ("N1", "Wake", "N1→Wake"),
    ("N2", "Wake", "N2→Wake"),
]

# Fragmentation fields: (key, label, unit, higher_is_worse). higher_is_worse
# drives the red/green comparison arrow vs cohort; None = neutral (context only).
FRAG_FIELDS = [
    ("n_awakenings",           "Awakenings",              "count",  True),
    ("awakenings_per_hr_sleep","Awakenings",              "/h sleep", True),
    ("brief_wake_intrusions",  "Brief wake intrusions",   "count",  True),
    ("stage_shift_index",      "Stage-shift index",       "/h sleep", True),
    ("single_epoch_spikes",    "Single-epoch stage spikes","count", True),
    ("spikes_per_hr_sleep",    "Stage spikes",            "/h sleep", True),
    ("n_wake_bouts",           "Wake bouts",              "count",  True),
    ("n_sleep_bouts",          "Sleep bouts",             "count",  None),
    ("mean_sleep_bout_min",    "Mean sleep-bout length",  "min",    False),
    ("mean_wake_bout_min",     "Mean wake-bout length",   "min",    True),
    ("n_rem_periods",          "REM periods",             "count",  None),
]

# --- cohort baseline (pooled mean over 1090 recordings; eda/transitions.json) ---
COHORT_FRAG = {
    "n_awakenings": 26.03, "awakenings_per_hr_sleep": 5.26,
    "brief_wake_intrusions": 12.01, "stage_shift_index": 19.15,
    "single_epoch_spikes": 22.15, "spikes_per_hr_sleep": 4.35,
    "n_wake_bouts": 26.97, "n_sleep_bouts": 26.22,
    "mean_sleep_bout_min": 16.10, "mean_wake_bout_min": 5.84,
    "n_rem_periods": 6.32,
    "trans_N2_to_N3_per_hr": 1.412, "trans_N3_to_N2_per_hr": 1.276,
    "trans_N2_to_REM_per_hr": 0.716, "trans_REM_to_Wake_per_hr": 0.516,
    "trans_N1_to_Wake_per_hr": 2.207, "trans_N2_to_Wake_per_hr": 2.411,
}
# Cohort-mean transition-probability matrix, rows/cols in STAGE_ORDER.
COHORT_MATRIX = [
    [0.857, 0.123, 0.015, 0.001, 0.004],   # from Wake
    [0.157, 0.523, 0.293, 0.000, 0.028],   # from N1
    [0.032, 0.014, 0.924, 0.020, 0.010],   # from N2
    [0.012, 0.000, 0.143, 0.843, 0.001],   # from N3
    [0.044, 0.025, 0.031, 0.000, 0.901],   # from REM
]


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


def _bouts(mask):
    """Lengths (in epochs) of each contiguous True run in `mask`."""
    if mask.size == 0:
        return []
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return [(idx[i + 1] - idx[i]) for i in range(0, len(idx), 2)]


def _recording_metrics(stage):
    """Fragmentation metrics + count matrix from a raw stage array.

    Returns (metrics_dict, count_matrix_5x5) or (None, None) if unusable. This
    mirrors scripts/eda/stats_transitions.recording_dynamics exactly so the
    per-patient numbers match the cohort EDA.
    """
    if stage is None or len(stage) == 0:
        return None, None
    st = np.rint(np.asarray(stage, float)).astype(int)
    seq = st[st != 9]                              # drop Unknown for dynamics
    if seq.size < 2:
        return None, None

    counts = transition_matrix(seq)
    asleep_mask = np.isin(seq, list(ASLEEP))
    n_asleep = int(asleep_mask.sum())
    tst_hours = n_asleep * EPOCH_SEC / 3600.0
    total_transitions = int((np.diff(seq) != 0).sum())

    d = {}
    onset = int(np.flatnonzero(asleep_mask)[0]) if n_asleep else None
    awakenings, brief_wake = 0, 0
    if onset is not None:
        for i in range(onset, seq.size - 1):
            if seq[i] in ASLEEP and seq[i + 1] == 5:
                awakenings += 1
        for i in range(onset + 1, seq.size - 1):
            if seq[i] == 5 and seq[i - 1] in ASLEEP and seq[i + 1] in ASLEEP:
                brief_wake += 1
    d["n_awakenings"] = awakenings
    d["awakenings_per_hr_sleep"] = round(awakenings / tst_hours, 2) if tst_hours > 0 else None
    d["brief_wake_intrusions"] = brief_wake
    d["stage_shift_index"] = round(total_transitions / tst_hours, 2) if tst_hours > 0 else None

    spikes = 0
    for i in range(1, seq.size - 1):
        if seq[i] != seq[i - 1] and seq[i] != seq[i + 1] and seq[i - 1] == seq[i + 1]:
            spikes += 1
    d["single_epoch_spikes"] = spikes
    d["spikes_per_hr_sleep"] = round(spikes / tst_hours, 2) if tst_hours > 0 else None

    wake_bouts = _bouts(seq == 5)
    sleep_bouts = _bouts(asleep_mask)
    d["n_wake_bouts"] = len(wake_bouts)
    d["n_sleep_bouts"] = len(sleep_bouts)
    d["mean_sleep_bout_min"] = round(float(np.mean(sleep_bouts)) * EPOCH_SEC / 60.0, 2) if sleep_bouts else None
    d["mean_wake_bout_min"] = round(float(np.mean(wake_bouts)) * EPOCH_SEC / 60.0, 2) if wake_bouts else None
    d["n_rem_periods"] = len(_bouts(seq == 4))

    for a, b, _ in NAMED_TRANSITIONS:
        n = int(counts[IDX[_NAME2CODE[a]], IDX[_NAME2CODE[b]]])
        d[f"trans_{a}_to_{b}_per_hr"] = round(n / tst_hours, 3) if tst_hours > 0 else None

    d["tst_hours"] = round(tst_hours, 2)
    return d, counts


def patient_dynamics(stage):
    """Full per-patient dynamics payload for the viewer, JSON-serialisable.

    { ok, stage_order, transition_pct[5][5], transition_counts[5][5],
      cohort_pct[5][5], fragmentation[], named_transitions[] }
    """
    metrics, counts = _recording_metrics(stage)
    if metrics is None:
        return {"ok": False, "error": "no usable staged sleep for this recording"}

    probs = _row_normalise(counts)
    pct = (probs * 100.0).round(1).tolist()
    cohort_pct = [[round(v * 100.0, 1) for v in row] for row in COHORT_MATRIX]

    frag = []
    for key, label, unit, worse in FRAG_FIELDS:
        frag.append({
            "key": key, "label": label, "unit": unit,
            "value": metrics.get(key),
            "cohort": COHORT_FRAG.get(key),
            "higher_is_worse": worse,
        })

    named = []
    for a, b, label in NAMED_TRANSITIONS:
        key = f"trans_{a}_to_{b}_per_hr"
        named.append({
            "key": key, "label": label, "unit": "/h sleep",
            "value": metrics.get(key),
            "cohort": COHORT_FRAG.get(key),
        })

    return {
        "ok": True,
        "stage_order": STAGE_NAMES,
        "transition_pct": pct,
        "transition_counts": counts.astype(int).tolist(),
        "cohort_pct": cohort_pct,
        "tst_hours": metrics.get("tst_hours"),
        "fragmentation": frag,
        "named_transitions": named,
    }
