"""Detailed sleep-stage dynamics from CAISR staging: transition matrices,
fragmentation ("spikes"), and bout structure.

For each recording we collapse the 30 s stage series to its *epoch* sequence
(dropping Unknown=9), then compute:

  - **Transition matrix** P(X -> Y): probability that an epoch in stage X is
    immediately followed by stage Y (rows sum to 1). We report per-recording
    matrices aggregated to a cohort-mean matrix, overall and split by CI label.
  - **Directed transition rates** per hour of sleep for the clinically salient
    moves (e.g. -> Wake = awakenings, N2->N3 = deepening, REM->Wake).
  - **Fragmentation / "spikes":** number of awakenings (sleep -> Wake), the
    sleep-stage shift index (all stage changes per hour of sleep), wake-bout
    and sleep-bout counts and mean durations, the count of distinct REM periods,
    and the number of brief (single-epoch) sleep intrusions.

Only the CAISR *stage* channel is needed; annotation EDFs are small so callers
read them in full. Stage codes: {1:N3, 2:N2, 3:N1, 4:REM, 5:Wake, 9:Unknown}.
"""
import os
from collections import defaultdict

import numpy as np
import edfio

from common import (CAISR_DIR, SITE_NAMES, list_sites, list_edfs,
                    parse_record_id, EPOCH_SEC)
from statutils import numeric_summary

# Stage order used for every matrix row/column (Unknown excluded).
STAGE_ORDER = [5, 3, 2, 1, 4]                 # Wake, N1, N2, N3, REM
STAGE_LABEL = {5: "Wake", 3: "N1", 2: "N2", 1: "N3", 4: "REM"}
ASLEEP = {1, 2, 3, 4}                          # everything but Wake
IDX = {code: i for i, code in enumerate(STAGE_ORDER)}

# Named directed transitions we track as per-hour rates (clinically meaningful).
NAMED_TRANSITIONS = [
    ("N2", "N3", "deepening (N2->N3)"),
    ("N3", "N2", "lightening (N3->N2)"),
    ("N2", "REM", "N2->REM"),
    ("REM", "Wake", "REM->Wake"),
    ("N1", "Wake", "N1->Wake"),
    ("N2", "Wake", "N2->Wake"),
]
_NAME2CODE = {"Wake": 5, "N1": 3, "N2": 2, "N3": 1, "REM": 4}


def transition_matrix(seq):
    """5x5 count matrix over consecutive epoch pairs in `seq` (codes, no 9)."""
    m = np.zeros((5, 5), dtype=float)
    for a, b in zip(seq[:-1], seq[1:]):
        if a in IDX and b in IDX:
            m[IDX[a], IDX[b]] += 1.0
    return m


def _row_normalise(counts):
    """Row-normalise a count matrix to transition probabilities (rows summing to
    1); rows with no outgoing transitions stay all-zero."""
    out = np.zeros_like(counts)
    rs = counts.sum(axis=1)
    nz = rs > 0
    out[nz] = counts[nz] / rs[nz, None]
    return out


def _bouts(mask):
    """Return the lengths (in epochs) of each contiguous True run in `mask`."""
    if mask.size == 0:
        return []
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return [(idx[i + 1] - idx[i]) for i in range(0, len(idx), 2)]


def recording_dynamics(stage):
    """Per-recording transition + fragmentation metrics from a raw stage array.

    Returns (metrics_dict, count_matrix_5x5). metrics_dict is None if there is
    no usable staged sleep.
    """
    if stage is None or len(stage) == 0:
        return None, None
    st = np.rint(np.asarray(stage, float)).astype(int)
    seq = st[st != 9]                          # drop Unknown for dynamics
    if seq.size < 2:
        return None, None

    counts = transition_matrix(seq)
    probs = _row_normalise(counts)

    asleep_mask = np.isin(seq, list(ASLEEP))
    n_asleep = int(asleep_mask.sum())
    tst_hours = n_asleep * EPOCH_SEC / 3600.0
    total_transitions = int((np.diff(seq) != 0).sum())

    d = {}
    # --- fragmentation / "spikes" ---
    # awakenings: any asleep -> Wake transition after onset
    onset = int(np.flatnonzero(asleep_mask)[0]) if n_asleep else None
    awakenings = 0
    brief_wake = 0            # single-epoch wake intrusions between sleep
    if onset is not None:
        for i in range(onset, seq.size - 1):
            if seq[i] in ASLEEP and seq[i + 1] == 5:
                awakenings += 1
        # single-epoch wake surrounded by sleep = micro-arousal-like "spike"
        for i in range(onset + 1, seq.size - 1):
            if seq[i] == 5 and seq[i - 1] in ASLEEP and seq[i + 1] in ASLEEP:
                brief_wake += 1
    d["n_awakenings"] = awakenings
    d["awakenings_per_hr_sleep"] = round(awakenings / tst_hours, 2) if tst_hours > 0 else None
    d["brief_wake_intrusions"] = brief_wake

    # stage-shift index: all stage changes per hour of sleep (a fragmentation
    # measure closely related to, but finer than, transitions_per_hr)
    d["stage_shift_index"] = round(total_transitions / tst_hours, 2) if tst_hours > 0 else None

    # single-epoch sleep-stage intrusions (a stage present for exactly 1 epoch
    # between two epochs of a different single stage) — rapid "spikes"
    spikes = 0
    for i in range(1, seq.size - 1):
        if seq[i] != seq[i - 1] and seq[i] != seq[i + 1] and seq[i - 1] == seq[i + 1]:
            spikes += 1
    d["single_epoch_spikes"] = spikes
    d["spikes_per_hr_sleep"] = round(spikes / tst_hours, 2) if tst_hours > 0 else None

    # bout structure
    wake_bouts = _bouts(seq == 5)
    sleep_bouts = _bouts(asleep_mask)
    d["n_wake_bouts"] = len(wake_bouts)
    d["n_sleep_bouts"] = len(sleep_bouts)
    d["mean_sleep_bout_min"] = round(float(np.mean(sleep_bouts)) * EPOCH_SEC / 60.0, 2) if sleep_bouts else None
    d["mean_wake_bout_min"] = round(float(np.mean(wake_bouts)) * EPOCH_SEC / 60.0, 2) if wake_bouts else None
    # distinct REM periods = number of contiguous REM runs
    d["n_rem_periods"] = len(_bouts(seq == 4))

    # named directed transition rates per hour of sleep
    for a, b, key in NAMED_TRANSITIONS:
        ca, cb = _NAME2CODE[a], _NAME2CODE[b]
        n = int(counts[IDX[ca], IDX[cb]])
        d[f"trans_{a}_to_{b}_per_hr"] = round(n / tst_hours, 3) if tst_hours > 0 else None

    return d, counts


# Numeric fragmentation fields to append to the per-recording table + aggregate.
DYNAMIC_FIELDS = [
    "n_awakenings", "awakenings_per_hr_sleep", "brief_wake_intrusions",
    "stage_shift_index", "single_epoch_spikes", "spikes_per_hr_sleep",
    "n_wake_bouts", "n_sleep_bouts", "mean_sleep_bout_min", "mean_wake_bout_min",
    "n_rem_periods",
] + [f"trans_{a}_to_{b}_per_hr" for a, b, _ in NAMED_TRANSITIONS]


def _load_stage(path):
    edf = edfio.read_edf(path, lazy_load_data=False)
    for s in edf.signals:
        if s.label.strip() == "stage_caisr":
            return np.asarray(s.data, float)
    return None


def _matrix_summary(mat_list):
    """Mean transition-probability matrix over a list of per-recording count
    matrices (each row-normalised first, so every recording weighs equally),
    plus the pooled matrix (all counts summed then normalised)."""
    if not mat_list:
        return None
    per_rec = np.stack([_row_normalise(m) for m in mat_list])
    # average only over rows that had data in each recording? Simpler + robust:
    # mean of per-recording normalised matrices, ignoring all-zero rows via nan.
    stacked = per_rec.copy()
    rowsums = np.stack([m.sum(axis=1) for m in mat_list])         # (R,5)
    stacked[rowsums == 0] = np.nan
    with np.errstate(invalid="ignore"):
        mean_mat = np.nanmean(stacked, axis=0)
    pooled = _row_normalise(np.sum(mat_list, axis=0))
    return {
        "stage_order": [STAGE_LABEL[c] for c in STAGE_ORDER],
        "mean_matrix": np.nan_to_num(mean_mat).round(4).tolist(),
        "pooled_matrix": pooled.round(4).tolist(),
        "n_recordings": len(mat_list),
    }


def run(limit_per_site=None, label_lookup=None, progress=None):
    """Compute cohort transition matrices + fragmentation aggregates.

    label_lookup: optional {(bids, session): 0/1} for CI-stratified matrices.
    """
    pooled = defaultdict(list)
    mats_all, mats_ci, mats_noci = [], [], []
    n_ok = 0
    n_files = 0
    errors = []

    for site in list_sites(CAISR_DIR):
        files = list_edfs(CAISR_DIR, site)
        if limit_per_site:
            files = files[:limit_per_site]
        for i, f in enumerate(files):
            n_files += 1
            try:
                stage = _load_stage(f)
                metrics, counts = recording_dynamics(stage)
            except Exception as e:
                errors.append({"file": os.path.basename(f), "error": str(e)})
                continue
            if metrics is None:
                continue
            n_ok += 1
            for k in DYNAMIC_FIELDS:
                v = metrics.get(k)
                pooled[k].append(v if (v is not None and np.isfinite(v)) else np.nan)
            mats_all.append(counts)
            if label_lookup is not None:
                bids, sess = parse_record_id(f.replace("_caisr_annotations.edf", ".edf"))
                lab = label_lookup.get((bids, sess))
                if lab == 1:
                    mats_ci.append(counts)
                elif lab == 0:
                    mats_noci.append(counts)
            if progress and (i + 1) % 100 == 0:
                progress(f"  {site}: {i+1}/{len(files)} transition scans")

    out = {
        "n_files": n_files,
        "n_ok": n_ok,
        "n_errors": len(errors),
        "errors": errors[:20],
        "pooled_fragmentation": {k: numeric_summary(pooled[k]) for k in DYNAMIC_FIELDS},
        "transition_matrix_all": _matrix_summary(mats_all),
    }
    if label_lookup is not None:
        out["transition_matrix_ci"] = _matrix_summary(mats_ci)
        out["transition_matrix_noci"] = _matrix_summary(mats_noci)
    return out


if __name__ == "__main__":
    import json, sys
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print(json.dumps(run(limit_per_site=lim, progress=lambda m: print(m, file=sys.stderr)),
                     indent=2, default=str))
