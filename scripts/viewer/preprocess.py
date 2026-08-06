"""CAISR hypnogram preprocessing — remove biologically implausible rapid
stage transitions.

Automated per-epoch scorers (CAISR included) occasionally assign a sleep stage
to a single 30-second epoch sandwiched between two epochs of another stage. A
real sleep stage has inertia — it does not appear for 30 s and vanish — so these
single-epoch "spikes" are almost always scoring noise, and they inflate every
fragmentation / transition statistic.

Method: **minimum-bout-duration smoothing** (an iterative shortest-bout merge).

  1. Segment the stage sequence into bouts (maximal runs of one stage code).
  2. Repeatedly take the shortest *interior* bout whose length is below
     `min_bout_epochs` and relabel its epochs to its longer-lasting neighbour
     (ties -> the preceding neighbour). Merging shrinks the bout count, so the
     process converges.
  3. Only interior bouts (a neighbour on *both* sides) are merged, so a short
     Wake bout at sleep onset/offset — which is legitimate, not an implausible
     transition — is preserved.

Why not a numeric median filter? The stage codes are categorical, not ordinal
(REM=4 is not "between" N1 and Wake biologically), so a numeric median produces
nonsensical stages. Why not an HMM? CAISR gives hard labels, not emission
probabilities; without an emission model a hidden-Markov decode collapses to a
transition-smoothing prior — exactly this, but far less interpretable. The
rule-based smoother states its one assumption (a minimum plausible bout length)
explicitly.

Default `min_bout_epochs = 2` (1 minute) removes only single-epoch spikes — the
most conservative, defensible setting. Everything is shown alongside the raw
hypnogram in the viewer, so no smoothing is hidden.

Unknown (code 9) epochs are inert: they are never relabelled and never used as a
merge target, so the number of Unknown epochs is invariant and the raw/cleaned
stage percentages share the same denominator.
"""
import numpy as np

EPOCH_SEC = 30.0
STAGE_CODES = {1: "N3", 2: "N2", 3: "N1", 4: "REM", 5: "Wake", 9: "Unknown"}
# Hypnogram plotting order (y position): Wake top ... N3 bottom, REM between.
STAGE_Y = {"Wake": 4, "REM": 3, "N1": 2, "N2": 1, "N3": 0}
_PCT_STAGES = [1, 2, 3, 4, 5]     # stages counted for stage-% (Unknown excluded)


def _segments(codes):
    """Return [(start, end_exclusive, code), ...] contiguous runs in `codes`."""
    segs = []
    if len(codes) == 0:
        return segs
    start = 0
    for i in range(1, len(codes)):
        if codes[i] != codes[i - 1]:
            segs.append((start, i, int(codes[start])))
            start = i
    segs.append((start, len(codes), int(codes[start])))
    return segs


def smooth_stages(codes, min_bout_epochs=2):
    """Merge implausibly short *interior* stage bouts into their longer neighbour.

    Returns (cleaned_codes:int array, changed_mask:bool array). Deterministic and
    convergent (each merge strictly reduces the bout count).

    Unknown (code 9) is treated as **inert**: an Unknown bout is never relabelled
    (we don't invent a stage the scorer left unscorable), and a real stage is
    never merged *into* Unknown (we don't erase a scored epoch). A short bout is
    therefore only merged into a non-Unknown neighbour; if both neighbours are
    Unknown it is left alone. This keeps the count of Unknown epochs invariant, so
    the raw and cleaned stage-% share the same denominator.
    """
    out = np.asarray(codes, int).copy()
    if out.size < 3 or min_bout_epochs < 2:
        return out, np.zeros(out.size, bool)

    while True:
        segs = _segments(out)
        # interior bouts only: index 1 .. len-2 (both neighbours exist). Pick the
        # shortest sub-threshold, non-Unknown bout that has a non-Unknown neighbour.
        shortest = None                      # (k, length)
        for k in range(1, len(segs) - 1):
            s, e, code = segs[k]
            if code == 9:                    # never relabel Unknown epochs
                continue
            length = e - s
            if length >= min_bout_epochs:
                continue
            if segs[k - 1][2] == 9 and segs[k + 1][2] == 9:
                continue                     # no real stage to merge into
            if shortest is None or length < shortest[1]:
                shortest = (k, length)
        if shortest is None:
            break
        k = shortest[0]
        s, e, _ = segs[k]
        ls, le, lcode = segs[k - 1]
        rs, re, rcode = segs[k + 1]
        # relabel to the longer non-Unknown neighbour; tie -> preceding (left)
        left_ok, right_ok = lcode != 9, rcode != 9
        if left_ok and right_ok:
            chosen = lcode if (le - ls) >= (re - rs) else rcode
        else:
            chosen = lcode if left_ok else rcode
        out[s:e] = chosen

    changed = out != np.asarray(codes, int)
    return out, changed


def _stage_pct(codes):
    """Stage-% of scored (non-Unknown) epochs, keyed by stage name."""
    arr = np.asarray(codes, int)
    valid = arr[arr != 9]
    pct = {}
    if valid.size:
        for c in _PCT_STAGES:
            pct[STAGE_CODES[c]] = round(100.0 * int(np.count_nonzero(valid == c)) / valid.size, 1)
    return pct


def _hypnogram(codes):
    """List of {t_min, y, stage} points for the stepped hypnogram plot."""
    hyp = []
    for i, c in enumerate(np.asarray(codes, int)):
        name = STAGE_CODES.get(int(c), "Unknown")
        hyp.append({"t_min": round(i * EPOCH_SEC / 60.0, 2),
                    "y": STAGE_Y.get(name, -1), "stage": name})
    return hyp


def _n_transitions(codes):
    """Number of epoch-to-epoch stage changes over the whole sequence."""
    arr = np.asarray(codes, int)
    return int((np.diff(arr) != 0).sum()) if arr.size > 1 else 0


def staging_report(raw_codes, min_bout_epochs=2):
    """Build the raw + preprocessed staging payload for the viewer.

    Returns raw & cleaned hypnograms, stage-%, and a change summary. Event
    indices (AHI etc.) are derived from separate CAISR channels and are NOT
    affected, so they are computed elsewhere and left untouched.
    """
    raw = np.asarray(raw_codes, int)
    clean, changed = smooth_stages(raw, min_bout_epochs=min_bout_epochs)

    n = int(raw.size)
    n_changed = int(changed.sum())
    tr_raw, tr_clean = _n_transitions(raw), _n_transitions(clean)
    return {
        "min_bout_min": round(min_bout_epochs * EPOCH_SEC / 60.0, 2),
        "min_bout_epochs": int(min_bout_epochs),
        "method": "minimum-bout-duration smoothing (interior single-epoch spikes merged into longer neighbour)",
        "hypnogram": _hypnogram(raw),
        "hypnogram_clean": _hypnogram(clean),
        "stage_pct": _stage_pct(raw),
        "stage_pct_clean": _stage_pct(clean),
        "summary": {
            "n_epochs": n,
            "n_changed": n_changed,
            "pct_changed": round(100.0 * n_changed / n, 2) if n else 0.0,
            "transitions_raw": tr_raw,
            "transitions_clean": tr_clean,
            "transitions_removed": tr_raw - tr_clean,
        },
    }
