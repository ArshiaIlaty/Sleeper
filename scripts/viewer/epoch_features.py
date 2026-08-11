"""Per-epoch (LONG) signal features — the fully un-aggregated view.

Where `stage_dispersion.py` reports the *spread* of a stage's per-epoch values and
the mean-only CSVs report their *mean*, this module emits the raw per-epoch values
themselves: one record per (recording, stage, epoch). It is the "long" companion
to the "wide" per-recording CSVs, for EDA, plotting distributions, and
sequence/temporal models that want the within-night trajectory rather than a
summary.

It reuses the exact low-level primitives the aggregating extractors use
(`eeg_spectral._epoch_band_powers`, the NeuroKit entropy calls, the spindle
detector), so an epoch's value here is identical to the value that was averaged /
dispersed elsewhere — verified by construction.

Three grains are produced (each its own generator; the exporter writes one CSV per
grain):
  * EEG SPECTRAL per epoch  — every 30 s epoch of a known stage: absolute + relative
    band power (delta/theta/alpha/sigma/beta) and the Theta/Alpha, Delta/Sigma,
    REM-slowing ratios. (~1 ms/epoch, so ALL epochs are emitted, no subsample.)
  * EEG COMPLEXITY per epoch — sample & permutation entropy. entropy_sample is
    O(n^2), so exactly like the aggregating extractor this is computed on a capped,
    evenly-spaced SUBSAMPLE of each stage's epochs (default <=25/stage); the
    exporter left-joins it onto the spectral rows by epoch index (blank where an
    epoch was not in the subsample).
  * SPINDLE EVENTS — a different grain (one row per detected spindle, not per
    epoch): stage, onset time, duration, peak sigma-envelope amplitude. This is the
    raw form the spindle amp_mean / dur_mean were computed from.

Respiratory rate is deliberately NOT emitted per epoch: `nk.rsp_rate` runs on the
*concatenated* epochs of a stage and returns an instantaneous-rate series that does
not align one-to-one with 30 s epochs, so there is no honest per-epoch value.

Staging is aligned to the signal by epoch index (epoch i = samples
[i*spe:(i+1)*spe], spe = round(fs*30)); the shorter length wins. Unknown (code 9)
epochs are skipped. Pure numpy + the same optional scipy/neurokit deps; degrades to
an empty list/dict rather than raising.
"""
import warnings

import numpy as np

import eeg_spectral as es
import nk_features as nkf

EPOCH_SEC = 30.0
STAGE_ORDER = [5, 3, 2, 1, 4]
POOL_TAG = {5: "wake", 3: "n1", 2: "n2", 1: "n3", 4: "rem"}
BANDS = ["delta", "theta", "alpha", "sigma", "beta"]


def _r(x, nd):
    try:
        v = float(x)
        return round(v, nd) if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------- EEG spectral
def eeg_spectral_epochs(eeg, fs, stage_codes):
    """Yield one dict per known-stage epoch with its band powers + ratios.

    Emits EVERY epoch of wake/n1/n2/n3/rem (Welch PSD is ~1 ms/epoch, so no cap).
    Each dict: {stage, epoch_index, start_s, abs_<band>, rel_<band>, theta_alpha,
    delta_sigma, rem_slowing}. Skips epochs with an unusable PSD (returns nothing
    for them) and unknown (code 9) epochs.
    """
    eeg = np.asarray(eeg, float)
    fs = float(fs)
    codes = np.rint(np.asarray(stage_codes, float)).astype(int)
    if not es._SCIPY_OK or fs <= 0 or eeg.size == 0 or codes.size == 0:
        return []
    spe, n_ep, codes = es._align_epochs(eeg.size, fs, codes)
    if spe is None or n_ep < 1 or fs < 2 * es._TOTAL_BAND[1]:
        return []

    rows = []
    for e in range(n_ep):
        code = int(codes[e])
        if code not in POOL_TAG:                 # skip unknown / out-of-range
            continue
        a, r = es._epoch_band_powers(eeg[e * spe:(e + 1) * spe], fs)
        if a is None:
            continue
        fast = a["alpha"] + a["sigma"] + a["beta"]
        slow = a["delta"] + a["theta"]
        row = {
            "stage": POOL_TAG[code],
            "epoch_index": e,
            "start_s": round(e * EPOCH_SEC, 1),
        }
        for b in BANDS:
            row[f"abs_{b}"] = _r(a[b], 3)
            row[f"rel_{b}"] = _r(r[b], 5)
        row["theta_alpha"] = _r(a["theta"] / a["alpha"], 4) if a["alpha"] > 0 else None
        row["delta_sigma"] = _r(a["delta"] / a["sigma"], 4) if a["sigma"] > 0 else None
        row["rem_slowing"] = _r(slow / fast, 4) if fast > 0 else None
        rows.append(row)
    return rows


# --------------------------------------------------------------- EEG complexity
def eeg_complexity_epochs(eeg, fs, stage_codes,
                          max_epochs=nkf.EEG_MAX_EPOCHS,
                          ep_samples=nkf.EEG_EPOCH_SAMPLES):
    """Return {epoch_index: {"sampen":.., "permen":.., "stage":..}} for the
    capped, evenly-spaced per-stage SUBSAMPLE (same selection as
    nk_features.eeg_complexity_by_stage). Keyed by absolute epoch index so the
    exporter can left-join it onto the spectral rows.
    """
    eeg = np.asarray(eeg, float)
    fs = float(fs)
    codes = np.rint(np.asarray(stage_codes, float)).astype(int)
    out = {}
    if not nkf._NK_OK or fs <= 0 or eeg.size == 0 or codes.size == 0:
        return out
    spe, n_ep, codes = nkf._align_epochs(eeg.size, fs, codes)
    if spe is None or n_ep < 1:
        return out

    for code in STAGE_ORDER:
        idx = np.where(codes == code)[0]
        if idx.size == 0:
            continue
        if idx.size > max_epochs:
            idx = idx[np.linspace(0, idx.size - 1, max_epochs).astype(int)]
        for e in idx:
            seg = eeg[e * spe:(e + 1) * spe]
            seg = seg[np.isfinite(seg)]
            if seg.size < spe // 2:
                continue
            seg = nkf._decimate_to(seg, ep_samples)
            seg = seg - seg.mean()
            se = pe = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    s, _ = nkf.nk.entropy_sample(seg)
                    p, _ = nkf.nk.entropy_permutation(seg)
                se = _r(s, 4) if np.isfinite(s) else None
                pe = _r(p, 4) if np.isfinite(p) else None
            except Exception:
                pass
            if se is not None or pe is not None:
                out[int(e)] = {"sampen": se, "permen": pe, "stage": POOL_TAG[code]}
    return out


# --------------------------------------------------------------- spindle events
def spindle_events(eeg, fs, stage_codes, k=es.SPINDLE_K):
    """One dict per detected sleep spindle in N2/N3.

    Uses eeg_spectral's exact detector (sigma-band envelope, mean+k*SD threshold
    calibrated on pooled NREM, 0.5-3 s duration, merge gap). Each dict:
    {stage, start_s, duration_s, amp_uv}. Returns (events, threshold_uv).
    """
    eeg = np.asarray(eeg, float)
    fs = float(fs)
    codes = np.rint(np.asarray(stage_codes, float)).astype(int)
    if not es._SCIPY_OK or fs <= 0 or eeg.size == 0 or codes.size == 0:
        return [], None
    if fs < 2 * es.SPINDLE_BAND[1]:
        return [], None
    spe, n_ep, codes = es._align_epochs(eeg.size, fs, codes)
    if spe is None or n_ep < 1:
        return [], None

    x = eeg[:n_ep * spe].astype(float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        env = es._bandpass_env(x, fs, es.SPINDLE_BAND)
    except Exception:
        return [], None
    if env is None:
        return [], None

    sample_stage = np.repeat(codes[:n_ep], spe)[:env.size]
    nrem_mask = np.isin(sample_stage, (2, 1))
    if nrem_mask.sum() < fs * 60:
        return [], None
    base = env[nrem_mask]
    thr = float(base.mean() + k * base.std())
    min_len = int(es.SPINDLE_MIN_S * fs)
    max_len = int(es.SPINDLE_MAX_S * fs)
    merge_gap = int(es.SPINDLE_MERGE_GAP_S * fs)

    events = []
    for code in (2, 1):                          # N2 (primary), N3
        idx = np.where(codes[:n_ep] == code)[0]
        minutes = idx.size * EPOCH_SEC / 60.0
        if minutes < es.MIN_STAGE_MIN_FOR_SPINDLE:
            continue
        stage_mask = np.isin(sample_stage, (code,))
        above = (env > thr) & stage_mask
        for a, b in es._detect_bursts(above, min_len, max_len, merge_gap=merge_gap):
            events.append({
                "stage": POOL_TAG[code],
                "start_s": round(a / fs, 2),
                "duration_s": round((b - a) / fs, 3),
                "amp_uv": round(float(env[a:b].max()), 2),
            })
    events.sort(key=lambda ev: ev["start_s"])
    return events, _r(thr, 2)
