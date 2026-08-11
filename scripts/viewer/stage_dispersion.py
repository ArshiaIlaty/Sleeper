"""Within-stage DISPERSION of the per-epoch signal features.

The webapp report and the two waveform exporters (`export_nk_features.py`,
`export_report_features.py`) summarise each sleep stage by the *mean* of its
per-epoch values — one central number per stage. That average throws away the
*within-night variability*, which is itself a cognitive-impairment signal (the
"blunted / more erratic modulation" thesis): two nights with the same mean N2
sigma power can differ sharply in how stable that power is epoch to epoch.

This module recomputes the same per-epoch quantities the existing extractors do,
but instead of collapsing them to a mean it reports their **dispersion across
epochs** — SD, coefficient of variation (CV = SD/|mean|, scale-free), and, for the
flagship EEG ratios, the 10th/50th/90th percentiles and IQR. It is the *non-avg*
companion to the existing CSVs, keyed the same way so it merges on
`(bids_folder, session)`.

Covered (everything that was mean-only or would benefit from spread):
  * EEG spectral   — relative band power per band, and the Theta/Alpha, Delta/Sigma,
                     REM-slowing ratios, per stage (dispersion across epochs).
  * EEG complexity — sample & permutation entropy per stage (dispersion across the
                     analysed epochs).
  * Respiratory    — instantaneous respiratory rate per stage (dispersion across
                     the stage's rate series).
  * Spindles       — spindle amplitude & duration per stage (dispersion across the
                     detected spindles in that stage).

Reuses the low-level per-epoch primitives from `eeg_spectral` / `nk_features`
verbatim (same PSD, same entropy calls, same rate pipeline, same detector), so a
mean recomputed here equals the mean in the existing CSVs — only the summary
statistic differs. Pure numpy + the same optional scipy/neurokit deps; every field
degrades to None rather than raising, safe to run unattended across a cohort.
"""
import warnings

import numpy as np

import eeg_spectral as es
import nk_features as nkf

EPOCH_SEC = 30.0
STAGE_ORDER = [5, 3, 2, 1, 4]
POOL_TAG = {5: "wake", 3: "n1", 2: "n2", 1: "n3", 4: "rem"}
STAGE_TAGS = ["wake", "n1", "n2", "n3", "rem"]
SPINDLE_TAGS = ["n2", "n3"]
BANDS = ["delta", "theta", "alpha", "sigma", "beta"]

# dispersion stat sets ---------------------------------------------------------
# compact stats (SD + CV) for the many per-band columns; full stats (SD/CV +
# percentiles + IQR) for the handful of flagship ratios / entropies / rates.
STATS_COMPACT = ["sd", "cv"]
STATS_FULL = ["sd", "cv", "p10", "p50", "p90", "iqr"]


def _finite(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _r(x, nd):
    v = _finite(x)
    return None if v is None else round(v, nd)


def _disp(vals, nd=4, full=False):
    """Dispersion stats for a list of per-epoch values.

    Returns a dict with keys from STATS_FULL (full=True) or STATS_COMPACT
    (full=False), plus always `n` and `mean` (mean kept only to verify it matches
    the averaged CSVs, and to derive CV). Non-finite values are dropped; with fewer
    than 2 finite values every spread stat is None.
    """
    v = np.asarray([x for x in (_finite(u) for u in vals) if x is not None], float)
    stats = STATS_FULL if full else STATS_COMPACT
    out = {"n": int(v.size), "mean": _r(v.mean(), nd) if v.size else None}
    for s in stats:
        out[s] = None
    if v.size >= 2:
        m = float(v.mean())
        sd = float(v.std(ddof=1))
        vals_map = {
            "sd": sd,
            "cv": (sd / abs(m)) if m != 0 else None,
            "p10": float(np.percentile(v, 10)),
            "p50": float(np.percentile(v, 50)),
            "p90": float(np.percentile(v, 90)),
            "iqr": float(np.percentile(v, 75) - np.percentile(v, 25)),
        }
        for s in stats:
            out[s] = _r(vals_map[s], nd)
    return out


# --------------------------------------------------------------- EEG spectral
def eeg_spectral_dispersion(eeg, fs, stage_codes, max_epochs=es.MAX_BP_EPOCHS):
    """Per-stage dispersion of relative band power + the three flagship ratios.

    For each stage we recompute every epoch's band powers with the SAME
    `eeg_spectral._epoch_band_powers` used by the averaged CSV, then report the
    spread across those epochs (rather than their mean). Relative power and ratios
    are used because they are scale-invariant (site-robust); absolute power is
    intentionally left to the mean-only CSV.

    Returns {"ok", "stages": {tag: {rel_<band>: {..}, theta_alpha:{..}, ...}}}.
    """
    eeg = np.asarray(eeg, float)
    fs = float(fs)
    codes = np.rint(np.asarray(stage_codes, float)).astype(int)
    if not es._SCIPY_OK or fs <= 0 or eeg.size == 0 or codes.size == 0:
        return {"ok": False, "error": "no usable EEG / staging / scipy"}
    spe, n_ep, codes = es._align_epochs(eeg.size, fs, codes)
    if spe is None or n_ep < 1 or fs < 2 * es._TOTAL_BAND[1]:
        return {"ok": False, "error": "signal too short or sampling rate too low"}

    stages = {}
    for code in STAGE_ORDER:
        idx = np.where(codes == code)[0]
        if idx.size == 0:
            continue
        if idx.size > max_epochs:
            idx = idx[np.linspace(0, idx.size - 1, max_epochs).astype(int)]
        # per-epoch series
        rel = {b: [] for b in BANDS}
        theta_alpha, delta_sigma, rem_slowing = [], [], []
        for e in idx:
            a, r = es._epoch_band_powers(eeg[e * spe:(e + 1) * spe], fs)
            if a is None:
                continue
            for b in BANDS:
                rel[b].append(r[b])
            # per-epoch ratios (guard divide-by-zero -> skip that epoch's ratio)
            if a["alpha"] > 0:
                theta_alpha.append(a["theta"] / a["alpha"])
            if a["sigma"] > 0:
                delta_sigma.append(a["delta"] / a["sigma"])
            fast = a["alpha"] + a["sigma"] + a["beta"]
            if fast > 0:
                rem_slowing.append((a["delta"] + a["theta"]) / fast)
        if not any(rel[b] for b in BANDS):
            continue
        s = {"n_epochs": int(idx.size)}
        for b in BANDS:
            s[f"rel_{b}"] = _disp(rel[b], nd=5, full=False)
        s["theta_alpha"] = _disp(theta_alpha, nd=4, full=True)
        s["delta_sigma"] = _disp(delta_sigma, nd=4, full=True)
        s["rem_slowing"] = _disp(rem_slowing, nd=4, full=True)
        stages[POOL_TAG[code]] = s
    return {"ok": True, "stages": stages}


# --------------------------------------------------------------- EEG complexity
def eeg_complexity_dispersion(eeg, fs, stage_codes,
                              max_epochs=nkf.EEG_MAX_EPOCHS,
                              ep_samples=nkf.EEG_EPOCH_SAMPLES):
    """Per-stage dispersion of per-epoch sample & permutation entropy.

    Mirrors `nk_features.eeg_complexity_by_stage` epoch-selection and decimation
    exactly, but keeps every epoch's entropy value and reports the spread rather
    than the mean (the averaged CSV already keeps only mean + a single SD).
    """
    eeg = np.asarray(eeg, float)
    fs = float(fs)
    codes = np.rint(np.asarray(stage_codes, float)).astype(int)
    if not nkf._NK_OK or fs <= 0 or eeg.size == 0 or codes.size == 0:
        return {"ok": False, "error": "no usable EEG / staging / neurokit"}
    spe, n_ep, codes = nkf._align_epochs(eeg.size, fs, codes)
    if spe is None or n_ep < 1:
        return {"ok": False, "error": "signal shorter than one epoch"}

    stages = {}
    for code in STAGE_ORDER:
        idx = np.where(codes == code)[0]
        if idx.size == 0:
            continue
        if idx.size > max_epochs:
            idx = idx[np.linspace(0, idx.size - 1, max_epochs).astype(int)]
        sampen, permen = [], []
        for e in idx:
            seg = eeg[e * spe:(e + 1) * spe]
            seg = seg[np.isfinite(seg)]
            if seg.size < spe // 2:
                continue
            seg = nkf._decimate_to(seg, ep_samples)
            seg = seg - seg.mean()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    se, _ = nkf.nk.entropy_sample(seg)
                    pe, _ = nkf.nk.entropy_permutation(seg)
                if np.isfinite(se):
                    sampen.append(float(se))
                if np.isfinite(pe):
                    permen.append(float(pe))
            except Exception:
                continue
        if sampen or permen:
            stages[POOL_TAG[code]] = {
                "n_epochs": int(idx.size),
                "sampen": _disp(sampen, nd=4, full=True),
                "permen": _disp(permen, nd=4, full=True),
            }
    return {"ok": True, "stages": stages}


# --------------------------------------------------------------- respiratory rate
def rsp_rate_dispersion(rsp, fs, stage_codes):
    """Per-stage dispersion of the instantaneous respiratory-rate series.

    Same pipeline as `nk_features.rsp_rate_by_stage` (concatenate a stage's epochs,
    rsp_clean, rsp_rate, keep 4-40 bpm), but reports the spread of the rate series
    (percentiles + IQR in addition to the SD/CV the averaged CSV already carries).
    """
    rsp = np.asarray(rsp, float)
    fs = float(fs)
    codes = np.rint(np.asarray(stage_codes, float)).astype(int)
    if not nkf._NK_OK or fs <= 0 or rsp.size == 0 or codes.size == 0:
        return {"ok": False, "error": "no usable respiratory / staging"}
    spe, n_ep, codes = nkf._align_epochs(rsp.size, fs, codes)
    if spe is None or n_ep < 1:
        return {"ok": False, "error": "signal shorter than one epoch"}

    stages = {}
    for code in STAGE_ORDER:
        idx = np.where(codes == code)[0]
        if idx.size < 4:
            continue
        seg = np.concatenate([rsp[e * spe:(e + 1) * spe] for e in idx])
        seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clean = nkf.nk.rsp_clean(seg, sampling_rate=fs)
                rate = nkf.nk.rsp_rate(clean, sampling_rate=fs)
            rate = np.asarray(rate, float)
            rate = rate[np.isfinite(rate) & (rate > 4) & (rate < 40)]
        except Exception:
            rate = np.array([])
        if rate.size:
            d = _disp(rate, nd=3, full=True)
            d["n_epochs"] = int(idx.size)
            stages[POOL_TAG[code]] = d
    return {"ok": True, "stages": stages}


# --------------------------------------------------------------- spindles
def spindle_dispersion(eeg, fs, stage_codes, k=es.SPINDLE_K):
    """Per-stage dispersion of spindle amplitude & duration.

    Reuses `eeg_spectral`'s exact detector (same envelope, same per-recording
    threshold, same duration/merge bounds) but keeps every detected spindle's
    amplitude and duration and reports their spread (the averaged CSV keeps only
    amp_mean / dur_mean). Count / density are already distributional and stay in
    the mean CSV.
    """
    eeg = np.asarray(eeg, float)
    fs = float(fs)
    codes = np.rint(np.asarray(stage_codes, float)).astype(int)
    if not es._SCIPY_OK or fs <= 0 or eeg.size == 0 or codes.size == 0:
        return {"ok": False, "error": "no usable EEG / staging / scipy"}
    if fs < 2 * es.SPINDLE_BAND[1]:
        return {"ok": False, "error": "sampling rate too low for the spindle band"}
    spe, n_ep, codes = es._align_epochs(eeg.size, fs, codes)
    if spe is None or n_ep < 1:
        return {"ok": False, "error": "signal shorter than one epoch"}

    x = eeg[:n_ep * spe].astype(float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        env = es._bandpass_env(x, fs, es.SPINDLE_BAND)
    except Exception as e:
        return {"ok": False, "error": f"spindle filter failed: {type(e).__name__}"}
    if env is None:
        return {"ok": False, "error": "spindle band collapses at this sampling rate"}

    sample_stage = np.repeat(codes[:n_ep], spe)[:env.size]
    nrem_mask = np.isin(sample_stage, (2, 1))
    if nrem_mask.sum() < fs * 60:
        return {"ok": False, "error": "insufficient NREM to calibrate spindle threshold"}
    base = env[nrem_mask]
    thr = float(base.mean() + k * base.std())
    min_len = int(es.SPINDLE_MIN_S * fs)
    max_len = int(es.SPINDLE_MAX_S * fs)
    merge_gap = int(es.SPINDLE_MERGE_GAP_S * fs)

    stages = {}
    for code in (2, 1):
        idx = np.where(codes[:n_ep] == code)[0]
        minutes = idx.size * EPOCH_SEC / 60.0
        if minutes < es.MIN_STAGE_MIN_FOR_SPINDLE:
            continue
        stage_mask = np.isin(sample_stage, (code,))
        above = (env > thr) & stage_mask
        bursts = es._detect_bursts(above, min_len, max_len, merge_gap=merge_gap)
        amps = [float(env[a:b].max()) for a, b in bursts] if bursts else []
        durs = [(b - a) / fs for a, b in bursts] if bursts else []
        stages[POOL_TAG[code]] = {
            "n_spindles": len(bursts),
            "amp": _disp(amps, nd=2, full=False),
            "dur": _disp(durs, nd=3, full=False),
        }
    return {"ok": True, "threshold_uv": _r(thr, 2), "stages": stages}
