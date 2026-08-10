"""Per-sleep-stage physiological features via NeuroKit2.

Given a recording's PSG channels + the (preprocessed) per-epoch CAISR stage
codes, this extracts features *within each sleep stage* — the idea being that
cognitive impairment (CI) shows up less in a night's average physiology than in
how physiology is *modulated across stages*:

  * **ECG heart-rate variability (HRV)** — parasympathetic tone dominates deep
    NREM (N3) and sympathetic activity surges in REM, so a healthy night shows a
    large swing in HR / RMSSD / LF-HF across stages. That stage-dependent
    modulation is blunted in autonomic dysregulation and neurodegeneration, so we
    report HRV *per stage* and, crucially, **cross-stage contrasts** (REM-vs-NREM
    ratios, wake-vs-sleep HR delta, the range of HR across stages).
  * **EEG complexity** — sample entropy + permutation entropy per stage. EEG
    slowing and reduced signal complexity (especially in slow-wave sleep) is a
    hallmark of cognitive decline.
  * **Respiratory rate & variability per stage** — breathing regularity changes
    with stage; irregularity/variability differences are an exploratory marker.

Why these NeuroKit calls and not others (measured on the pdmle box, no numba):
`hrv_time` / `hrv_frequency` are ~0.03 s / 0.5 s even on a whole night's beats,
but `hrv_nonlinear` / the umbrella `nk.hrv()` are O(n^2) and time out past a few
thousand beats, and `fractal_higuchi` takes ~4 min. So HRV uses only the linear
domains (Poincare SD1/SD2 are added in closed form), and EEG complexity is run on
length-capped, decimated epochs so its cost is bounded regardless of sampling rate.

Staging is aligned to a signal by epoch index (epoch i = samples
[i*spe : (i+1)*spe], spe = round(fs*30)); the shorter length wins. Unknown
(code 9) epochs are ignored. Every value is JSON-serialisable and every failure
mode degrades to NaN/None rather than raising, so it is safe to run across a whole
cohort unattended.
"""
import warnings

import numpy as np

try:                                   # NeuroKit pulls in scipy/pandas/sklearn
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import neurokit2 as nk
    _NK_OK = True
except Exception:                      # pragma: no cover - env without neurokit
    nk = None
    _NK_OK = False

EPOCH_SEC = 30.0
# Stage code -> short name (CAISR coding, shared with the rest of the viewer).
STAGE_LABEL = {5: "Wake", 3: "N1", 2: "N2", 1: "N3", 4: "REM"}
STAGE_ORDER = [5, 3, 2, 1, 4]                       # Wake, N1, N2, N3, REM
# Pools we summarise HRV over: the five stages, plus pooled NREM (N1+N2+N3) and
# pooled Sleep (all NREM+REM). Keys are the lower-case tags used in feature names.
NREM_CODES = (3, 2, 1)
SLEEP_CODES = (3, 2, 1, 4)
POOL_TAG = {5: "wake", 3: "n1", 2: "n2", 1: "n3", 4: "rem"}

# Physiologic RR-interval bounds (ms): 300 ms = 200 bpm, 2000 ms = 30 bpm.
RR_MIN_MS, RR_MAX_MS = 300.0, 2000.0
MIN_BEATS_TIME = 30                                 # below this, HRV time-domain -> NaN
MIN_BEATS_FREQ = 50                                 # frequency domain needs more beats
# EEG complexity cost control: decimate each analysed epoch to <= this many
# samples (keeps entropy_sample, which is O(n^2), fast at any sampling rate), and
# analyse at most this many epochs per stage (sampled evenly across the stage).
EEG_EPOCH_SAMPLES = 1024
EEG_MAX_EPOCHS = 25

# HRV features we keep from nk.hrv_time / nk.hrv_frequency (+ derived).
_HRV_TIME_KEYS = ["HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50",
                  "HRV_SDSD", "HRV_CVNN", "HRV_MedianNN"]
_HRV_FREQ_KEYS = ["HRV_LF", "HRV_HF", "HRV_LFHF", "HRV_LFn", "HRV_HFn", "HRV_TP"]


# --------------------------------------------------------------------------- utils
def _finite(x):
    """float(x) if finite else None (JSON-safe)."""
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _r(x, nd=3):
    v = _finite(x)
    return None if v is None else round(v, nd)


def _pick_channel(labels, roles, wanted_roles, prefer=None):
    """First label whose inferred role is in `wanted_roles`.

    `prefer` is an optional list of lower-case substrings tried in order — the
    first label matching an earlier substring wins (e.g. prefer central EEG for
    slow-wave complexity), falling back to the first label of the wanted role.
    """
    cands = [lab for lab in labels if roles.get(lab) in wanted_roles]
    if not cands:
        return None
    if prefer:
        for sub in prefer:
            for lab in cands:
                if sub in lab.lower():
                    return lab
    return cands[0]


def _align_epochs(n_samples, fs, codes):
    """Return (spe, n_ep, codes[:n_ep]) aligning a signal to per-epoch stage codes."""
    spe = int(round(fs * EPOCH_SEC))
    if spe < 1:
        return None, 0, codes
    n_ep = min(len(codes), n_samples // spe)
    return spe, n_ep, codes[:n_ep]


# --------------------------------------------------------------------------- HRV
def _poincare(rr_ms):
    """Closed-form Poincare SD1/SD2 from an RR series (ms). Avoids the O(n^2)
    nk.hrv_nonlinear. SD1^2 = 0.5*SDSD^2 ; SD2^2 = 2*SDNN^2 - 0.5*SDSD^2."""
    if rr_ms.size < 3:
        return None, None, None
    sdsd = float(np.std(np.diff(rr_ms), ddof=1)) if rr_ms.size > 2 else np.nan
    sdnn = float(np.std(rr_ms, ddof=1))
    sd1_sq = 0.5 * sdsd * sdsd
    sd2_sq = 2.0 * sdnn * sdnn - 0.5 * sdsd * sdsd
    sd1 = float(np.sqrt(sd1_sq)) if sd1_sq > 0 else np.nan
    sd2 = float(np.sqrt(sd2_sq)) if sd2_sq > 0 else np.nan
    ratio = (sd1 / sd2) if (np.isfinite(sd1) and np.isfinite(sd2) and sd2 > 0) else np.nan
    return sd1, sd2, ratio


def _hrv_from_rr(rr_ms, fs):
    """HRV feature dict from a *clean* per-stage RR-interval series (ms).

    Reconstructs a pseudo peak train (cumsum of RR) so NeuroKit's own definitions
    are used, then keeps the linear time/frequency features + closed-form Poincare
    + heart rate. Returns a dict of {feature: value|None}; sparse stages give None.
    """
    out = {"n_beats": int(rr_ms.size)}
    keys = ([k.replace("HRV_", "").lower() for k in _HRV_TIME_KEYS]
            + [k.replace("HRV_", "").lower() for k in _HRV_FREQ_KEYS]
            + ["hr_mean", "sd1", "sd2", "sd1sd2"])
    for k in keys:
        out[k] = None
    if not _NK_OK or rr_ms.size < MIN_BEATS_TIME:
        return out

    # pseudo peaks (sample indices) from the RR series; nk recomputes RR = diff.
    peaks = np.cumsum(rr_ms / 1000.0 * fs).astype(np.int64)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ht = nk.hrv_time(peaks, sampling_rate=fs)
        for k in _HRV_TIME_KEYS:
            if k in ht.columns:
                out[k.replace("HRV_", "").lower()] = _r(ht[k].iloc[0])
        mean_nn = out.get("meannn")
        out["hr_mean"] = _r(60000.0 / mean_nn) if mean_nn else None
    except Exception:
        pass
    if rr_ms.size >= MIN_BEATS_FREQ:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hf = nk.hrv_frequency(peaks, sampling_rate=fs)
            for k in _HRV_FREQ_KEYS:
                if k in hf.columns:
                    out[k.replace("HRV_", "").lower()] = _r(hf[k].iloc[0], 5)
        except Exception:
            pass
    sd1, sd2, ratio = _poincare(rr_ms)
    out["sd1"], out["sd2"], out["sd1sd2"] = _r(sd1), _r(sd2), _r(ratio, 4)
    return out


def ecg_hrv_by_stage(ecg, fs, stage_codes):
    """Per-stage HRV + cross-stage contrasts from one ECG channel.

    Detects R-peaks once on the whole night, computes consecutive RR intervals,
    labels each RR by the sleep stage of its starting beat, drops RR that are
    non-physiologic OR that straddle a stage change (so no bout boundary
    contaminates a stage's HRV), then summarises each stage/pool with
    `_hrv_from_rr`. Returns {"ok", "stages": {tag: {...}}, "contrasts": {...}}.
    """
    ecg = np.asarray(ecg, float)
    fs = float(fs)
    codes = np.rint(np.asarray(stage_codes, float)).astype(int)
    if not _NK_OK:
        return {"ok": False, "error": "neurokit2 unavailable"}
    if fs <= 0 or ecg.size < int(10 * fs) or codes.size == 0:
        return {"ok": False, "error": "no usable ECG / staging"}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clean = nk.ecg_clean(ecg, sampling_rate=fs)
            _, info = nk.ecg_peaks(clean, sampling_rate=fs)
        rpeaks = np.asarray(info.get("ECG_R_Peaks", []), dtype=np.int64)
    except Exception as e:
        return {"ok": False, "error": f"peak detection failed: {type(e).__name__}"}
    if rpeaks.size < MIN_BEATS_TIME + 1:
        return {"ok": False, "error": f"only {rpeaks.size} beats detected"}

    # RR interval i spans beats [i, i+1]; label it by the stage of beat i.
    rr_ms = np.diff(rpeaks) / fs * 1000.0
    ep = np.clip((rpeaks[:-1] / fs // EPOCH_SEC).astype(int), 0, codes.size - 1)
    ep_next = np.clip((rpeaks[1:] / fs // EPOCH_SEC).astype(int), 0, codes.size - 1)
    beat_stage = codes[ep]
    next_stage = codes[ep_next]
    # keep physiologic RR whose both endpoints are the *same* real stage (no
    # cross-bout / stage-transition gaps polluting the interval series)
    good = ((rr_ms >= RR_MIN_MS) & (rr_ms <= RR_MAX_MS)
            & (beat_stage == next_stage) & (beat_stage != 9))
    rr_ms = rr_ms[good]
    beat_stage = beat_stage[good]

    stages = {}
    for code in STAGE_ORDER:
        sel = rr_ms[beat_stage == code]
        if sel.size:
            stages[POOL_TAG[code]] = _hrv_from_rr(sel, fs)
    nrem = rr_ms[np.isin(beat_stage, NREM_CODES)]
    if nrem.size:
        stages["nrem"] = _hrv_from_rr(nrem, fs)
    sleep = rr_ms[np.isin(beat_stage, SLEEP_CODES)]
    if sleep.size:
        stages["sleep"] = _hrv_from_rr(sleep, fs)

    return {
        "ok": True,
        "n_beats_total": int(rpeaks.size),
        "n_beats_clean": int(rr_ms.size),
        "stages": stages,
        "contrasts": _hrv_contrasts(stages),
    }


def _hrv_contrasts(stages):
    """Cross-stage HRV modulation features — the CI-relevant signal.

    A healthy autonomic system swings hard between stages (vagal N3, sympathetic
    REM); a flattened swing is the marker. We express that as ratios/deltas
    between pools and the spread of HR across the individual stages.
    """
    def g(tag, feat):
        return (stages.get(tag) or {}).get(feat)

    def ratio(a, b):
        a, b = _finite(a), _finite(b)
        return _r(a / b, 4) if (a is not None and b not in (None, 0)) else None

    def delta(a, b):
        a, b = _finite(a), _finite(b)
        return _r(a - b) if (a is not None and b is not None) else None

    out = {
        "rem_nrem_rmssd_ratio": ratio(g("rem", "rmssd"), g("nrem", "rmssd")),
        "rem_nrem_lfhf_ratio": ratio(g("rem", "lfhf"), g("nrem", "lfhf")),
        "rem_nrem_hr_delta": delta(g("rem", "hr_mean"), g("nrem", "hr_mean")),
        "wake_sleep_hr_delta": delta(g("wake", "hr_mean"), g("sleep", "hr_mean")),
        "n3_wake_rmssd_ratio": ratio(g("n3", "rmssd"), g("wake", "rmssd")),
    }
    # spread of HR / RMSSD across the individual stages (blunted -> small)
    hrs = [_finite(g(POOL_TAG[c], "hr_mean")) for c in STAGE_ORDER]
    hrs = [h for h in hrs if h is not None]
    if len(hrs) >= 2:
        out["stage_hr_range"] = _r(max(hrs) - min(hrs))
    rmssds = [_finite(g(POOL_TAG[c], "rmssd")) for c in STAGE_ORDER]
    rmssds = [v for v in rmssds if v is not None]
    if len(rmssds) >= 2 and np.mean(rmssds) > 0:
        out["stage_rmssd_cv"] = _r(float(np.std(rmssds) / np.mean(rmssds)), 4)
    return out


# --------------------------------------------------------------------------- EEG
def _decimate_to(x, target):
    """Stride-decimate a 1-D array to <= `target` samples (cheap, alias-tolerant
    enough for ordinal-pattern / sample-entropy complexity on sub-30 Hz EEG)."""
    n = x.size
    if n <= target:
        return x
    step = int(np.ceil(n / target))
    return x[::step]


def eeg_complexity_by_stage(eeg, fs, stage_codes,
                            max_epochs=EEG_MAX_EPOCHS, ep_samples=EEG_EPOCH_SAMPLES):
    """Per-stage EEG sample-entropy + permutation-entropy (means over epochs).

    Each analysed epoch is decimated to <= `ep_samples` samples so entropy_sample
    (O(n^2)) stays fast at any sampling rate; at most `max_epochs` epochs per stage
    are analysed, sampled evenly across the stage's epochs. Reduced complexity
    (lower entropy), especially in N3, is an exploratory CI marker.
    """
    eeg = np.asarray(eeg, float)
    fs = float(fs)
    codes = np.rint(np.asarray(stage_codes, float)).astype(int)
    if not _NK_OK or fs <= 0 or eeg.size == 0 or codes.size == 0:
        return {"ok": False, "error": "no usable EEG / staging"}
    spe, n_ep, codes = _align_epochs(eeg.size, fs, codes)
    if spe is None or n_ep < 1:
        return {"ok": False, "error": "signal shorter than one epoch"}

    stages = {}
    for code in STAGE_ORDER:
        idx = np.where(codes == code)[0]
        if idx.size == 0:
            continue
        if idx.size > max_epochs:                     # sample evenly across the stage
            idx = idx[np.linspace(0, idx.size - 1, max_epochs).astype(int)]
        sampen, permen = [], []
        for e in idx:
            seg = eeg[e * spe:(e + 1) * spe]
            seg = seg[np.isfinite(seg)]
            if seg.size < spe // 2:
                continue
            seg = _decimate_to(seg, ep_samples)
            seg = seg - seg.mean()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    se, _ = nk.entropy_sample(seg)
                    pe, _ = nk.entropy_permutation(seg)
                if np.isfinite(se):
                    sampen.append(float(se))
                if np.isfinite(pe):
                    permen.append(float(pe))
            except Exception:
                continue
        if sampen or permen:
            stages[POOL_TAG[code]] = {
                "n_epochs": int(idx.size),
                "sampen_mean": _r(np.mean(sampen), 4) if sampen else None,
                "sampen_sd": _r(np.std(sampen), 4) if len(sampen) > 1 else None,
                "permen_mean": _r(np.mean(permen), 4) if permen else None,
                "permen_sd": _r(np.std(permen), 4) if len(permen) > 1 else None,
            }

    contrasts = {}
    def g(tag, feat):
        return (stages.get(tag) or {}).get(feat)
    a, b = _finite(g("n3", "sampen_mean")), _finite(g("wake", "sampen_mean"))
    if a is not None and b not in (None, 0):
        contrasts["n3_wake_sampen_ratio"] = _r(a / b, 4)
    return {"ok": True, "stages": stages, "contrasts": contrasts}


# --------------------------------------------------------------------------- RSP
def rsp_rate_by_stage(rsp, fs, stage_codes):
    """Per-stage respiratory rate (breaths/min) mean + variability from one effort
    or airflow channel. Each stage's epochs are concatenated, cleaned, and passed
    to nk.rsp_rate; the instantaneous-rate series gives mean and SD (RRV)."""
    rsp = np.asarray(rsp, float)
    fs = float(fs)
    codes = np.rint(np.asarray(stage_codes, float)).astype(int)
    if not _NK_OK or fs <= 0 or rsp.size == 0 or codes.size == 0:
        return {"ok": False, "error": "no usable respiratory / staging"}
    spe, n_ep, codes = _align_epochs(rsp.size, fs, codes)
    if spe is None or n_ep < 1:
        return {"ok": False, "error": "signal shorter than one epoch"}

    stages = {}
    for code in STAGE_ORDER:
        idx = np.where(codes == code)[0]
        if idx.size < 4:                              # need >~2 min for a rate
            continue
        seg = np.concatenate([rsp[e * spe:(e + 1) * spe] for e in idx])
        seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clean = nk.rsp_clean(seg, sampling_rate=fs)
                rate = nk.rsp_rate(clean, sampling_rate=fs)
            rate = np.asarray(rate, float)
            rate = rate[np.isfinite(rate) & (rate > 4) & (rate < 40)]
        except Exception:
            rate = np.array([])
        if rate.size:
            stages[POOL_TAG[code]] = {
                "n_epochs": int(idx.size),
                "rate_mean": _r(np.mean(rate), 2),
                "rate_sd": _r(np.std(rate), 3),
                "rrv_cv": _r(float(np.std(rate) / np.mean(rate)), 4) if np.mean(rate) > 0 else None,
            }
    return {"ok": True, "stages": stages}


# --------------------------------------------------------------------------- top
def nk_stage_features(channels, fss, roles, stage_codes,
                      do_ecg=True, do_eeg=True, do_rsp=True):
    """Run every per-stage NeuroKit extractor available for this recording.

    channels: {label -> 1-D samples}, fss: {label -> Hz}, roles: {label -> role}
    (roles as produced by app.channel_role), stage_codes: per-epoch int codes.
    Returns a nested dict {"ecg":..., "eeg":..., "rsp":..., "channels_used":...}.
    Picks the first channel of each needed role; missing roles are skipped.
    """
    labels = list(channels.keys())
    used = {}
    out = {"channels_used": used}

    if do_ecg:
        ch = _pick_channel(labels, roles, {"ecg"})
        if ch is not None:
            used["ecg"] = ch
            out["ecg"] = ecg_hrv_by_stage(channels[ch], fss.get(ch, 0), stage_codes)
    if do_eeg:
        # prefer central (C3/C4) derivations — best for slow-wave complexity —
        # then occipital, else the first EEG channel.
        ch = _pick_channel(labels, roles, {"eeg"}, prefer=["c3", "c4", "o1", "o2"])
        if ch is not None:
            used["eeg"] = ch
            out["eeg"] = eeg_complexity_by_stage(channels[ch], fss.get(ch, 0), stage_codes)
    if do_rsp:
        ch = _pick_channel(labels, roles, {"effort", "airflow"})
        if ch is not None:
            used["rsp"] = ch
            out["rsp"] = rsp_rate_by_stage(channels[ch], fss.get(ch, 0), stage_codes)
    return out


# ------------------------------------------------------------------ flat schema
def _flat_hrv(prefix, stages, contrasts):
    row = {}
    for tag in ["wake", "n1", "n2", "n3", "rem", "nrem", "sleep"]:
        s = stages.get(tag) or {}
        for feat in ["hr_mean", "sdnn", "rmssd", "pnn50", "sdsd", "cvnn",
                     "sd1", "sd2", "sd1sd2", "lf", "hf", "lfhf", "lfn", "hfn",
                     "tp", "n_beats"]:
            row[f"{prefix}_{tag}_{feat}"] = s.get(feat)
    for k, v in (contrasts or {}).items():
        row[f"{prefix}_{k}"] = v
    return row


def _flat_eeg(prefix, stages, contrasts):
    row = {}
    for tag in ["wake", "n1", "n2", "n3", "rem"]:
        s = stages.get(tag) or {}
        for feat in ["sampen_mean", "sampen_sd", "permen_mean", "permen_sd", "n_epochs"]:
            row[f"{prefix}_{tag}_{feat}"] = s.get(feat)
    for k, v in (contrasts or {}).items():
        row[f"{prefix}_{k}"] = v
    return row


def _flat_rsp(prefix, stages):
    row = {}
    for tag in ["wake", "n1", "n2", "n3", "rem"]:
        s = stages.get(tag) or {}
        for feat in ["rate_mean", "rate_sd", "rrv_cv"]:
            row[f"{prefix}_{tag}_{feat}"] = s.get(feat)
    return row


def flatten_features(nk_out):
    """Flatten the nested nk_stage_features output into one wide {col: value} row
    for CSV export. Column order is stable (see NK_FEATURE_COLUMNS)."""
    row = {}
    ecg = nk_out.get("ecg") or {}
    if ecg.get("ok"):
        row.update(_flat_hrv("hrv", ecg.get("stages", {}), ecg.get("contrasts", {})))
    eeg = nk_out.get("eeg") or {}
    if eeg.get("ok"):
        row.update(_flat_eeg("eegc", eeg.get("stages", {}), eeg.get("contrasts", {})))
    rsp = nk_out.get("rsp") or {}
    if rsp.get("ok"):
        row.update(_flat_rsp("rsp", rsp.get("stages", {})))
    return row


def _build_columns():
    """The full stable column list, so the CSV header is fixed even when a
    recording is missing a modality (missing values -> empty)."""
    dummy = {
        "ecg": {"ok": True, "stages": {}, "contrasts": {
            "rem_nrem_rmssd_ratio": None, "rem_nrem_lfhf_ratio": None,
            "rem_nrem_hr_delta": None, "wake_sleep_hr_delta": None,
            "n3_wake_rmssd_ratio": None, "stage_hr_range": None,
            "stage_rmssd_cv": None}},
        "eeg": {"ok": True, "stages": {}, "contrasts": {"n3_wake_sampen_ratio": None}},
        "rsp": {"ok": True, "stages": {}},
    }
    return list(flatten_features(dummy).keys())


NK_FEATURE_COLUMNS = _build_columns()
