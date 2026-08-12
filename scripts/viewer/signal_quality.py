"""NeuroKit2 signal-quality assessment + diagnostic plots for the PSG channels.

Complements the *feature* extractors: before trusting any HRV / respiratory number
we should know how clean the underlying waveform is. NeuroKit ships quality
estimators and diagnostic plotters; this module wraps the ones that apply to the
channels we decode and makes them safe + cheap to run across a cohort.

QUALITY (numeric, per recording):
  * ECG — `nk.ecg_quality`:
      - `averageQRS`: a per-sample 0-1 score (correlation of each beat to the
        average QRS template). We summarise it as mean / median / %-good (>=0.8) /
        %-bad (<0.5).
      - `zhao2018`: a categorical verdict ("Excellent" / "Barely acceptable" /
        "Unacceptable") for the sampled signal.
  * RSP — `nk.rsp_quality`: a per-sample 0-1 score; summarised as mean / %-good.
  * EEG / EOG — NeuroKit's `signal_quality` requires a signal_type it supports
      (ppg/ecg/rsp), so there is no native EEG/EOG quality score; those channels
      get a simple flat-line / clipping / NaN-fraction sanity check instead.

Whole-night `ecg_quality` is ~50 s/recording (8 h @ 200 Hz), too slow for a cohort,
and a single window can land on an artifact. So quality is computed over several
evenly-spaced windows (default 8 x 90 s) and pooled — robust and ~1-2 s/recording.
Per-stage quality is computed from the longest contiguous bout of each stage, so a
stage's score is not contaminated by epoch-boundary seams.

PLOTS (PNG, per recording): `nk.ecg_plot` (R-peaks, cleaned trace, quality overlay,
instantaneous HR, average-beat morphology with P/Q/S/T delineation) on a short
window, and `nk.rsp_plot` (raw/clean, breathing rate, amplitude, RVT, cycle
symmetry). Saved to disk; nothing is displayed.

Pure numpy + neurokit2 + matplotlib (Agg backend). Every field degrades to None and
plotting degrades to a skipped file rather than raising, so it is safe unattended.
"""
import os
import warnings

import numpy as np

# NumPy renamed `trapz` -> `trapezoid` in 2.0; NeuroKit's zhao2018 ECG-quality path
# calls `np.trapezoid`, which is absent on the box's NumPy <2.0, so every zhao
# verdict silently raised (caught in zhao_verdict) and serialised blank. Alias it
# back so the categorical verdict works regardless of the installed NumPy.
if not hasattr(np, "trapezoid") and hasattr(np, "trapz"):
    np.trapezoid = np.trapz          # type: ignore[attr-defined]

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import neurokit2 as nk
    _NK_OK = True
except Exception:                                    # pragma: no cover
    nk = None
    _NK_OK = False

EPOCH_SEC = 30.0
STAGE_ORDER = [5, 3, 2, 1, 4]
POOL_TAG = {5: "wake", 3: "n1", 2: "n2", 1: "n3", 4: "rem"}

# quality-sampling: N evenly-spaced windows of WIN_SEC each, pooled.
N_WINDOWS = 8
WIN_SEC = 90.0
GOOD_Q, BAD_Q = 0.8, 0.5                              # quality thresholds


def _finite(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _r(x, nd=3):
    v = _finite(x)
    return None if v is None else round(v, nd)


def _windows(n_samples, fs, n_win=N_WINDOWS, win_sec=WIN_SEC):
    """Evenly-spaced (start, end) sample windows across a signal; fewer if short."""
    w = int(win_sec * fs)
    if w < 1 or n_samples < w:
        return [(0, n_samples)] if n_samples > 0 else []
    if n_samples <= n_win * w:                       # short signal: contiguous chunks
        starts = list(range(0, n_samples - w + 1, w))
    else:
        starts = np.linspace(0, n_samples - w, n_win).astype(int).tolist()
    return [(s, s + w) for s in starts]


def _pooled_quality(sig, fs, kind):
    """Pooled NeuroKit per-sample quality over evenly-spaced windows.

    kind='ecg' -> nk.ecg_quality(averageQRS); kind='rsp' -> nk.rsp_quality.
    Returns (pooled_quality_array, n_beats_total) with quality in [0,1], or
    (None, 0). Windows that error are skipped.
    """
    if not _NK_OK or sig is None or fs <= 0 or sig.size == 0:
        return None, 0
    qs, n_beats = [], 0
    for a, b in _windows(sig.size, fs):
        seg = np.asarray(sig[a:b], float)
        seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
        if seg.size < int(fs * 5):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if kind == "ecg":
                    clean = nk.ecg_clean(seg, sampling_rate=fs)
                    _, info = nk.ecg_peaks(clean, sampling_rate=fs)
                    rp = np.asarray(info.get("ECG_R_Peaks", []), int)
                    n_beats += int(rp.size)
                    if rp.size < 3:
                        continue
                    q = nk.ecg_quality(clean, rpeaks=rp, sampling_rate=fs,
                                       method="averageQRS")
                else:
                    clean = nk.rsp_clean(seg, sampling_rate=fs)
                    q = nk.rsp_quality(clean, sampling_rate=fs)
            q = np.asarray(q, float)
            q = q[np.isfinite(q)]
            if q.size:
                qs.append(np.clip(q, 0.0, 1.0))
        except Exception:
            continue
    if not qs:
        return None, n_beats
    return np.concatenate(qs), n_beats


def _summ(q, prefix):
    """mean/median/%good/%bad summary of a pooled quality array."""
    out = {f"{prefix}_q_mean": None, f"{prefix}_q_median": None,
           f"{prefix}_q_pct_good": None, f"{prefix}_q_pct_bad": None,
           f"{prefix}_q_n_samples": 0}
    if q is None or q.size == 0:
        return out
    out[f"{prefix}_q_mean"] = _r(np.mean(q), 4)
    out[f"{prefix}_q_median"] = _r(np.median(q), 4)
    out[f"{prefix}_q_pct_good"] = _r(100.0 * np.mean(q >= GOOD_Q), 1)
    out[f"{prefix}_q_pct_bad"] = _r(100.0 * np.mean(q < BAD_Q), 1)
    out[f"{prefix}_q_n_samples"] = int(q.size)
    return out


def _longest_bout(codes, code):
    """(start_epoch, end_epoch_exclusive) of the longest contiguous run of `code`."""
    best = (0, 0)
    i, n = 0, len(codes)
    while i < n:
        if codes[i] == code:
            j = i + 1
            while j < n and codes[j] == code:
                j += 1
            if (j - i) > (best[1] - best[0]):
                best = (i, j)
            i = j
        else:
            i += 1
    return best


def ecg_quality_by_stage(ecg, fs, stage_codes):
    """Mean ECG quality within the longest contiguous bout of each stage.

    A single clean bout per stage avoids epoch-seam artifacts. Returns
    {tag: q_mean}. Bouts shorter than ~1 min are skipped.
    """
    if not _NK_OK or ecg is None or fs <= 0 or ecg.size == 0:
        return {}
    codes = np.rint(np.asarray(stage_codes, float)).astype(int)
    spe = int(round(fs * EPOCH_SEC))
    if spe < 1:
        return {}
    n_ep = min(len(codes), ecg.size // spe)
    codes = codes[:n_ep]
    out = {}
    for code in STAGE_ORDER:
        s_ep, e_ep = _longest_bout(codes, code)
        if (e_ep - s_ep) < 2:                        # < 1 min of this stage
            continue
        seg = ecg[s_ep * spe:e_ep * spe]
        # cap to WIN_SEC*4 so a huge bout stays cheap
        cap = int(WIN_SEC * 4 * fs)
        if seg.size > cap:
            mid = seg.size // 2
            seg = seg[mid - cap // 2: mid + cap // 2]
        q, _ = _pooled_quality(seg, fs, "ecg")
        if q is not None and q.size:
            out[POOL_TAG[code]] = _r(np.mean(q), 4)
    return out


def zhao_verdict(ecg, fs):
    """NeuroKit zhao2018 categorical ECG-quality verdict on a mid-recording
    window (one representative window; the method returns a single label)."""
    if not _NK_OK or ecg is None or fs <= 0 or ecg.size == 0:
        return None
    w = int(WIN_SEC * fs)
    mid = ecg.size // 2
    seg = ecg[max(0, mid - w // 2): mid + w // 2]
    seg = np.nan_to_num(np.asarray(seg, float), nan=0.0)
    if seg.size < int(fs * 10):
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clean = nk.ecg_clean(seg, sampling_rate=fs)
            _, info = nk.ecg_peaks(clean, sampling_rate=fs)
            rp = np.asarray(info.get("ECG_R_Peaks", []), int)
            if rp.size < 3:
                return None
            v = nk.ecg_quality(clean, rpeaks=rp, sampling_rate=fs, method="zhao2018")
        return str(v)
    except Exception:
        return None


def basic_channel_quality(sig, fs, prefix, flat_win_sec=1.0, flat_thresh=0.02):
    """Native-quality-free sanity check for EEG/EOG/EMG: NaN fraction, flat-line
    fraction, and clipping fraction (at the signal's min/max rails). All in [0,1];
    higher NaN/flat/clip = worse.

    flat_frac is **windowed** and resolution-independent: the signal is split into
    `flat_win_sec` windows and a window counts as flat when its standard deviation
    is below `flat_thresh` of the channel's own active level (the 90th percentile of
    window SDs — the p90, not the median, so a channel that is dead for *most* of
    the night, whose median SD is 0, still has a non-zero active reference). An
    earlier per-sample derivative test (`|diff| < 1e-6*MAD`) collapsed to
    "consecutive samples byte-identical", which on coarse-ADC sites (e.g. Emory's
    ~2 uV EEG quantizer) flagged a fully live signal as ~70% flat. The windowed SD
    ignores quantization (a live window still has real variance) yet still catches
    genuine sustained dropout / dead channels (a frozen window has ~zero SD relative
    to the active level). Calibrated on the standard cohort: at 2% of p90 all live
    channels (Emory-quantized included) score <=0.005 while a 4.5 h frozen Kaiser
    channel scores 0.68."""
    out = {f"{prefix}_nan_frac": None, f"{prefix}_flat_frac": None,
           f"{prefix}_clip_frac": None}
    if sig is None or fs <= 0 or sig.size == 0:
        return out
    x = np.asarray(sig, float)
    out[f"{prefix}_nan_frac"] = _r(np.mean(~np.isfinite(x)), 4)
    finite = x[np.isfinite(x)]
    if finite.size < 2:
        return out
    lo, hi = np.min(finite), np.max(finite)
    rail = (np.isclose(finite, lo) | np.isclose(finite, hi))
    out[f"{prefix}_clip_frac"] = _r(np.mean(rail), 4)
    # windowed flat-line detection, referenced to the channel's own active level
    w = max(int(round(flat_win_sec * fs)), 2)
    n_win = finite.size // w
    if n_win < 1:
        return out
    w_std = finite[:n_win * w].reshape(n_win, w).std(axis=1)
    active = float(np.percentile(w_std, 90))
    flat = (w_std < flat_thresh * active) if active > 0 else (w_std <= 0)
    out[f"{prefix}_flat_frac"] = _r(np.mean(flat), 4)
    return out


# ------------------------------------------------------------------- plots
def save_ecg_plot(ecg, fs, path, win_sec=20.0):
    """Render nk.ecg_plot for a short mid-recording window to `path` (PNG)."""
    if not _NK_OK or ecg is None or fs <= 0:
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = int(win_sec * fs)
    mid = ecg.size // 2
    seg = np.nan_to_num(np.asarray(ecg[max(0, mid - w // 2): mid + w // 2], float), nan=0.0)
    if seg.size < int(fs * 5):
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sigs, info = nk.ecg_process(seg, sampling_rate=fs)
            nk.ecg_plot(sigs, info)
        fig = plt.gcf()
        fig.set_size_inches(12, 7)
        fig.savefig(path, dpi=90, bbox_inches="tight")
        plt.close("all")
        return True
    except Exception:
        plt.close("all")
        return False


def save_rsp_plot(rsp, fs, path, win_sec=90.0):
    """Render nk.rsp_plot for a mid-recording window to `path` (PNG)."""
    if not _NK_OK or rsp is None or fs <= 0:
        return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    w = int(win_sec * fs)
    mid = rsp.size // 2
    seg = np.nan_to_num(np.asarray(rsp[max(0, mid - w // 2): mid + w // 2], float), nan=0.0)
    if seg.size < int(fs * 20):
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sigs, info = nk.rsp_process(seg, sampling_rate=fs)
            nk.rsp_plot(sigs, info)
        fig = plt.gcf()
        fig.set_size_inches(11, 9)
        fig.savefig(path, dpi=90, bbox_inches="tight")
        plt.close("all")
        return True
    except Exception:
        plt.close("all")
        return False


def quality_row(channels, fss, roles, stage_codes,
                ecg_ch=None, rsp_ch=None, eeg_ch=None, eog_ch=None):
    """Assemble the full per-recording quality dict from the decoded channels."""
    row = {}
    ecg = channels.get(ecg_ch) if ecg_ch else None
    rsp = channels.get(rsp_ch) if rsp_ch else None
    eeg = channels.get(eeg_ch) if eeg_ch else None
    eog = channels.get(eog_ch) if eog_ch else None

    q_ecg, n_beats = _pooled_quality(ecg, fss.get(ecg_ch, 0), "ecg") if ecg is not None else (None, 0)
    row.update(_summ(q_ecg, "ecg"))
    row["ecg_n_beats"] = int(n_beats)
    row["ecg_zhao_verdict"] = zhao_verdict(ecg, fss.get(ecg_ch, 0)) if ecg is not None else None
    for tag, v in (ecg_quality_by_stage(ecg, fss.get(ecg_ch, 0), stage_codes)
                   if ecg is not None else {}).items():
        row[f"ecg_q_{tag}"] = v

    q_rsp, _ = _pooled_quality(rsp, fss.get(rsp_ch, 0), "rsp") if rsp is not None else (None, 0)
    row.update(_summ(q_rsp, "rsp"))

    row.update(basic_channel_quality(eeg, fss.get(eeg_ch, 0), "eeg") if eeg is not None
               else basic_channel_quality(None, 0, "eeg"))
    row.update(basic_channel_quality(eog, fss.get(eog_ch, 0), "eog") if eog is not None
               else basic_channel_quality(None, 0, "eog"))
    return row
