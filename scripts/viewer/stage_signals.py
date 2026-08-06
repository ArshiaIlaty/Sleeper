"""Chunk a PSG channel by sleep stage and summarise each stage.

Given one channel's samples, its sampling rate, and the (preprocessed) per-epoch
stage codes, this groups every 30-second epoch by its stage and produces, per
stage:

  * amplitude statistics — mean, SD, min, max, p5/p95 — over all samples in that
    stage, plus how many epochs / minutes / percent of the night it covers;
  * a **representative example epoch** — the raw 30 s waveform from the middle of
    that stage's longest contiguous bout (the "purest" stretch), decimated for
    transport, so the morphology in each stage is directly visible;
  * for EEG channels only, **relative spectral band power** (delta / theta / alpha
    / sigma / beta), averaged per-epoch — the physiologically meaningful
    "average" for EEG, since a time-domain average of unlocked oscillations just
    cancels out. Delta dominates N3, sigma (spindles) rises in N2, etc.

Staging is aligned to the signal by epoch index (epoch i = samples
[i*spe : (i+1)*spe], spe = round(fs*30)); the shorter of the two lengths wins,
so a signal and hypnogram of slightly different length still line up.

Unknown (code 9) epochs are ignored. Pure numpy — no SciPy — so it runs in the
same minimal environment as the rest of the viewer.
"""
import numpy as np

EPOCH_SEC = 30.0
STAGE_ORDER = [5, 3, 2, 1, 4]                       # Wake, N1, N2, N3, REM
STAGE_LABEL = {5: "Wake", 3: "N1", 2: "N2", 1: "N3", 4: "REM"}
STAGE_NAME2CODE = {v: k for k, v in STAGE_LABEL.items()}
STAGE_COLOR_KEY = {5: "wake", 3: "n1", 2: "n2", 1: "n3", 4: "rem"}
# EEG relative-power bands (Hz). Sigma = spindle band.
BANDS = [("delta", 0.5, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 12.0),
         ("sigma", 12.0, 16.0), ("beta", 16.0, 30.0)]
_TOTAL_BAND = (0.5, 30.0)                            # denominator for relative power
MAX_EPOCH_POINTS = 900                               # example-epoch transport cap


def _decimate_envelope(arr, n_out):
    """Min/max-envelope downsample of `arr` to ~n_out points (spikes survive)."""
    arr = np.asarray(arr, float)
    n = arr.size
    if n <= n_out:
        xs = np.arange(n)
        ys = [float(v) if np.isfinite(v) else None for v in arr]
        return xs, ys
    buckets = max(1, n_out // 2)
    edges = np.linspace(0, n, buckets + 1, dtype=int)
    xs, ys = [], []
    for i in range(buckets):
        a, b = edges[i], edges[i + 1]
        if b <= a:
            continue
        seg = arr[a:b]
        finite = seg[np.isfinite(seg)]
        if finite.size == 0:
            xs.append(a); ys.append(None); continue
        imn = a + int(np.nanargmin(seg)); imx = a + int(np.nanargmax(seg))
        lo, hi = (imn, imx) if imn < imx else (imx, imn)
        xs.append(lo); ys.append(float(arr[lo]))
        xs.append(hi); ys.append(float(arr[hi]))
    return np.array(xs), ys


def _longest_bout_mid(mask):
    """Index of the middle epoch of the longest contiguous True run in `mask`,
    or None if `mask` is all False."""
    best_len, best_mid, run_start = 0, None, None
    for i, v in enumerate(mask):
        if v and run_start is None:
            run_start = i
        elif not v and run_start is not None:
            if i - run_start > best_len:
                best_len, best_mid = i - run_start, (run_start + i - 1) // 2
            run_start = None
    if run_start is not None and mask.size - run_start > best_len:
        best_mid = (run_start + mask.size - 1) // 2
    return best_mid


def _band_powers(epochs, fs):
    """Mean relative band power across `epochs` (rows = epochs, cols = samples).

    Per epoch: rFFT power spectrum, integrate each band, divide by total
    0.5-30 Hz power; then average the per-epoch relative powers. Returns a dict
    band -> mean relative power (0..1), or None if fs is too low to resolve beta.
    """
    if epochs.size == 0 or fs < 2 * _TOTAL_BAND[1]:   # Nyquist below 30 Hz -> skip
        return None
    spe = epochs.shape[1]
    freqs = np.fft.rfftfreq(spe, d=1.0 / fs)
    acc = {b[0]: [] for b in BANDS}
    total_mask = (freqs >= _TOTAL_BAND[0]) & (freqs < _TOTAL_BAND[1])
    band_masks = [(nm, (freqs >= lo) & (freqs < hi)) for nm, lo, hi in BANDS]
    for row in epochs:
        if not np.all(np.isfinite(row)):
            row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
        row = row - row.mean()                        # remove DC before FFT
        p = np.abs(np.fft.rfft(row)) ** 2
        tot = float(p[total_mask].sum())
        if tot <= 0:
            continue
        for nm, m in band_masks:
            acc[nm].append(float(p[m].sum()) / tot)
    out = {}
    for nm in acc:
        out[nm] = round(float(np.mean(acc[nm])), 4) if acc[nm] else None
    return out


def _epoch_bounds(codes, code, n_ep):
    """Contiguous [start_epoch, end_epoch) runs of `code` within codes[:n_ep]."""
    runs = []
    start = None
    for i in range(n_ep):
        if codes[i] == code and start is None:
            start = i
        elif codes[i] != code and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, n_ep))
    return runs


def stage_concat_signal(data, fs, stage_codes, stage_name, t0=None, t1=None,
                        points=2500):
    """Concatenate every epoch of one stage into a single continuous trace.

    All epochs assigned to `stage_name` (in the preprocessed staging) are joined
    end to end into one synthetic signal whose own timeline runs 0..(n_epochs*30)s.
    Optionally windowed to [t0, t1] on that concatenated timeline BEFORE decimating,
    so a short window returns near-raw samples (same zoom model as /api/signals).

    Returns the decimated trace plus:
      * `bouts`: the original-night bouts that make up this stage, each with its
        position on the concatenated timeline and its real clock time in the night,
        so the UI can draw seams and report where a sample truly came from;
      * `total_sec`: length of the concatenated stage timeline.
    """
    data = np.asarray(data, float)
    fs = float(fs)
    codes = np.rint(np.asarray(stage_codes, float)).astype(int)
    code = STAGE_NAME2CODE.get(stage_name)
    if code is None:
        return {"ok": False, "error": f"unknown stage {stage_name}"}
    if fs <= 0 or data.size == 0 or codes.size == 0:
        return {"ok": False, "error": "no signal or staging to align"}
    spe = int(round(fs * EPOCH_SEC))
    if spe < 1:
        return {"ok": False, "error": "sampling rate too low"}
    n_ep = min(codes.size, data.size // spe)
    runs = _epoch_bounds(codes, code, n_ep)
    if not runs:
        return {"ok": False, "error": f"no {stage_name} epochs in this recording"}

    # Build the concatenated sample array and a per-bout position map. Bouts keep
    # their real night start time (concat is a view for inspection, not re-timing).
    segments = []
    bouts = []
    concat_off = 0                                    # sample offset in concat array
    for (s_ep, e_ep) in runs:
        a, b = s_ep * spe, e_ep * spe
        seg = data[a:b]
        segments.append(seg)
        n = seg.size
        bouts.append({
            "concat_start_s": round(concat_off / fs, 3),
            "concat_end_s": round((concat_off + n) / fs, 3),
            "night_start_s": round(a / fs, 1),
            "night_start_min": round(a / fs / 60.0, 2),
            "epochs": int(e_ep - s_ep),
        })
        concat_off += n
    concat = np.concatenate(segments) if segments else np.array([])
    total_sec = concat.size / fs if fs else 0.0

    # optional zoom window on the concatenated timeline
    win = None
    if t0 is not None and t1 is not None:
        try:
            a = min(max(0.0, float(t0)), total_sec)
            b = min(max(0.0, float(t1)), total_sec)
            if b - a > 1e-6:
                win = (a, b)
        except (TypeError, ValueError):
            win = None
    t_off = 0.0
    view = concat
    if win:
        i0 = max(0, min(int(np.floor(win[0] * fs)), concat.size))
        i1 = max(i0 + 1, min(int(np.ceil(win[1] * fs)), concat.size))
        view = concat[i0:i1]
        t_off = i0 / fs

    xi, ys = _decimate_envelope(view, points)
    t = [round(float(x) / fs + t_off, 3) for x in xi]
    out = {
        "ok": True, "stage": stage_name, "code": code,
        "color_key": STAGE_COLOR_KEY[code], "fs": fs,
        "n_epochs": sum(e - s for s, e in runs),
        "n_bouts": len(runs),
        "total_sec": round(total_sec, 1),
        "total_min": round(total_sec / 60.0, 1),
        "bouts": bouts,
        "n_samples": int(view.size),
        "t": t, "y": ys,
    }
    if win:
        out["window"] = [round(win[0], 3), round(win[1], 3)]
    return out


def stage_signal_profile(data, fs, stage_codes, role="other",
                         max_epoch_points=MAX_EPOCH_POINTS):
    """Per-stage summary of one channel. See module docstring.

    data: 1-D channel samples (whole recording). fs: Hz. stage_codes: int array,
    one (preprocessed) code per 30 s epoch. role: coarse channel role for gating
    the EEG band-power panel. Returns a JSON-serialisable dict."""
    data = np.asarray(data, float)
    fs = float(fs)
    codes = np.rint(np.asarray(stage_codes, float)).astype(int)
    if fs <= 0 or data.size == 0 or codes.size == 0:
        return {"ok": False, "error": "no signal or staging to align"}
    spe = int(round(fs * EPOCH_SEC))
    if spe < 1:
        return {"ok": False, "error": "sampling rate too low"}
    n_ep = min(codes.size, data.size // spe)
    if n_ep < 1:
        return {"ok": False, "error": "signal shorter than one epoch"}
    codes = codes[:n_ep]
    epochs = data[:n_ep * spe].reshape(n_ep, spe)     # (epoch, sample-in-epoch)

    is_eeg = role == "eeg"
    finite_all = data[:n_ep * spe]
    finite_all = finite_all[np.isfinite(finite_all)]
    overall = {
        "mean": round(float(finite_all.mean()), 3) if finite_all.size else None,
        "std": round(float(finite_all.std()), 3) if finite_all.size else None,
        "min": round(float(finite_all.min()), 3) if finite_all.size else None,
        "max": round(float(finite_all.max()), 3) if finite_all.size else None,
    }

    stages = []
    for code in STAGE_ORDER:
        mask = codes == code
        n = int(mask.sum())
        if n == 0:
            continue
        rows = epochs[mask]
        flat = rows.reshape(-1)
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            continue
        entry = {
            "code": code, "stage": STAGE_LABEL[code],
            "color_key": STAGE_COLOR_KEY[code],
            "n_epochs": n, "minutes": round(n * EPOCH_SEC / 60.0, 1),
            "pct_epochs": round(100.0 * n / n_ep, 1),
            "mean": round(float(flat.mean()), 3),
            "std": round(float(flat.std()), 3),
            "min": round(float(flat.min()), 3),
            "max": round(float(flat.max()), 3),
            "p5": round(float(np.percentile(flat, 5)), 3),
            "p95": round(float(np.percentile(flat, 95)), 3),
        }
        # representative example epoch: middle of this stage's longest bout
        mid = _longest_bout_mid(mask)
        if mid is not None:
            seg = epochs[mid]
            xi, ys = _decimate_envelope(seg, max_epoch_points)
            entry["example"] = {
                "epoch_index": int(mid),
                "start_min": round(mid * EPOCH_SEC / 60.0, 2),
                "t": [round(float(x) / fs, 3) for x in xi],   # seconds within epoch
                "y": ys,
            }
        if is_eeg:
            bp = _band_powers(rows, fs)
            if bp is not None:
                entry["bandpower"] = bp
        stages.append(entry)

    return {
        "ok": True,
        "role": role, "is_eeg": is_eeg, "fs": fs,
        "epoch_sec": EPOCH_SEC, "n_epochs": n_ep,
        "bands": [b[0] for b in BANDS] if is_eeg else [],
        "overall": overall,
        "stages": stages,
    }
