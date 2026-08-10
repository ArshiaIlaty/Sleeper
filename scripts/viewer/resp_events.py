"""Respiratory-event features from the CAISR resp annotation channel.

`resp_caisr` is a 1 Hz per-second label stream with codes
{1: obstructive apnea, 2: central apnea, 4: hypopnea, 5: RERA}. From it we derive:

  * **Counts** of each event type and the **AHI** (apneas + hypopneas per hour of
    sleep) and **RDI** (adding RERAs).
  * **Event durations** — mean / median / max seconds, since a run of identical
    consecutive labels is one event of that length.
  * **Recovery features** (needs SpO2): after each apnea/hypopnea ends, the
    **post-event SpO2 overshoot** (how far saturation rebounds above the
    pre-event baseline — a marker of the ventilatory/autonomic surge that
    terminates the event) and the **respiratory recovery time** (seconds from the
    event's oxygen nadir back up to within 1% of the pre-event baseline). Blunted
    overshoot / prolonged recovery indicate impaired arousal and chemoreflex
    control.

Durations/counts are pure-numpy on the 1 Hz code stream. Recovery features align a
(scale-normalised) SpO2 signal to the same seconds axis; if SpO2 is missing they
degrade to None. Everything is JSON-serialisable and never raises.
"""
import numpy as np

RESP_LABELS = {1: "obstructive_apnea", 2: "central_apnea", 4: "hypopnea", 5: "RERA"}
APNEA_HYPOPNEA = (1, 2, 4)                     # counted in AHI
AHI_RDI = (1, 2, 4, 5)                         # counted in RDI
RECOVERY_WIN_S = 45.0                          # window after event end to track recovery
BASELINE_WIN_S = 30.0                          # pre-event SpO2 baseline window
RECOVERY_TOL = 1.0                             # % of baseline "recovered"


def _finite(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _r(x, nd=3):
    v = _finite(x)
    return None if v is None else round(v, nd)


def _events(codes, code):
    """(start_s, end_exclusive_s) runs of `code` in a 1 Hz int stream."""
    mask = (codes == code)
    if mask.size == 0:
        return []
    idx = np.flatnonzero(np.diff(np.concatenate(([0], mask.view(np.int8), [0]))))
    return [(int(idx[i]), int(idx[i + 1])) for i in range(0, len(idx), 2)]


def resp_event_features(resp_codes, tst_hours=None, spo2=None, spo2_fs=None):
    """Respiratory-event counts / durations / recovery.

    resp_codes: 1 Hz int label stream (the CAISR resp channel).
    tst_hours: total sleep time (h) for AHI/RDI; falls back to recording hours.
    spo2, spo2_fs: optional scale-normalised SpO2 (percent) + its sampling rate,
    for the recovery features. Returns a JSON-serialisable dict.
    """
    codes = np.rint(np.asarray(resp_codes, float)).astype(int)
    out = {}
    if codes.size == 0:
        return {"ok": False, "error": "no resp annotation channel"}
    rec_hours = codes.size / 3600.0            # 1 Hz -> seconds/3600
    hours = tst_hours if (tst_hours and tst_hours > 0) else rec_hours

    all_events = []                            # (start, end, code) for apnea+hypopnea
    n_by_type = {}
    dur_by_type = {}
    for code, name in RESP_LABELS.items():
        evs = _events(codes, code)
        n_by_type[name] = len(evs)
        durs = [float(e - s) for s, e in evs]
        dur_by_type[name] = durs
        out[f"n_{name}"] = len(evs)
        out[f"{name}_dur_mean_s"] = _r(np.mean(durs), 2) if durs else None
        out[f"{name}_dur_max_s"] = _r(np.max(durs), 1) if durs else None
        if code in APNEA_HYPOPNEA:
            all_events.extend((s, e, code) for s, e in evs)

    n_ah = sum(n_by_type[RESP_LABELS[c]] for c in APNEA_HYPOPNEA)
    n_rdi = sum(n_by_type[RESP_LABELS[c]] for c in AHI_RDI)
    out["n_apnea_hypopnea"] = int(n_ah)
    out["ahi"] = _r(n_ah / hours, 2) if hours > 0 else None
    out["rdi"] = _r(n_rdi / hours, 2) if hours > 0 else None

    # pooled apnea+hypopnea duration stats
    ah_durs = [float(e - s) for s, e, _ in all_events]
    out["event_dur_mean_s"] = _r(np.mean(ah_durs), 2) if ah_durs else None
    out["event_dur_median_s"] = _r(np.median(ah_durs), 1) if ah_durs else None
    out["event_dur_max_s"] = _r(np.max(ah_durs), 1) if ah_durs else None
    out["tst_hours_used"] = _r(hours, 3)

    # ---- recovery features (need SpO2) ----
    over, rec_time = None, None
    if spo2 is not None and spo2_fs and spo2_fs > 0 and all_events:
        over, rec_time = _recovery(all_events, codes.size, spo2, float(spo2_fs))
    out["post_event_overshoot"] = over
    out["resp_recovery_time_s"] = rec_time
    out["ok"] = True
    return out


def _recovery(events, n_seconds, spo2, fs):
    """Mean post-event SpO2 overshoot (%) and recovery time (s) across events.

    For each event [s, e) (seconds), take a pre-event baseline (median SpO2 over
    the BASELINE_WIN_S before it), find the oxygen nadir within the event +
    recovery window, then:
      * overshoot = max SpO2 in the recovery window - baseline (rebound above
        baseline; can be ~0 if there is no overshoot);
      * recovery time = seconds from the nadir until SpO2 first returns within
        RECOVERY_TOL of baseline.
    SpO2 is resampled onto the 1 Hz seconds axis so indexing matches the codes.
    """
    # resample SpO2 to 1 Hz (one value per second), scale already normalised
    x = np.asarray(spo2, float)
    secs = np.arange(n_seconds)
    src_t = np.arange(x.size) / fs
    finite = np.isfinite(x)
    if finite.sum() < 10:
        return None, None
    s1 = np.interp(secs, src_t[finite], x[finite])   # 1 Hz SpO2, percent

    overs, times = [], []
    for (s, e, _code) in events:
        b0 = max(0, s - int(BASELINE_WIN_S))
        base = np.median(s1[b0:s]) if s > b0 else np.median(s1[:s + 1]) if s > 0 else None
        if base is None or not np.isfinite(base):
            continue
        w1 = min(n_seconds, e + int(RECOVERY_WIN_S))
        win = s1[s:w1]
        if win.size == 0:
            continue
        nadir_i = int(np.argmin(win))
        peak = float(np.max(win))
        overs.append(peak - float(base))
        # recovery time: from nadir until back within tol of baseline
        tail = win[nadir_i:]
        recovered = np.where(tail >= base - RECOVERY_TOL)[0]
        if recovered.size:
            times.append(float(recovered[0]))
    over = _r(np.mean(overs), 3) if overs else None
    rec = _r(np.mean(times), 2) if times else None
    return over, rec
