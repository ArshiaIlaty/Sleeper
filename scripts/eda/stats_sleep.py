"""Sleep-architecture statistics from CAISR algorithmic annotations.

CAISR annotation EDFs are small (~500 KB) so we read them fully. Per recording we
derive standard PSG summaries, then aggregate across the cohort and per site:

  - recording duration, total sleep time (TST), sleep efficiency, WASO
  - stage percentages (Wake/N1/N2/N3/REM) and time-in-stage (minutes)
  - sleep-onset / REM / N3 latencies, stage-transition rate, stage entropy
  - event indices per hour: AHI (respiratory), arousal index, PLMI (limb)
  - respiratory event subtypes (obstructive/central apnea, hypopnea, RERA)
  - fraction of "Unknown"(9) epochs — annotation-quality signal
  - CAISR-vs-expert stage agreement (where both files exist)

Codes (from team_code.py): stage {1:N3,2:N2,3:N1,4:REM,5:Wake,9:Unknown};
resp {1:obstructive,2:central,4:hypopnea,5:RERA}; limb {1:isolated,2:periodic}.
"""
import os
import glob
from collections import defaultdict

import numpy as np
import edfio

from common import (CAISR_DIR, EXPERT_DIR, SITE_NAMES, list_sites, list_edfs,
                    parse_record_id, STAGE_CODES, ASLEEP_CODES, RESP_CODES,
                    LIMB_CODES, EPOCH_SEC)
from statutils import numeric_summary, value_counts


def _load_channels(path):
    """Return {label: np.ndarray} for a small annotation EDF."""
    edf = edfio.read_edf(path, lazy_load_data=False)
    return {s.label.strip(): np.asarray(s.data, dtype=float) for s in edf.signals}


def _event_rate_per_hour(sig, hours, codes=None):
    """Onset (rising-edge) count per hour.

    ``codes`` selects which event classes count: None = any positive code;
    an int or iterable of ints = only those codes (membership test).
    """
    if sig is None or len(sig) == 0 or hours <= 0:
        return np.nan
    if codes is None:
        mask = sig > 0
    else:
        code_set = [codes] if np.isscalar(codes) else list(codes)
        mask = np.isin(sig, code_set)
    s = mask.astype(int)
    return float(np.count_nonzero(np.diff(s, prepend=0) == 1) / hours)


def _psg_summary(chans):
    """Per-recording PSG summary dict from CAISR channels."""
    stage = chans.get("stage_caisr")
    resp = chans.get("resp_caisr")
    arousal = chans.get("arousal_caisr")
    limb = chans.get("limb_caisr")

    # Recording duration: prefer 1 Hz resp length, else 30 s epochs.
    if resp is not None and len(resp) > 0:
        hours = len(resp) / 3600.0
    elif stage is not None and len(stage) > 0:
        hours = len(stage) * EPOCH_SEC / 3600.0
    else:
        hours = np.nan

    d = {"duration_hours": float(hours) if np.isfinite(hours) else None}
    # Hours of sleep (TST); AHI/arousal/PLMI are conventionally per hour of
    # sleep, not per hour of recording. Filled in the stage block below.
    tst_hours = np.nan

    if stage is not None and len(stage) > 0:
        st = np.rint(stage).astype(int)  # round, don't truncate (5.0 must stay 5)
        n_epochs = st.size
        n_unknown = int(np.count_nonzero(st == 9))
        valid = st[st != 9]
        d["n_epochs"] = int(n_epochs)
        d["pct_unknown_epochs"] = round(100.0 * n_unknown / n_epochs, 2)

        # stage percentages over scored (non-unknown) epochs
        scored = valid.size
        for code, name in STAGE_CODES.items():
            if code == 9:
                continue
            d[f"pct_{name}"] = round(100.0 * np.count_nonzero(valid == code) / scored, 2) if scored else None

        asleep_mask = np.isin(st, ASLEEP_CODES)
        n_asleep = int(np.count_nonzero(asleep_mask))
        tst_min = float(n_asleep * EPOCH_SEC / 60.0)
        tst_hours = tst_min / 60.0
        d["tst_min"] = round(tst_min, 1)
        # sleep efficiency = TST / time-in-bed (all epochs)
        d["sleep_efficiency_pct"] = round(100.0 * n_asleep / n_epochs, 2) if n_epochs else None

        asleep_idx = np.where(asleep_mask)[0]
        onset = int(asleep_idx[0]) if asleep_idx.size else None
        d["sleep_latency_min"] = round(onset * EPOCH_SEC / 60.0, 1) if onset is not None else None
        if onset is not None:
            # WASO = Wake between sleep onset and the FINAL sleep epoch (wake
            # after the last sleep epoch is not counted).
            offset = int(asleep_idx[-1])
            waso = int(np.count_nonzero(st[onset:offset + 1] == 5))
            d["waso_min"] = round(waso * EPOCH_SEC / 60.0, 1)
            rem_idx = np.where(st == 4)[0]
            n3_idx = np.where(st == 1)[0]
            d["rem_latency_min"] = round((rem_idx[0] - onset) * EPOCH_SEC / 60.0, 1) if rem_idx.size else None
            d["n3_latency_min"] = round((n3_idx[0] - onset) * EPOCH_SEC / 60.0, 1) if n3_idx.size else None
        d["transitions_per_hr"] = round(float(np.count_nonzero(np.diff(st[st != 9]) != 0)) / hours, 2) if hours > 0 else None

        # stage entropy (normalised) over N3,N2,N1,REM,Wake
        counts = np.array([np.count_nonzero(valid == c) for c in (1, 2, 3, 4, 5)], float)
        p = counts / counts.sum() if counts.sum() > 0 else counts
        p = p[p > 0]
        d["stage_entropy"] = round(float(-np.sum(p * np.log(p)) / np.log(len(p))), 3) if p.size > 1 else 0.0
    else:
        d["n_epochs"] = 0
        d["stage_missing"] = True

    # Event indices are per hour of SLEEP (TST) by clinical convention; fall
    # back to recording hours only if staging (hence TST) is unavailable.
    per_hr = tst_hours if np.isfinite(tst_hours) and tst_hours > 0 else hours
    # AHI = apneas + hypopneas only (obstructive 1, central 2, hypopnea 4);
    # RERA (5) is deliberately excluded (it belongs to RDI, not AHI).
    d["ahi"] = _event_rate_per_hour(resp, per_hr, codes=(1, 2, 4))
    d["arousal_index"] = _event_rate_per_hour(arousal, per_hr)
    # PLMI = periodic limb movements only (code 2), not isolated (1).
    d["plmi"] = _event_rate_per_hour(limb, per_hr, codes=2)
    for code, name in RESP_CODES.items():
        d[f"resp_{name}_idx"] = _event_rate_per_hour(resp, per_hr, codes=code)
    for code, name in LIMB_CODES.items():
        d[f"{name}_idx"] = _event_rate_per_hour(limb, per_hr, codes=code)

    return d


def _expert_agreement(caisr_stage_path, site):
    """CAISR-vs-expert epoch-level stage agreement, if the expert file exists."""
    base = os.path.basename(caisr_stage_path).replace("_caisr_annotations.edf", "")
    exp = os.path.join(EXPERT_DIR, site, base + "_expert_annotations.edf")
    if not os.path.exists(exp):
        return None
    try:
        ec = _load_channels(exp)
        cc = _load_channels(caisr_stage_path)
    except Exception:
        return None
    a = cc.get("stage_caisr")
    b = ec.get("stage_expert")
    if a is None or b is None:
        return None
    n = min(len(a), len(b))
    if n == 0:
        return None
    a, b = a[:n].astype(int), b[:n].astype(int)
    mask = (a != 9) & (b != 9)  # both scored
    if not mask.any():
        return None
    return float(np.mean(a[mask] == b[mask]))


# Numeric fields to aggregate across recordings.
_AGG_FIELDS = [
    "duration_hours", "tst_min", "sleep_efficiency_pct", "waso_min",
    "sleep_latency_min", "rem_latency_min", "n3_latency_min",
    "transitions_per_hr", "stage_entropy", "pct_unknown_epochs",
    "pct_Wake", "pct_N1", "pct_N2", "pct_N3", "pct_REM",
    "ahi", "arousal_index", "plmi",
    "resp_obstructive_apnea_idx", "resp_central_apnea_idx",
    "resp_hypopnea_idx", "resp_RERA_idx",
    "isolated_limb_idx", "periodic_limb_idx",
]


def run(limit_per_site=None, expert_agreement_sample=60, progress=None):
    sites = list_sites(CAISR_DIR)
    per_site = {}
    pooled = defaultdict(list)
    agreements = []
    n_files = 0
    n_ok = 0
    n_stage_missing = 0
    errors = []

    for site in sites:
        files = list_edfs(CAISR_DIR, site)
        if limit_per_site:
            files = files[:limit_per_site]
        site_vals = defaultdict(list)
        site_agree = []
        for i, f in enumerate(files):
            n_files += 1
            try:
                chans = _load_channels(f)
                s = _psg_summary(chans)
            except Exception as e:
                errors.append({"file": os.path.basename(f), "error": str(e)})
                continue
            n_ok += 1
            if s.get("stage_missing"):
                n_stage_missing += 1
            for k in _AGG_FIELDS:
                v = s.get(k)
                # Record NaN (not skip) when a metric is absent for this
                # recording, so numeric_summary's missing_pct reflects true
                # coverage across ALL read recordings rather than always 0.
                val = v if (v is not None and np.isfinite(v)) else np.nan
                site_vals[k].append(val)
                pooled[k].append(val)
            # expert agreement on a capped sample (full read of a 2nd file each)
            if expert_agreement_sample and i < expert_agreement_sample:
                ag = _expert_agreement(f, site)
                if ag is not None:
                    site_agree.append(ag)
                    agreements.append(ag)
            if progress and (i + 1) % 100 == 0:
                progress(f"  {site}: {i+1}/{len(files)} CAISR files read")

        per_site[site] = {
            "site_name": SITE_NAMES.get(site, site),
            "n_files": len(files),
            "summaries": {k: numeric_summary(site_vals[k]) for k in _AGG_FIELDS},
            "caisr_expert_stage_agreement": numeric_summary(site_agree) if site_agree else None,
        }

    return {
        "n_files": n_files,
        "n_read_ok": n_ok,
        "n_stage_missing": n_stage_missing,
        "pooled_summaries": {k: numeric_summary(pooled[k]) for k in _AGG_FIELDS},
        "caisr_expert_stage_agreement_pooled": numeric_summary(agreements) if agreements else None,
        "per_site": per_site,
        "n_errors": len(errors),
        "errors": errors[:20],
    }


if __name__ == "__main__":
    import json, sys
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print(json.dumps(run(limit_per_site=lim, progress=lambda m: print(m, file=sys.stderr)), indent=2))
