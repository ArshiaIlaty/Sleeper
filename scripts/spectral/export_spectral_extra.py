#!/usr/bin/env python3
"""Extract spectral-SHAPE features the champion band-power block never captured.

Reuses the exact harness of `export_epoch_features.py` (central-EEG pick, CAISR
staging, shard/resume, edfio lazy read) but, instead of the 5 band powers, computes
four spectral-shape families per epoch from the full Welch PSD (same nperseg as the
shipped `eeg_spectral._epoch_band_powers`, so the frequency grid matches the
band-power block), then aggregates to ONE row per recording keyed by
(dataset, bids_folder, session):

  * aperiodic 1/f      exponent (= -slope of log10 PSD vs log10 f, 2-30 Hz) + offset
  * spectral entropy   normalized Shannon entropy of the PSD over 0.5-30 Hz
  * SEF95 / med_freq   spectral edge (95%) and median frequency (Hz)
  * across-night var.   std of each per-epoch metric across the night (within-N2 and
                        pooled-sleep) -> the "spectral instability" family

Emitted per recording, per metric m in {aper_exp, aper_off, spec_ent, sef95, med_freq}:
  sp__{m}_wake, sp__{m}_n2, sp__{m}_n3, sp__{m}_rem   (stage means)
  sp__{m}_n2std                                        (within-N2 across-night std)
  sp__{m}_sleepstd                                     (all-sleep across-night std)
plus QC counts sp__n_{wake,n2,n3,rem}. Per-recording rows stay on the box (DUA);
only the aggregate A/B result leaves. Pure stdlib + numpy + scipy + edfio.
"""
import argparse
import csv
import os
import sys
import warnings

import numpy as np
import edfio
from scipy.signal import welch

from sources import get_dataset, REGISTRY
from app import channel_role, _caisr_stage_codes, _NK_EEG_PREFER
import nk_features as nkf
import eeg_spectral as es

_EEG_ROLES = {"eeg"}
BAND_LO, BAND_HI = es._TOTAL_BAND          # (0.5, 30.0) Hz -- same band as band powers
FIT_LO, FIT_HI = 2.0, 30.0                 # aperiodic log-log fit range (skip the sub-2 Hz knee)
METRICS = ["aper_exp", "aper_off", "spec_ent", "sef95", "med_freq"]
POOLS = ["wake", "n2", "n3", "rem"]        # stage means we keep
SLEEP_TAGS = {"n1", "n2", "n3", "rem"}

ID_COLS = ["dataset", "bids_folder", "session", "site", "label", "eeg_channel"]
FEAT_COLS = []
for m in METRICS:
    for p in POOLS:
        FEAT_COLS.append(f"sp__{m}_{p}")
    FEAT_COLS.append(f"sp__{m}_n2std")
    FEAT_COLS.append(f"sp__{m}_sleepstd")
QC_COLS = [f"sp__n_{p}" for p in POOLS]
OUT_COLS = ID_COLS + FEAT_COLS + QC_COLS


def _r(x, nd=5):
    if x is None:
        return None
    v = float(x)
    return round(v, nd) if np.isfinite(v) else None


def _load_eeg(edf):
    labels = [s.label.strip() for s in edf.signals]
    roles = {l: channel_role(l) for l in labels}
    eeg = nkf._pick_channel(labels, roles, _EEG_ROLES, prefer=_NK_EEG_PREFER)
    if eeg is None:
        return None, None, None
    for s in edf.signals:
        if s.label.strip() == eeg:
            return eeg, np.asarray(s.data, float), float(s.sampling_frequency)
    return None, None, None


def _epoch_shape(seg, fs):
    """Per-epoch spectral-shape metrics from the full Welch PSD, or None."""
    seg = seg[np.isfinite(seg)]
    if seg.size < int(fs * 2) or fs < 2 * BAND_HI:
        return None
    seg = seg - seg.mean()
    nper = int(min(seg.size, max(fs * 2, 256)))       # matches _epoch_band_powers
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f, pxx = welch(seg, fs=fs, nperseg=nper)
    except Exception:
        return None
    band = (f >= BAND_LO) & (f < BAND_HI) & np.isfinite(pxx) & (pxx > 0)
    fb, pb = f[band], pxx[band]
    if fb.size < 8:
        return None
    # spectral entropy: PSD as a probability mass over the band, normalized to [0,1]
    p = pb / pb.sum()
    ent = float(-np.sum(p * np.log(p)) / np.log(p.size))
    # cumulative power (trapezoid) -> SEF95 + median frequency
    cum = np.concatenate([[0.0], np.cumsum((pb[1:] + pb[:-1]) * 0.5 * np.diff(fb))])
    if cum[-1] <= 0:
        return None
    cum /= cum[-1]
    sef95 = float(np.interp(0.95, cum, fb))
    medf = float(np.interp(0.50, cum, fb))
    # aperiodic 1/f: OLS of log10(PSD) on log10(f) over the fit band
    fitm = (fb >= FIT_LO) & (fb <= FIT_HI)
    aexp = aoff = None
    if int(fitm.sum()) >= 5:
        slope, intercept = np.polyfit(np.log10(fb[fitm]), np.log10(pb[fitm]), 1)
        aexp = float(-slope)          # positive = steeper 1/f falloff
        aoff = float(intercept)       # extrapolated log10 power at 1 Hz
    return {"aper_exp": aexp, "aper_off": aoff, "spec_ent": ent,
            "sef95": sef95, "med_freq": medf}


def _aggregate(per_stage):
    """per_stage: {tag: {metric: [values]}} -> flat feature dict."""
    out = {c: None for c in FEAT_COLS + QC_COLS}
    # per-stage epoch count (metrics share the same epoch set within a stage)
    for p in POOLS:
        vals = per_stage.get(p, {})
        out[f"sp__n_{p}"] = max((len(v) for v in vals.values()), default=0)
    for m in METRICS:
        for p in POOLS:
            arr = np.asarray(per_stage.get(p, {}).get(m, []), float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                out[f"sp__{m}_{p}"] = _r(np.mean(arr))
        n2 = np.asarray(per_stage.get("n2", {}).get(m, []), float)
        n2 = n2[np.isfinite(n2)]
        if n2.size >= 3:
            out[f"sp__{m}_n2std"] = _r(np.std(n2))
        sleep = []
        for t in SLEEP_TAGS:
            sleep.extend(per_stage.get(t, {}).get(m, []))
        sleep = np.asarray(sleep, float)
        sleep = sleep[np.isfinite(sleep)]
        if sleep.size >= 5:
            out[f"sp__{m}_sleepstd"] = _r(np.std(sleep))
    return out


def _recording_row(ds, rec):
    bids = rec.get("BidsFolder", "")
    site, sess = rec.get("SiteID", ""), rec.get("SessionID", "1")
    codes, _ = _caisr_stage_codes(ds, rec, bids)
    if codes is None:
        return None
    try:
        f = ds.physio_path(site, bids, sess)
    except Exception:
        f = None
    if not f:
        return None
    edf = edfio.read_edf(f, lazy_load_data=True)
    eeg_ch, sig, fs = _load_eeg(edf)
    ids = {"dataset": ds.key, "bids_folder": bids, "session": sess, "site": site,
           "label": rec.get("Cognitive_Impairment", ""), "eeg_channel": eeg_ch or ""}
    if eeg_ch is None:
        return {**ids, **{c: None for c in FEAT_COLS + QC_COLS}}
    sig = np.asarray(sig, float)
    codes = np.rint(np.asarray(codes, float)).astype(int)
    if not es._SCIPY_OK or fs <= 0 or sig.size == 0 or codes.size == 0:
        return {**ids, **{c: None for c in FEAT_COLS + QC_COLS}}
    spe, n_ep, codes = es._align_epochs(sig.size, fs, codes)
    if spe is None or n_ep < 1 or fs < 2 * BAND_HI:
        return {**ids, **{c: None for c in FEAT_COLS + QC_COLS}}
    per_stage = {}
    for e in range(n_ep):
        tag = es.POOL_TAG.get(int(codes[e]))
        if tag is None:
            continue
        met = _epoch_shape(sig[e * spe:(e + 1) * spe], fs)
        if met is None:
            continue
        d = per_stage.setdefault(tag, {m: [] for m in METRICS})
        for m in METRICS:
            if met[m] is not None:
                d[m].append(met[m])
    return {**ids, **_aggregate(per_stage)}


def _done(path):
    done = set()
    if os.path.exists(path):
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                done.add((r.get("bids_folder", ""), r.get("session", "")))
    return done


def run(dataset_key, outdir, limit=None, resume=False, shard=0, nshards=1, progress=None):
    ds = get_dataset(dataset_key)
    rows = ds.demographics()
    if nshards > 1:
        rows = [r for i, r in enumerate(rows) if i % nshards == shard]
    if limit:
        rows = rows[:limit]
    os.makedirs(outdir, exist_ok=True)
    suffix = dataset_key if nshards == 1 else f"{dataset_key}_s{shard}"
    out_path = os.path.join(outdir, f"spectral_extra_{suffix}.csv")
    done = _done(out_path) if resume else set()
    mode = "a" if (resume and os.path.exists(out_path)) else "w"

    n_rec = n_skip = n_nofile = n_err = n_noeeg = 0
    with open(out_path, mode, newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_COLS, extrasaction="ignore")
        if mode == "w":
            w.writeheader()
        for i, rec in enumerate(rows):
            bids, sess = rec.get("BidsFolder", ""), rec.get("SessionID", "1")
            if (bids, sess) in done:
                n_skip += 1
                continue
            try:
                row = _recording_row(ds, rec)
            except Exception as e:
                n_err += 1
                if progress:
                    progress(f"  ! {bids}: {type(e).__name__}: {e}")
                continue
            if row is None:
                n_nofile += 1
                continue
            if not row.get("eeg_channel"):
                n_noeeg += 1
            w.writerow(row)
            n_rec += 1
            if progress and (i + 1) % 25 == 0:
                fh.flush()
                progress(f"  {suffix}: {i+1}/{len(rows)} scanned, {n_rec} rows, "
                         f"{n_noeeg} no-eeg, {n_nofile} no-file, {n_err} err")
    summary = {"out": out_path, "n_rows": n_rec, "n_noeeg": n_noeeg,
               "n_nofile": n_nofile, "n_err": n_err, "n_skip": n_skip}
    if progress:
        progress(f"DONE {summary}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="standard", choices=list(REGISTRY.keys()))
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    s = run(args.dataset, args.outdir, limit=args.limit, resume=args.resume,
            shard=args.shard, nshards=args.nshards,
            progress=lambda m: print(m, file=sys.stderr, flush=True))
    print(f"Wrote {s['n_rows']} rows to {s['out']} "
          f"({s['n_noeeg']} no-eeg, {s['n_nofile']} no-file, {s['n_err']} err)")


if __name__ == "__main__":
    main()
