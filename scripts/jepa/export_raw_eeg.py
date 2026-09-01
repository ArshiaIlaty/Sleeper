"""Extract raw multichannel-EEG epoch tensors for the raw-signal JEPA.

For each recording: decode the six standard scalp derivations (F3,F4,C3,C4,O1,O2)
present at all three sites, common-average re-reference (CAR, montage-robust),
resample every channel to a common rate, robustly per-channel z-score, and slice
into stage-aligned 30 s epochs. Saves one compact .npz per recording:

    epochs   (n_epoch, 6, L)  float16     canonical channel order, CAR, z-scored
    stages   (n_epoch,)       int8        CAISR codes (5=W 3=N1 2=N2 1=N3 4=REM)
    chan_mask(6,)             int8        1 if that canonical derivation was present
    meta scalars: bids, site, label, fs, n_epoch

This is the (long-pole) preprocessing step; it is resumable (--resume) and shardable
(--shard/--nshards) exactly like the other cohort exporters. Nothing leaves the box.
"""
import os
import sys
import time
import argparse

import numpy as np
import edfio
from scipy import signal as sp_signal

from sources import get_dataset, REGISTRY
from app import channel_role, _caisr_stage_codes

TARGET_FS = 64.0                     # Hz; nyquist 32 covers delta..beta (<=30 Hz)
EPOCH_SEC = 30.0
L = int(round(TARGET_FS * EPOCH_SEC))          # 1920 samples/epoch
CANON = ["f3", "f4", "c3", "c4", "o1", "o2"]    # canonical channel order / slots
_MASTOID_CUES = ("m1", "m2")
NREM_CODES = (3, 2, 1)


def _is_scalp(label):
    low = label.lower()
    return any(c in low for c in
               ("f3", "f4", "c3", "c4", "o1", "o2", "cz", "pz", "fz",
                "fp1", "fp2", "f7", "f8", "p3", "p4", "t3", "t4"))


def _pick_eeg(edf):
    """Decode scalp EEG derivations (label, signal, fs), mastoids excluded."""
    out = []
    seen = set()
    for s in edf.signals:
        lab = s.label.strip()
        if lab in seen or channel_role(lab) != "eeg" or not _is_scalp(lab):
            continue
        low = lab.lower()
        toks = low.replace("-", " ").split()
        if any(t in _MASTOID_CUES for t in toks) and not any(
                c in low for c in CANON):
            continue
        out.append((lab, np.asarray(s.data, float), float(s.sampling_frequency)))
        seen.add(lab)
    return out


def _canonicalize(picked):
    """Map picked derivations to the 6 canonical slots by cue; return (sigs, fss, mask).
    Missing slot -> None (filled with zeros later)."""
    sigs = [None] * 6
    fss = [None] * 6
    for slot, cue in enumerate(CANON):
        for lab, data, fs in picked:
            if cue in lab.lower():
                sigs[slot] = data
                fss[slot] = fs
                break
    mask = np.array([0 if s is None else 1 for s in sigs], dtype=np.int8)
    return sigs, fss, mask


def _resample(x, fs_in):
    if abs(fs_in - TARGET_FS) < 1e-6:
        return x
    up, down = int(round(TARGET_FS)), int(round(fs_in))
    g = np.gcd(up, down)
    return sp_signal.resample_poly(x, up // g, down // g)


def _zscore(x):
    m = np.nanmean(x)
    s = np.nanstd(x)
    if not np.isfinite(s) or s < 1e-8:
        s = 1.0
    z = (np.nan_to_num(x, nan=m if np.isfinite(m) else 0.0) - (m if np.isfinite(m) else 0.0)) / s
    return np.clip(z, -8.0, 8.0)


def _build_epochs(sigs, fss, mask, codes):
    """CAR + resample + z-score -> (n_epoch, 6, L) float16 aligned to stage codes."""
    # resample present channels to TARGET_FS
    res = []
    for slot in range(6):
        if sigs[slot] is None:
            res.append(None)
        else:
            res.append(_resample(np.asarray(sigs[slot], float), fss[slot]))
    present = [r for r in res if r is not None]
    if len(present) < 5:                          # require >=5 of 6 derivations
        return None, None
    n = min(len(r) for r in present)
    # stack into (6, n) with zeros for missing slots
    stacked = np.zeros((6, n), dtype=np.float64)
    for slot in range(6):
        if res[slot] is not None:
            stacked[slot] = res[slot][:n]
    # CAR over PRESENT channels only
    pres_idx = np.where(mask == 1)[0]
    car_mean = stacked[pres_idx].mean(axis=0, keepdims=True)
    stacked[pres_idx] = stacked[pres_idx] - car_mean
    # per-channel robust z-score
    for slot in pres_idx:
        stacked[slot] = _zscore(stacked[slot])

    spe = int(round(TARGET_FS * EPOCH_SEC))
    n_ep = min(len(codes), n // spe)
    if n_ep < 1:
        return None, None
    ep = np.empty((n_ep, 6, spe), dtype=np.float16)
    for e in range(n_ep):
        ep[e] = stacked[:, e * spe:(e + 1) * spe].astype(np.float16)
    return ep, np.asarray(codes[:n_ep], dtype=np.int8)


def _one(ds, rec, out_dir):
    bids = rec.get("BidsFolder", "")
    site, sess = rec.get("SiteID", ""), rec.get("SessionID", "1")
    codes, _ = _caisr_stage_codes(ds, rec, bids)
    if codes is None:
        return "nostage"
    try:
        f = ds.physio_path(site, bids, sess)
    except Exception:
        f = None
    if not f:
        return "nofile"
    edf = edfio.read_edf(f, lazy_load_data=True)
    picked = _pick_eeg(edf)
    if len(picked) < 5:
        return "fewchan"
    sigs, fss, mask = _canonicalize(picked)
    codes = np.rint(np.asarray(codes, float)).astype(int)
    ep, st = _build_epochs(sigs, fss, mask, codes)
    if ep is None:
        return "shortsig"
    lab = str(rec.get("Cognitive_Impairment", "")).strip().lower()
    y = 1 if lab in ("1", "true", "yes") else 0
    np.savez_compressed(os.path.join(out_dir, f"{bids}.npz"),
                        epochs=ep, stages=st, chan_mask=mask,
                        bids=bids, site=site, label=y, fs=TARGET_FS,
                        n_epoch=ep.shape[0])
    return "ok"


def run(dataset_key, out_dir, limit=None, resume=False, shard=0, nshards=1,
        progress=None):
    ds = get_dataset(dataset_key)
    rows = ds.demographics()
    if limit:
        rows = rows[:limit]
    if nshards > 1:
        rows = [r for i, r in enumerate(rows) if i % nshards == shard]
    os.makedirs(out_dir, exist_ok=True)
    counts = {}
    for i, rec in enumerate(rows):
        bids = rec.get("BidsFolder", "")
        if resume and os.path.exists(os.path.join(out_dir, f"{bids}.npz")):
            counts["skip"] = counts.get("skip", 0) + 1
            continue
        try:
            r = _one(ds, rec, out_dir)
        except Exception as e:
            r = "err"
            if progress:
                progress(f"  ! {bids}: {type(e).__name__}: {e}")
        counts[r] = counts.get(r, 0) + 1
        if progress and (i + 1) % 10 == 0:
            progress(f"  [{shard}/{nshards}] {i+1}/{len(rows)} {counts}")
    if progress:
        progress(f"DONE [{shard}/{nshards}] {counts}")
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="standard", choices=list(REGISTRY.keys()))
    ap.add_argument("--out", required=True, help="output directory for per-recording npz")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    counts = run(args.dataset, args.out, limit=args.limit, resume=args.resume,
                 shard=args.shard, nshards=args.nshards,
                 progress=lambda m: print(m, file=sys.stderr, flush=True))
    print(f"counts={counts} out={args.out}")


if __name__ == "__main__":
    main()
