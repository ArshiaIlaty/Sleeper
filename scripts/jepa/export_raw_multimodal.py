"""Extract raw MULTIMODAL PSG epoch tensors for the multimodal JEPA.

Escalation from the EEG-only raw JEPA (raw_eeg_jepa_v2.py): that representation was
genuine (jepa-only AUROC 0.63) but REDUNDANT with the handcrafted EEG features. The
only remaining orthogonal-signal lever is the non-EEG PSG the champion under-exploits.
This extractor decodes a canonical 15-channel full-PSG montage present (under different
names) across all three sites:

    6 EEG   F3,F4,C3,C4,O1,O2   (CAR over present EEG, as in the EEG extractor)
    1 ECG
    2 EOG   left (E1/LOC), right (E2/ROC)
    1 chin EMG
    1 airflow           (nasal pressure / thermistor / PTAF)
    2 effort            chest, abdomen
    1 SpO2
    1 limb EMG          (leg / LAT / RAT)

Per recording it: canonicalizes by role+cue (channel_role from app.py gates the role,
the cue disambiguates the slot), resamples every present channel to 64 Hz, CAR-refs the
EEG block, robustly per-channel z-scores ALL channels (this also fixes the 854 SpO2
channels mislabeled 'uV'), and slices into stage-aligned 30 s epochs. Saves one npz:

    epochs   (n_epoch, 15, 1920)  float16   canonical order, z-scored (EEG also CAR'd)
    stages   (n_epoch,)           int8      CAISR codes (5=W 3=N1 2=N2 1=N3 4=REM)
    chan_mask(15,)                int8      1 if that canonical slot was present
    meta scalars: bids, site, label, fs, n_epoch

Resumable (--resume) and shardable (--shard/--nshards) like the EEG extractor.
Nothing leaves the box.
"""
import os
import sys
import argparse

import numpy as np
import edfio
from scipy import signal as sp_signal

from sources import get_dataset, REGISTRY
from app import channel_role, _caisr_stage_codes

TARGET_FS = 64.0
EPOCH_SEC = 30.0
L = int(round(TARGET_FS * EPOCH_SEC))          # 1920 samples/epoch

# canonical slots: (name, role-from-channel_role, ordered cue preferences).
# role gates (must match channel_role), cue disambiguates the slot within a role.
MM_SLOTS = [
    ("f3", "eeg", ["f3"]),
    ("f4", "eeg", ["f4"]),
    ("c3", "eeg", ["c3"]),
    ("c4", "eeg", ["c4"]),
    ("o1", "eeg", ["o1"]),
    ("o2", "eeg", ["o2"]),
    ("ecg", "ecg", ["ekg", "ecg"]),
    ("eog_l", "eog", ["e1", "loc"]),
    ("eog_r", "eog", ["e2", "roc"]),
    ("chin", "chin_emg", ["chin"]),
    ("flow", "airflow", ["flow", "ptaf", "nasal", "therm", "npt", "cpres",
                          "press", "cpap"]),
    ("chest", "effort", ["chest", "thor", "thorac"]),
    ("abd", "effort", ["abd", "abdomen"]),
    ("spo2", "spo2", ["spo2", "sao2"]),
    ("leg", "limb_emg", ["leg", "lat", "rat", "lleg", "rleg", "plm"]),
]
N_CH = len(MM_SLOTS)                            # 15
SLOT_NAMES = [s[0] for s in MM_SLOTS]
EEG_SLOTS = [i for i, s in enumerate(MM_SLOTS) if s[1] == "eeg"]


def _decode_all(edf):
    """Return list of (label, role, data, fs) for every EDF signal, deduped by label."""
    out, seen = [], set()
    for s in edf.signals:
        lab = s.label.strip()
        if lab in seen:
            continue
        seen.add(lab)
        out.append((lab, channel_role(lab), np.asarray(s.data, float),
                    float(s.sampling_frequency)))
    return out


def _canonicalize(decoded):
    """Assign decoded signals to the 15 canonical slots by role+cue (first match,
    each source channel used once). Return (sigs, fss, mask)."""
    sigs = [None] * N_CH
    fss = [None] * N_CH
    used = [False] * len(decoded)
    for slot, (name, role, cues) in enumerate(MM_SLOTS):
        for j, (lab, drole, data, fs) in enumerate(decoded):
            if used[j] or drole != role:
                continue
            low = lab.lower()
            if any(c in low for c in cues):
                sigs[slot] = data
                fss[slot] = fs
                used[j] = True
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
    mm = m if np.isfinite(m) else 0.0
    z = (np.nan_to_num(x, nan=mm) - mm) / s
    return np.clip(z, -8.0, 8.0)


def _build_epochs(sigs, fss, mask, codes):
    """Resample present channels -> CAR over present EEG -> per-channel z-score ->
    (n_epoch, 15, L) float16 aligned to stage codes."""
    if mask[EEG_SLOTS].sum() < 5:                 # require >=5 of 6 EEG (as V1/V2)
        return None, None
    res = [None] * N_CH
    for slot in range(N_CH):
        if sigs[slot] is not None:
            res[slot] = _resample(np.asarray(sigs[slot], float), fss[slot])
    present = [r for r in res if r is not None]
    n = min(len(r) for r in present)
    stacked = np.zeros((N_CH, n), dtype=np.float64)
    for slot in range(N_CH):
        if res[slot] is not None:
            stacked[slot] = res[slot][:n]
    # CAR over present EEG channels only (non-EEG are not referential montages)
    eeg_present = [i for i in EEG_SLOTS if mask[i] == 1]
    if eeg_present:
        stacked[eeg_present] -= stacked[eeg_present].mean(axis=0, keepdims=True)
    # per-channel robust z-score for every present channel
    for slot in np.where(mask == 1)[0]:
        stacked[slot] = _zscore(stacked[slot])

    spe = L
    n_ep = min(len(codes), n // spe)
    if n_ep < 1:
        return None, None
    ep = np.empty((n_ep, N_CH, spe), dtype=np.float16)
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
    decoded = _decode_all(edf)
    sigs, fss, mask = _canonicalize(decoded)
    if mask[EEG_SLOTS].sum() < 5:
        return "fewchan"
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
    present_tot = np.zeros(N_CH, dtype=np.int64)
    n_ok = 0
    for i, rec in enumerate(rows):
        bids = rec.get("BidsFolder", "")
        if resume and os.path.exists(os.path.join(out_dir, f"{bids}.npz")):
            counts["skip"] = counts.get("skip", 0) + 1
            continue
        try:
            r = _one(ds, rec, out_dir)
            if r == "ok":
                n_ok += 1
                present_tot += np.load(os.path.join(out_dir, f"{bids}.npz"),
                                       allow_pickle=True)["chan_mask"]
        except Exception as e:
            r = "err"
            if progress:
                progress(f"  ! {bids}: {type(e).__name__}: {e}")
        counts[r] = counts.get(r, 0) + 1
        if progress and (i + 1) % 10 == 0:
            progress(f"  [{shard}/{nshards}] {i+1}/{len(rows)} {counts}")
    if progress:
        pres = {SLOT_NAMES[k]: int(present_tot[k]) for k in range(N_CH)}
        progress(f"DONE [{shard}/{nshards}] {counts} | ok={n_ok} presence={pres}")
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="standard", choices=list(REGISTRY.keys()))
    ap.add_argument("--out", required=True)
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
