"""One-off real-data probe for batch-2 extractors (B3 coupling, E2 RSWA, A2 CAP).

Decodes ONE physio EDF's central EEG + chin EMG (+ the CAISR arousal stream) for a
few recordings and runs the three new extractors, so we can sanity-check the
real-data path (channel-role matching incl. the "chin"-substring EMG limitation,
CAP rate calibration vs the ~0.3-0.6 literature range) before a full extraction.

  PYTHONPATH=... python3 probe_batch2.py --dataset standard --limit 3
"""
import argparse
import sys

import numpy as np
import edfio

from sources import get_dataset, REGISTRY
from app import channel_role, _caisr_stage_codes
import nk_features as nkf
import eeg_coupling as ecpl
import emg_atonia as ema
import cap_events as cap

_EEG_PREFER = ["c3", "c4", "o1", "o2"]


def probe(ds, rec):
    bids = rec.get("BidsFolder", "")
    site, sess = rec.get("SiteID", ""), rec.get("SessionID", "1")
    codes, reason = _caisr_stage_codes(ds, rec, bids)
    if codes is None:
        return f"{bids}: no staging ({reason})"
    # arousal stream from the CAISR EDF
    ar, ar_fs = None, None
    edf_c = ds.open_caisr(site, bids, sess)
    if edf_c is not None:
        for s in edf_c.signals:
            if s.label.strip() == "arousal_caisr":
                ar = np.asarray(s.data, float); ar_fs = float(s.sampling_frequency)
    try:
        f = ds.physio_path(site, bids, sess)
    except Exception:
        f = None
    if not f:
        return f"{bids}: no physio file"
    edf = edfio.read_edf(f, lazy_load_data=True)
    labels = [s.label.strip() for s in edf.signals]
    roles = {l: channel_role(l) for l in labels}
    eeg_ch = nkf._pick_channel(labels, roles, {"eeg"}, prefer=_EEG_PREFER)
    emg_ch = nkf._pick_channel(labels, roles, {"chin_emg"})
    eeg = emg = None
    eeg_fs = emg_fs = None
    for s in edf.signals:
        lab = s.label.strip()
        if lab == eeg_ch:
            eeg = np.asarray(s.data, float); eeg_fs = float(s.sampling_frequency)
        elif lab == emg_ch:
            emg = np.asarray(s.data, float); emg_fs = float(s.sampling_frequency)
    out = [f"{bids}  eeg={eeg_ch}@{eeg_fs}  emg={emg_ch}@{emg_fs}  arousal_fs={ar_fs}"]
    if eeg is not None:
        c = ecpl.so_spindle_coupling(eeg, eeg_fs, codes)
        n2 = (c.get("stages") or {}).get("n2", {}) if c.get("ok") else {}
        out.append(f"   B3 couple: ok={c.get('ok')} n2_strength={n2.get('coupling_strength')} "
                   f"n2_nspin={n2.get('n_spindles')} err={c.get('error','')}")
        cp = cap.cap_features(eeg, eeg_fs, codes, arousal_codes=ar, arousal_fs=ar_fs)
        out.append(f"   A2 CAP: ok={cp.get('ok')} rate={cp.get('cap_rate')} a_index={cp.get('cap_a_index')} "
                   f"A1%={cp.get('cap_a1_pct')} A2%={cp.get('cap_a2_pct')} A3%={cp.get('cap_a3_pct')} "
                   f"nseq={cp.get('cap_n_sequences')} err={cp.get('error','')}")
    else:
        out.append("   no EEG channel")
    if emg is not None:
        r = ema.rswa_features(emg, emg_fs, codes)
        out.append(f"   E2 RSWA: ok={r.get('ok')} rem_nrem={r.get('rswa_rem_nrem_ratio')} "
                   f"tonic_frac={r.get('rswa_tonic_fraction')} phasic/min={r.get('rswa_phasic_per_min')} "
                   f"err={r.get('error','')}")
    else:
        out.append(f"   no chin-EMG channel (labels: {[l for l in labels if 'chin' in l.lower() or 'emg' in l.lower()]})")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="standard", choices=list(REGISTRY.keys()))
    ap.add_argument("--limit", type=int, default=3)
    args = ap.parse_args()
    ds = get_dataset(args.dataset)
    rows = ds.demographics()[:args.limit]
    for rec in rows:
        try:
            print(probe(ds, rec)); print()
        except Exception as e:
            print(f"{rec.get('BidsFolder','')}: ERROR {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()
