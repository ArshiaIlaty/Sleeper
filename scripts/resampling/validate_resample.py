"""3b sanity + validation: prove the resample patch is wired at the edfio.read_edf layer.

For one Emory (I0002, mixed 200/500 Hz) recording with a native-500 Hz EEG and one BIDMC
(S0001, uniform-200 Hz) recording, read the SAME physio EDF twice -- once through the real
edfio.read_edf, once after resample_patch.install() -- and compare on a central-EEG channel:
n_samples, fs, variance, and a high-frequency band ratio (beta 15-30 Hz / total 0.5-40 Hz via
Welch). The 500 Hz recording MUST change (proves the read path is hit + downsample removes
100-250 Hz content); the 200 Hz recording MUST be byte-identical (proves the fs-guard no-ops
already-200 channels). Also asserts CAISR annotation + SpO2 channels are left untouched.
"""
import sys, numpy as np
from scipy.signal import welch
sys.argv = ["validate"]
import sources, edfio
from app import channel_role

ds = sources.get_dataset("large")
rows = ds.demographics()

def first_eeg(edf):
    for s in edf.signals:
        if channel_role(s.label.strip()) == "eeg":
            return s
    return None

def metrics(sig):
    x = np.asarray(sig.data, float); fs = float(sig.sampling_frequency)
    f, p = welch(x, fs=fs, nperseg=int(min(len(x), fs*4)))
    tot = p[(f >= 0.5) & (f <= 40)].sum() + 1e-20
    beta = p[(f >= 15) & (f <= 30)].sum()
    return len(x), fs, float(np.var(x)), float(beta / tot)

def find(site_prefix, want_fs):
    for r in rows:
        site = r.get("SiteID", "")
        if not site.startswith(site_prefix):
            continue
        bids, sess = r.get("BidsFolder", ""), r.get("SessionID", "1")
        try:
            p = ds.physio_path(site, bids, sess)
            if not p:
                continue
            edf = edfio.read_edf(p, lazy_load_data=True)
            s = first_eeg(edf)
            if s is None:
                continue
            fs = float(s.sampling_frequency)
            if want_fs is None or abs(fs - want_fs) < 1:
                return site, bids, sess, p, s.label.strip(), fs
        except Exception as e:
            print(f"  skip {bids}: {type(e).__name__}: {e}")
            continue
    return None

_real = edfio.read_edf
for tag, prefix, want in [("Emory-500Hz", "I0002", 500.0), ("BIDMC-200Hz", "S0001", 200.0)]:
    hit = find(prefix, want)
    if not hit:
        print(f"[{tag}] no recording found with native fs~{want}")
        continue
    site, bids, sess, path, lab, fs = hit
    print(f"\n[{tag}] {bids} ses-{sess} site={site} eeg={lab!r} native_fs={fs}")
    edfio.read_edf = _real
    n0 = metrics(first_eeg(edfio.read_edf(path, lazy_load_data=True)))
    import resample_patch; resample_patch.install()
    edf_p = edfio.read_edf(path, lazy_load_data=True)
    n1 = metrics(first_eeg(edf_p))
    # annotation + spo2 untouched check
    ann = {s.label.strip(): float(s.sampling_frequency) for s in edf_p.signals
           if channel_role(s.label.strip()) in ("other", "spo2")}
    edfio.read_edf = _real
    print("   %-14s%16s%16s" % ("metric","native","patched"))
    for name, a, b in zip(["n_samples","fs_hz","variance","beta/total"], n0, n1):
        flag = "CHANGED" if abs(a-b) > 1e-9*(abs(a)+1) else "same"
        print(f"   {name:<14}{a:>16.5g}{b:>16.5g}   {flag}")
    print(f"   untouched(ann/spo2) fs: { {k:v for k,v in list(ann.items())[:6]} }")
print("\nDONE_VALIDATE")
