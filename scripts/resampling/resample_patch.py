"""Non-invasive resample-to-TARGET_HZ shim for the physio-viewer feature extractors.

Monkeypatches edfio.read_edf so every high-rate PHYSIOLOGICAL waveform (EEG/EOG/EMG/ECG/
airflow/effort with native fs>=FS_MIN) is polyphase-resampled (scipy.signal.resample_poly) to
TARGET_HZ and reports sampling_frequency==TARGET_HZ. CAISR annotation channels (stage/resp/
arousal/limb codes -> role "other") and low-rate SpO2 are LEFT UNTOUCHED (resampling integer
code streams would corrupt them). Extractors call edfio.read_edf(path, lazy_load_data=True) then
read sig.data / sig.sampling_frequency, so patching read_edf covers all five families at one
choke point without editing any production module. Env knobs: RESAMPLE_HZ (200), RESAMPLE_FS_MIN
(20), RESAMPLE_ROLES (csv), RESAMPLE_VERBOSE.
"""
import os, sys
import numpy as np
import edfio
from fractions import Fraction
from scipy.signal import resample_poly
from app import channel_role

TARGET_HZ = float(os.environ.get("RESAMPLE_HZ", "200"))
FS_MIN    = float(os.environ.get("RESAMPLE_FS_MIN", "20"))
ROLES     = set(os.environ.get("RESAMPLE_ROLES",
                 "eeg,eog,chin_emg,limb_emg,ecg,airflow,effort").split(","))
_real_read_edf = edfio.read_edf


class _RSig:
    """Proxy EdfSignal exposing resampled data + new fs; forwards everything else."""
    def __init__(self, sig, data, fs):
        object.__setattr__(self, "_sig", sig)
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_fs", float(fs))
    @property
    def label(self): return self._sig.label
    @property
    def data(self): return self._data
    @property
    def sampling_frequency(self): return self._fs
    def __getattr__(self, k): return getattr(self._sig, k)


class _REdf:
    """Proxy Edf overriding .signals; forwards everything else to the real Edf."""
    def __init__(self, edf, signals):
        object.__setattr__(self, "_edf", edf)
        object.__setattr__(self, "_signals", signals)
    @property
    def signals(self): return self._signals
    def __getattr__(self, k): return getattr(self._edf, k)


def _resample_one(x, fs):
    if fs <= 0 or abs(fs - TARGET_HZ) < 1e-6:
        return np.asarray(x, float), fs, False
    frac = Fraction(int(round(TARGET_HZ)), int(round(fs))).limit_denominator(10000)
    up, dn = frac.numerator, frac.denominator
    if up == dn:
        return np.asarray(x, float), fs, False
    y = resample_poly(np.asarray(x, float), up, dn)
    return y.astype(float), TARGET_HZ, True


def read_edf_resampled(*a, **k):
    edf = _real_read_edf(*a, **k)
    out, changed = [], []
    for sg in edf.signals:
        lab = sg.label.strip(); role = channel_role(lab); fs = float(sg.sampling_frequency)
        if role in ROLES and fs >= FS_MIN and abs(fs - TARGET_HZ) > 1e-6:
            data, nfs, ch = _resample_one(np.asarray(sg.data, float), fs)
            out.append(_RSig(sg, data, nfs))
            if ch: changed.append((lab, fs, nfs))
        else:
            out.append(sg)
    if os.environ.get("RESAMPLE_VERBOSE") and changed:
        print(f"[resample200] resampled {len(changed)} chan(s): {changed}", file=sys.stderr)
    return _REdf(edf, out)


def install():
    edfio.read_edf = read_edf_resampled
    return read_edf_resampled
