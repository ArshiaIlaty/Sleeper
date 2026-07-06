#!/usr/bin/env python
"""Generate a tiny synthetic dataset in the Challenge-2026 layout to smoke-test
the pipeline (no real data needed). Injects a noisy signal->label relationship:
positives have more EEG slowing, weaker spindles, less N3, more arousals."""
import os, sys, shutil
import numpy as np
import pandas as pd
import edfio

rng = np.random.default_rng(0)

def sig(data, fs, label):
    data = np.asarray(data, dtype=np.float64)
    lo, hi = float(np.min(data)) - 1, float(np.max(data)) + 1
    return edfio.EdfSignal(data, sampling_frequency=fs, label=label,
                           physical_range=(lo, hi))

def make_eeg(dur, fs, pos):
    t = np.arange(int(dur*fs)) / fs
    # Slowing increases with label; spindle (sigma ~13Hz) decreases with label.
    delta = (1.4 if pos else 1.0) * np.sin(2*np.pi*1.5*t)
    theta = (1.3 if pos else 1.0) * np.sin(2*np.pi*6*t)
    alpha = (0.7 if pos else 1.0) * np.sin(2*np.pi*10*t)
    spindle = (0.5 if pos else 1.0) * np.sin(2*np.pi*13*t)
    noise = rng.normal(0, 0.5, t.size)
    return 20*(delta+theta+alpha+spindle) + 10*noise

def make_caisr(dur, pos):
    n = int(dur)  # 1 Hz traces
    # stages 1=N3..5=W; positives get less N3, more wake/arousal
    probs = [0.05,0.35,0.30,0.05,0.25] if pos else [0.18,0.30,0.28,0.04,0.20]
    stages = rng.choice([1,2,3,4,5], size=n, p=probs).astype(float)
    def events(rate):
        s = np.zeros(n); 
        k = rng.poisson(rate*dur/3600)
        for _ in range(int(k)):
            i = rng.integers(0, n-2); s[i:i+2] = 1
        return s
    resp = events(25 if pos else 12)
    arousal = events(30 if pos else 15)
    limb = events(10 if pos else 6)
    return stages, resp, arousal, limb

def build(root, n_per_site=100):
    if os.path.exists(root): shutil.rmtree(root)
    sites = ['S0001', 'I0002', 'I0006']
    rows = []
    for site in sites:
        os.makedirs(os.path.join(root,'physiological_data',site), exist_ok=True)
        os.makedirs(os.path.join(root,'algorithmic_annotations',site), exist_ok=True)
        for k in range(n_per_site):
            pid = f'sub-{site}{k:03d}'; sess='01'
            pos = int(rng.random() < 0.5)          # balanced training set
            age = int(rng.normal(68 + 6*pos, 8))   # positives skew older (realistic)
            dur = 600
            eeg_fs = 128
            chans = []
            for lab in ['F3','C3','O1','E1','Chin1']:
                chans.append(sig(make_eeg(dur, eeg_fs, pos), eeg_fs, lab))
            for lab in ['M2','M1','O2','E2','Chin2']:
                chans.append(sig(rng.normal(0,5,dur*eeg_fs), eeg_fs, lab))
            # ECG present at every site EXCEPT I0006 (simulate a missing modality)
            if site != 'I0006':
                ecg_fs = 200
                tt = np.arange(dur*ecg_fs)/ecg_fs
                hr = 1.0 + 0.15*pos
                ecg = np.zeros_like(tt)
                for beat in np.arange(0, dur, 1.0/hr):
                    idx = int(beat*ecg_fs)
                    if idx < ecg.size: ecg[idx] = 5.0
                chans.append(sig(ecg + rng.normal(0,0.1,tt.size), ecg_fs, 'ECG'))
            spo2 = np.clip(rng.normal(95 - 3*pos, 1.5, dur), 70, 100)
            chans.append(sig(spo2, 1, 'SpO2'))
            edfio.Edf(signals=chans).write(
                os.path.join(root,'physiological_data',site,f'{pid}_ses-{sess}.edf'))

            stages, resp, arousal, limb = make_caisr(dur, pos)
            cch = [sig(stages,1,'stage_caisr'), sig(resp,1,'resp_caisr'),
                   sig(arousal,1,'arousal_caisr'), sig(limb,1,'limb_caisr')]
            edfio.Edf(signals=cch).write(os.path.join(
                root,'algorithmic_annotations',site,
                f'{pid}_ses-{sess}_caisr_annotations.edf'))

            rows.append(dict(SiteID=site, BDSPPatientID=1000+len(rows),
                CreationTime='2015-01-01', BidsFolder=pid, SessionID=sess,
                Age=age, Sex=rng.choice(['Male','Female']),
                Race=rng.choice(['White','Black','Asian','Other']),
                Ethnicity=rng.choice(['Not Hispanic','Hispanic']),
                BMI=round(float(rng.normal(28,4)),1),
                Time_to_Event=(rng.integers(400,2000) if pos else ''),
                Cognitive_Impairment=bool(pos),
                Last_Known_Visit_Date='2021-01-01', Time_to_Last_Visit=2000))
    pd.DataFrame(rows).to_csv(os.path.join(root,'demographics.csv'), index=False)
    print(f'Built {len(rows)} patients across {len(sites)} sites at {root}')

if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else 'synthetic_data')
