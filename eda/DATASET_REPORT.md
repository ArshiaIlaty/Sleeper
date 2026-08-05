# PhysioNet Challenge 2026 — Dataset Statistics

_Data root: `/data-temp/shared-physionet26-dataset/extracted`_  
Sites: `S0001`=BIDMC, `I0002`=Emory, `I0006`=Kaiser

> **How to read this report.** This is a full statistical profile of the Challenge 2026 dataset. Each section begins with a short explanation (in a quote block like this one) of the terms and why they matter, followed by the measured numbers. Sections: **1** demographics & the prevalence/age structure the scoring metric targets; **2** the raw biosignals and montage variability; **3** sleep architecture from CAISR annotations; **4** data quality and cross-modality coverage. A companion `DATASET_TREE.md` shows the file layout with concrete samples of each data type.


## 1. Cohort & Demographics

> **How to read this section.** *Prevalence* is the fraction of patients who
> actually have the condition — here, cognitive impairment. At 7.6% it is the
> *rare* (positive) class, which is exactly why the Challenge scores with a
> prevalence-weighted Reward rather than plain accuracy: a model that predicts
> "nobody is impaired" would be 92% accurate yet clinically useless. The
> *95% confidence interval* (CI) is the range the true prevalence plausibly falls
> in given this sample size (computed with the Wilson method); wider intervals =
> fewer patients = less certainty (note Emory's wide band, n=54).

- **Rows:** 1103  |  **Unique patients:** 1103  |  **Sessions:** 5  |  max rows/patient: 1
- **Label (Cognitive_Impairment):** 84 positive / 1019 negative  (0 missing)
- **Prevalence:** 7.62%  (95% CI 6.19–9.33%)

### Per site

| Site | Name | Patients | Rows | Positives | Prevalence % (95% CI) |
|---|---|--:|--:|--:|---|
| I0002 | Emory | 54 | 54 | 8 | 14.81 (7.70–26.60) |
| I0006 | Kaiser | 192 | 192 | 20 | 10.42 (6.84–15.54) |
| S0001 | BIDMC | 857 | 857 | 56 | 6.53 (5.07–8.39) |

### Numeric fields

| Field | n | mean | std | median | p5–p95 | missing |
|---|--:|--:|--:|--:|:-:|--:|
| Age (yr) | 1103 | 62.0 | 8.5 | 61.0 | 50.0–77.0 | 0.0% |
| BMI | 266 | 32.8 | 8.0 | 31.4 | 23.5–48.7 | 75.9% |
| Time_to_Event (d) | 84 | 1183 | 506 | 1206 | 422–2019 | 92.4% |
| Time_to_Last_Visit (d) | 1103 | 3559 | 1121 | 3541 | 2241–5235 | 0.0% |

### Age distribution

```
      [0,40) |  0
     [40,50) |  0
     [50,60) | ████████████████████████████████████████ 494
     [60,70) | ███████████████████████████████ 384
     [70,80) | ███████████████ 189
     [80,90) | ██ 36
    [90,120) |  0
```

### BMI distribution

```
    [0,18.5) |  0
   [18.5,25) | ███████████████████████ 44
     [25,30) | ████████████████████████████████████████ 74
     [30,35) | ████████████████████████████████████ 67
     [35,40) | ████████████████ 30
     [40,80) | ███████████████████████████ 51
```

### Categorical breakdowns

**Sex:** Male (585), Female (518)

**Race:** White (849), Black (146), Others (54), Unavailable (31), Asian (23)

**Ethnicity:** Not Hispanic (976), Unavailable (66), Hispanic (61)

### Prevalence by stratum (confounder check)

> **Why the age breakdown matters.** Impairment risk climbs steeply with age, so
> age is a *confounder*: a model could look accurate simply by learning
> "older = higher risk" without reading any real physiology. The Challenge guards
> against this with an age-conditioned AUROC (it only compares patients of similar
> age), so we must show signal *within* an age band, not just across ages. A
> *confounder* is a variable linked to both the input and the outcome that can
> create a misleading association.

**By age band** — the age-conditioned metric targets exactly this gradient:

| Age band | n | positives | prevalence % (95% CI) |
|---|--:|--:|---|
| <50 | 0 | 0 | — (—–—) |
| 50-59 | 494 | 10 | 2.02 (1.10–3.69) |
| 60-69 | 384 | 28 | 7.29 (5.09–10.34) |
| 70-79 | 189 | 33 | 17.46 (12.71–23.51) |
| 80+ | 36 | 13 | 36.11 (22.48–52.42) |

**By sex:**

| Sex | n | prevalence % |
|---|--:|--:|
| Female | 518 | 6.37 |
| Male | 585 | 8.72 |

### Missingness (all columns)

> **Missingness** is the fraction of rows where a field is blank. It drives
> feature decisions: BMI is missing for ~76% of patients and Time_to_Event for
> ~92% (it is only recorded for positives), so neither can be a load-bearing
> feature. *Time_to_Event* = days from the recording to the impairment diagnosis;
> *Time_to_Last_Visit* = days of follow-up, used to confirm negatives really
> stayed negative.

| Column | missing | % |
|---|--:|--:|
| SiteID | 0 | 0.0 |
| BDSPPatientID | 0 | 0.0 |
| CreationTime | 0 | 0.0 |
| BidsFolder | 0 | 0.0 |
| SessionID | 0 | 0.0 |
| Age | 0 | 0.0 |
| Sex | 0 | 0.0 |
| Race | 0 | 0.0 |
| Ethnicity | 0 | 0.0 |
| BMI | 837 | 75.9 |
| Time_to_Event | 1019 | 92.4 |
| Cognitive_Impairment | 0 | 0.0 |
| Last_Known_Visit_Date | 0 | 0.0 |
| Time_to_Last_Visit | 0 | 0.0 |

### Follow-up field availability

- Time_to_Event present: 84  (positives with TTE: 84)
- Time_to_Last_Visit present: 1103  (negatives with TTLV: 1019)

### ICD diagnosis codes

> **ICD-10 codes** are the standardized diagnosis codes clinicians record; here
> they define/justify the positive label. Each starts with a letter+number
> category. The ones in this cohort are all dementia/cognitive-disorder codes —
> see the decoder below. A patient can carry several (median 12), reflecting how
> the diagnosis evolved across visits.

- Code rows: 2526  |  patients with codes: 84
- Label-positive patients with ≥1 code: 84 / 84
- Codes per patient: median 12, max 449

**Top ICD-10 codes (count):** G31.84 (345), F03.90 (304), F02.80 (204), G30.1 (135), G30.9 (115), F01.50 (105), F02.818 (100), G31.83 (90), F02.81 (67), F03.91 (38), G30.0 (29), G30.8 (26), F01.51 (19), G31.09 (14), G31.01 (5)

**What these codes mean:**

| ICD-10 | Meaning |
|---|---|
| `G31.84` | Mild cognitive impairment (MCI), so stated |
| `F03.90` | Unspecified dementia, without behavioral disturbance |
| `F02.80` | Dementia in other diseases classified elsewhere, without behavioral disturbance |
| `G30.1` | Alzheimer's disease with late onset |
| `G30.9` | Alzheimer's disease, unspecified |
| `F01.50` | Vascular dementia, without behavioral disturbance |
| `F02.818` | Dementia in other diseases classified elsewhere, with other behavioral disturbance |
| `G31.83` | Dementia with Lewy bodies |
| `F02.81` | Dementia in other diseases classified elsewhere, with behavioral disturbance |
| `F03.91` | Unspecified dementia, with behavioral disturbance |
| `G30.0` | Alzheimer's disease with early onset |
| `G30.8` | Other Alzheimer's disease |
| `F01.51` | Vascular dementia, with behavioral disturbance |
| `G31.09` | Other frontotemporal dementia / frontotemporal degeneration |
| `G31.01` | Pick's disease (a frontotemporal dementia) |

## 2. Biosignals (physiological EDF — header-only scan)

> **Biosignals** here are the raw overnight recordings, stored as EDF (European
> Data Format) files — one per patient, 16–88 channels each. This scan reads only
> the *headers* (channel list, sampling rate, units), never the ~170 GB of
> samples. *Channel role* groups differently-named channels by what they measure
> (EEG = brain, EOG = eyes, EMG = muscle, ECG = heart, airflow/effort/SpO₂ =
> breathing). *Sampling Hz* is samples recorded per second. *Montage* = the exact
> set of channels in a recording; when it varies within a site, features must be
> computed by role and tolerate missing channels.

- **Files scanned:** 1103  |  read errors: 0
- **Recording duration (h):** mean 7.49, median 7.59, range 0.01–11.42

### Channel inventory (pooled)

| Channel | Role | Files | % | Sampling Hz | Units |
|---|---|--:|--:|---|---|
| EKG | ecg | 1091 | 98.9 | 200.000, 250.000, 500.000 | uV |
| CHEST | effort | 904 | 82.0 | 200.000, 250.000, 500.000, 512.000 | uV |
| F3-M2 | eeg | 899 | 81.5 | 200.000, 250.000, 500.000, 512.000 | uV |
| SaO2 | spo2 | 899 | 81.5 | 25.000, 200.000 | %, uV |
| C3-M2 | eeg | 898 | 81.4 | 200.000, 250.000, 500.000, 512.000 | uV |
| O1-M2 | eeg | 898 | 81.4 | 200.000, 250.000, 500.000, 512.000 | uV |
| LAT | limb_emg | 897 | 81.3 | 200.000, 250.000, 500.000 | uV |
| F4-M1 | eeg | 896 | 81.2 | 200.000, 250.000, 500.000, 512.000 | uV |
| O2-M1 | eeg | 896 | 81.2 | 200.000, 250.000, 500.000, 512.000 | uV |
| C4-M1 | eeg | 895 | 81.1 | 200.000, 250.000, 500.000, 512.000 | uV |
| RAT | limb_emg | 892 | 80.9 | 200.000, 250.000, 500.000 | uV |
| ABD | effort | 852 | 77.2 | 200.000, 250.000, 512.000 | uV |
| E1-M2 | eog | 848 | 76.9 | 200.000, 250.000, 512.000 | uV |
| AIRFLOW | other | 842 | 76.3 | 200.000 | uV |
| PTAF | airflow | 828 | 75.1 | 25.000, 200.000, 250.000, 512.000 | mV, uV |
| CFLOW | other | 736 | 66.7 | 20.000, 200.000 | L/min, mL/min, uV |
| CHIN1-CHIN2 | chin_emg | 584 | 52.9 | 200.000, 250.000, 512.000 | uV |
| E2-M1 | eog | 514 | 46.6 | 200.000, 250.000, 512.000 | uV |
| CPRES | other | 423 | 38.3 | 200.000 | uV |
| E2-M2 | eog | 334 | 30.3 | 200.000 | uV |
| E1 | eog | 249 | 22.6 | 200.000, 500.000, 512.000 | uV |
| E2 | eog | 249 | 22.6 | 200.000, 500.000, 512.000 | uV |
| Chin1-Chin2 | chin_emg | 238 | 21.6 | 200.000 | uV |
| M1 | other | 204 | 18.5 | 200.000, 250.000, 512.000 | uV |
| M2 | other | 204 | 18.5 | 200.000, 250.000, 512.000 | uV |
| SpO2 | spo2 | 203 | 18.4 | 10.000, 250.000, 512.000 | %, uV |
| C3 | eeg | 195 | 17.7 | 200.000, 512.000 | uV |
| C4 | eeg | 195 | 17.7 | 200.000, 512.000 | uV |
| O2 | eeg | 195 | 17.7 | 200.000, 512.000 | uV |
| O1 | eeg | 195 | 17.7 | 200.000, 512.000 | uV |
| F3 | eeg | 195 | 17.7 | 200.000, 512.000 | uV |
| F4 | eeg | 195 | 17.7 | 200.000, 512.000 | uV |
| Abdomen | other | 194 | 17.6 | 50.000, 512.000 | V, u, uV |
| Flow_DR | airflow | 194 | 17.6 | 200.000, 512.000 | m, mL/s, uV |
| ChinA | other | 192 | 17.4 | 200.000 | uV |
| ChinR | other | 192 | 17.4 | 200.000 | uV |
| ChinL | other | 192 | 17.4 | 200.000 | uV |
| Nasal Pressure | airflow | 191 | 17.3 | 200.000 | mbar, u, ubar |
| Thorax | other | 191 | 17.3 | 50.000 | u, uV |
| Thermistor | other | 191 | 17.3 | 50.000, 100.000, 200.000 | m, mV |
| Right Leg | other | 190 | 17.2 | 200.000 | uV |
| Left Leg | other | 190 | 17.2 | 200.000 | uV |
| CPAP Pressure | airflow | 174 | 15.8 | 20.000 | cmH20, cmH2O, mcmH20, mcmH2O |
| CHIN | chin_emg | 54 | 4.9 | 200.000, 500.000 | uV |
| ABDOMINAL | effort | 54 | 4.9 | 200.000, 500.000 | uV |
| THERM | airflow | 54 | 4.9 | 200.000, 500.000 | uV |
| C PRESS | airflow | 54 | 4.9 | 25.000, 200.000 | cmH2O |
| NPT | other | 52 | 4.7 | 25.000, 200.000, 500.000 | V, mV, uV |
| C-FLOW | airflow | 50 | 4.5 | 25.000, 200.000 | V, mV |
| CHIN1-CHIN3 | chin_emg | 17 | 1.5 | 200.000 | uV |
| CHIN2 | chin_emg | 10 | 0.9 | 512.000 | uV |
| RLEG+ | limb_emg | 10 | 0.9 | 512.000 | uV |
| RLEG- | limb_emg | 10 | 0.9 | 512.000 | uV |
| LLEG+ | limb_emg | 10 | 0.9 | 512.000 | uV |
| LLEG- | limb_emg | 10 | 0.9 | 512.000 | uV |
| Pressure | other | 10 | 0.9 | 512.000 | uV |
| ECG-LA | ecg | 10 | 0.9 | 512.000 | uV |
| ECG-RA | ecg | 10 | 0.9 | 512.000 | uV |
| ECG-LL | ecg | 10 | 0.9 | 512.000 | uV |
| ECG-V1 | ecg | 10 | 0.9 | 512.000 | uV |
| ECG-V2 | ecg | 10 | 0.9 | 512.000 | uV |
| AirFlow | other | 9 | 0.8 | 250.000, 512.000 | uV |
| CFlow | other | 9 | 0.8 | 250.000, 512.000 | uV |
| Chin1-Chin3 | chin_emg | 8 | 0.7 | 200.000 | uV |
| F3-AVG | eeg | 7 | 0.6 | 200.000 | uV |
| F4-AVG | eeg | 7 | 0.6 | 200.000 | uV |
| C3-AVG | eeg | 7 | 0.6 | 200.000 | uV |
| C4-AVG | eeg | 7 | 0.6 | 200.000 | uV |
| O1-AVG | eeg | 7 | 0.6 | 200.000 | uV |
| O2-AVG | eeg | 7 | 0.6 | 200.000 | uV |
| Airflow2 | other | 7 | 0.6 | 512.000 | uV |
| CPAP Press | airflow | 6 | 0.5 | 20.000 | cmH2O, mcmH2O |
| Thermistor 2 | other | 5 | 0.5 | 50.000 | mV, uV |
| E1-AVG | eog | 5 | 0.5 | 200.000 | uV |
| E2-AVG | eog | 5 | 0.5 | 200.000 | uV |
| CPAP Pressure 1 | airflow | 4 | 0.4 | 20.000 | mcmH20 |
| Cpress | other | 4 | 0.4 | 20.000 | cmH2O, mcmH2O |
| CHIN1 | chin_emg | 3 | 0.3 | 512.000 | uV |
| Flow | airflow | 3 | 0.3 | 512.000 | uV |
| Chest | effort | 3 | 0.3 | 512.000 | uV |
| THORACIC | effort | 2 | 0.2 | 200.000 | uV |
| CPAP PRESSURE | airflow | 2 | 0.2 | 20.000 | mcmH2O |
| Chin2 | chin_emg | 2 | 0.2 | 250.000 | uV |
| R EMG | chin_emg | 1 | 0.1 | 200.000 | uV |
| L EMG | chin_emg | 1 | 0.1 | 200.000 | uV |

### Per-site montage


**I0002 (Emory)** — 54 files, 8 distinct montage(s); duration median 7.66 h, channels median 19
  - Dominant montage (44 files, 19 ch): `ABDOMINAL, C PRESS, C-FLOW, C3-M2, C4-M1, CHEST, CHIN, E1, E2, EKG, F3-M2, F4-M1, LAT, NPT, O1-M2, O2-M1, RAT, SaO2, THERM`
  - ⚠ 8 montage variants — channel set is not uniform within site.

**I0006 (Kaiser)** — 192 files, 9 distinct montage(s); duration median 6.98 h, channels median 24
  - Dominant montage (168 files, 24 ch): `Abdomen, C3, C4, CFLOW, CPAP Pressure, ChinA, ChinL, ChinR, E1, E2, EKG, F3, F4, Flow_DR, Left Leg, M1, M2, Nasal Pressure, O1, O2, Right Leg, SpO2, Thermistor, Thorax`
  - ⚠ 9 montage variants — channel set is not uniform within site.

**S0001 (BIDMC)** — 857 files, 42 distinct montage(s); duration median 7.73 h, channels median 18
  - Dominant montage (315 files, 19 ch): `ABD, AIRFLOW, C3-M2, C4-M1, CFLOW, CHEST, CHIN1-CHIN2, CPRES, E1-M2, E2-M1, EKG, F3-M2, F4-M1, LAT, O1-M2, O2-M1, PTAF, RAT, SaO2`
  - ⚠ 42 montage variants — channel set is not uniform within site.

## 3. Sleep Architecture (CAISR annotations)

> **Sleep architecture** comes from CAISR, an automated sleep-staging algorithm
> that labels every 30-second epoch. Because it is automated, we also report its
> agreement with a human expert (~76% of epochs) — treat these as strong but
> imperfect labels. Key terms:
>
> - **Sleep stages** — *Wake*; *N1* (lightest), *N2* (intermediate), *N3* (deep,
>   restorative slow-wave sleep); *REM* (dreaming sleep). Healthy adults spend
>   most of the night in N2, with ~13–23% N3+REM.
> - **Total sleep time (TST)** — minutes actually asleep. **Sleep efficiency** —
>   TST ÷ time in bed (>85% is good). **WASO** — minutes awake *after* first
>   falling asleep (fragmentation). **Latency** — minutes to fall asleep / to
>   reach REM or N3.
> - **AHI** (Apnea–Hypopnea Index) — apneas (breathing stops) + hypopneas
>   (shallow breaths) per hour of sleep; >30 = severe sleep apnea. **Arousal
>   index** — brief awakenings per hour. **PLMI** — periodic leg movements per
>   hour. These are per *hour of sleep*, not per hour of recording.
> - **Stage entropy / transitions per hour** — how varied and how unstable the
>   night's stage pattern is; high transitions = fragmented sleep.

- **Files read:** 1090/1090  |  stage-missing: 0  |  errors: 0
- **CAISR vs. expert epoch stage agreement:** mean 76.0% (n=442 recordings sampled)

### Pooled PSG summaries

| Metric | n | mean | std | median | p5–p95 | missing |
|---|--:|--:|--:|--:|:-:|--:|
| Duration (h) | 1090 | 7.51 | 1.01 | 7.59 | 6.29–8.73 | 0.0% |
| Total sleep time (min) | 1090 | 332 | 78 | 345 | 194–436 | 0.0% |
| Sleep efficiency (%) | 1090 | 73.6 | 14.7 | 76.8 | 45.6–91.2 | 0.0% |
| WASO (min) | 1087 | 85 | 56 | 72 | 19–192 | 0.3% |
| Sleep latency (min) | 1087 | 23.8 | 23.7 | 16.5 | 1.5–71.4 | 0.3% |
| REM latency (min) | 1061 | 135.6 | 92.9 | 111.0 | 2.5–317.0 | 2.7% |
| N3 latency (min) | 1027 | 78.2 | 78.8 | 47.0 | 15.0–253.2 | 5.8% |
| Stage transitions/h | 1090 | 13.2 | 4.3 | 12.7 | 7.5–20.8 | 0.0% |
| Stage entropy (norm) | 1090 | 0.791 | 0.103 | 0.808 | 0.612–0.913 | 0.0% |
| Unknown epochs (%) | 1090 | 0.69 | 0.41 | 0.66 | 0.57–0.79 | 0.0% |
| % Wake | 1090 | 25.9 | 14.6 | 22.6 | 8.2–53.9 | 0.0% |
| % N1 | 1090 | 7.6 | 4.4 | 6.7 | 2.4–16.2 | 0.0% |
| % N2 | 1090 | 45.8 | 11.7 | 46.6 | 24.9–63.6 | 0.0% |
| % N3 | 1090 | 9.2 | 6.1 | 9.1 | 0.0–18.8 | 0.0% |
| % REM | 1090 | 11.5 | 6.8 | 11.4 | 0.6–22.8 | 0.0% |
| AHI (/h sleep) | 1085 | 47.9 | 28.2 | 42.1 | 13.3–94.9 | 0.5% |
| Arousal index (/h sleep) | 1088 | 39.7 | 19.2 | 38.2 | 8.7–73.7 | 0.2% |
| PLMI periodic (/h sleep) | 1082 | 88.4 | 76.1 | 68.2 | 10.1–230.4 | 0.7% |
| Obstructive apnea (/h) | 1085 | 7.42 | 9.29 | 4.52 | 0.29–24.47 | 0.5% |
| Central apnea (/h) | 1085 | 3.38 | 6.93 | 1.17 | 0.00–13.93 | 0.5% |
| Hypopnea (/h) | 1085 | 37.08 | 21.91 | 33.37 | 10.83–73.11 | 0.5% |
| RERA (/h) | 1085 | 1.66 | 2.18 | 0.89 | 0.00–5.98 | 0.5% |
| Isolated limb (/h) | 1082 | 70.54 | 115.81 | 34.84 | 13.02–243.76 | 0.7% |
| Periodic limb (/h) | 1082 | 88.44 | 76.07 | 68.15 | 10.08–230.37 | 0.7% |

### Key metrics by site

| Site | duration hours | tst min | sleep efficiency pct | %N3 | %REM | ahi | arousal index | %unknown epochs |
|---|---|---|---|---|---|---|---|---|
| I0002 (Emory) | 7.7 | 340.0 | 70.6 | 8.8 | 10.7 | 36.1 | 38.7 | 0.7 |
| I0006 (Kaiser) | 7.0 | 312.5 | 74.9 | 6.0 | 8.5 | 40.1 | 30.7 | 0.7 |
| S0001 (BIDMC) | 7.7 | 353.5 | 77.4 | 9.6 | 11.8 | 43.4 | 40.1 | 0.6 |

## 4. Data Quality & Cross-Modality Coverage

> **Data quality & coverage.** Not every recording has every modality. The three
> label sources are the raw *physio* EDF, the *CAISR* automated annotations, and
> the *expert* annotations. Recordings missing CAISR fall back to
> demographics/global features only. *Short recordings* (well under a full night)
> give unreliable sleep statistics and are flagged for the training pipeline.
> *Unit inconsistencies* are cases where the same physical quantity is labelled
> with different units across files (e.g. SaO₂ tagged "uv" instead of "%") — a
> cleaning step, not a data error per se.

- **Short physio recordings:** 9 under 1 h, 9 between 1–4 h (candidates to drop or flag in training).

  Recordings under 1 h:

  | File | Duration (s) | Site |
  |---|--:|---|
  | sub-S0001119527021_ses-1.edf | 30.0 | S0001 |
  | sub-S0001116358144_ses-1.edf | 102.0 | S0001 |
  | sub-S0001119843094_ses-1.edf | 114.0 | S0001 |
  | sub-I0002150027361_ses-4.edf | 147.0 | I0002 |
  | sub-S0001120595221_ses-1.edf | 630.0 | S0001 |
  | sub-S0001122348072_ses-1.edf | 870.0 | S0001 |
  | sub-S0001116371150_ses-1.edf | 1830.0 | S0001 |
  | sub-I0002150029068_ses-2.edf | 2788.0 | I0002 |
  | sub-S0001119476468_ses-1.edf | 2940.0 | S0001 |

### Cross-modality coverage (pooled)

- Physio records: 1103  |  CAISR: 1090  |  expert: 1097
- **Physio without CAISR annotation:** 13  (these records lose all CAISR-derived features → global fallback only)
- Physio without expert annotation: 6
  - e.g. sub-I0002150027361|ses-4, sub-I0006179006610|ses-1, sub-I0006179009708|ses-1, sub-S0001111343090|ses-1, sub-S0001113559394|ses-1, sub-S0001113886057|ses-1, sub-S0001114882422|ses-1, sub-S0001116358144|ses-1, sub-S0001116364109|ses-1, sub-S0001116971802|ses-1

| Site | Physio | CAISR | Expert | Physio w/o CAISR | Physio w/o expert |
|---|--:|--:|--:|--:|--:|
| I0002 (Emory) | 54 | 53 | 54 | 1 | 0 |
| I0006 (Kaiser) | 192 | 190 | 192 | 2 | 0 |
| S0001 (BIDMC) | 857 | 847 | 851 | 10 | 6 |

### Demographics ↔ EDF linkage

- Demographics rows: 1103
- Demographics rows without a physio EDF: 0
- Physio EDFs without a demographics row: 0

### Physical-unit inconsistencies

- **spo2**: 854 channel-instances with unexpected units (e.g. `SaO2` labelled `uv`)
