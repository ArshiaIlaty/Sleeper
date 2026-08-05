# PhysioNet Challenge 2026 Dataset - Tree & Samples

Reference document for the polysomnography (PSG) cohort used in the PhysioNet Challenge 2026 (cognitive-impairment prediction from sleep). The dataset pairs overnight PSG recordings with de-identified patient demographics and ICD diagnosis codes, plus two layers of sleep annotations: automated **CAISR** scoring and **expert** human scoring. Three contributing sites are present. All identifiers are de-identified BDSP IDs and are shown verbatim.

- **Data root:** `/data-temp/shared-physionet26-dataset/extracted`
- **Total on-disk size:** 215G
- **Patients / recordings (physio EDFs):** 1103
- **Sites:** `S0001` = BIDMC (857 recordings), `I0002` = Emory (54 recordings), `I0006` = Kaiser (192 recordings)

| Top-level item | Size | Contents |
|---|---|---|
| `physiological_data/` | 214G | 1103 raw PSG EDFs (~170 MB each) |
| `algorithmic_annotations/` | 466M | 1090 CAISR auto-scoring EDFs |
| `human_annotations/` | 232M | 1097 expert-scoring EDFs |
| `demographics.csv` | 129K | 1103 patient rows |
| `ICD_codes_CI.csv` | 87K | ICD-10 codes for positive patients |

## 1. Directory tree

```text
extracted/   (215G total)
|-- demographics.csv                 (129K)
|-- ICD_codes_CI.csv                 (87K)
|-- physiological_data/   (214G)
|     |-- S0001/   [BIDMC]  857 files
|     |          sub-S0001111191757_ses-1.edf
|     |          sub-S0001111210276_ses-2.edf
|     |          sub-S0001111212184_ses-1.edf
|     |          ... (857 files total)
|     |-- I0002/   [Emory]  54 files
|     |          sub-I0002150000076_ses-1.edf
|     |          sub-I0002150000540_ses-2.edf
|     |          sub-I0002150000686_ses-1.edf
|     |          ... (54 files total)
|     `-- I0006/   [Kaiser]  192 files
|                sub-I0006179000025_ses-1.edf
|                sub-I0006179000097_ses-1.edf
|                sub-I0006179000408_ses-1.edf
|                ... (192 files total)
|
|-- algorithmic_annotations/   (466M)
|     |-- S0001/   [BIDMC]  847 files
|     |          sub-S0001111191757_ses-1_caisr_annotations.edf
|     |          sub-S0001111210276_ses-2_caisr_annotations.edf
|     |          sub-S0001111212184_ses-1_caisr_annotations.edf
|     |          ... (847 files total, CAISR)
|     |-- I0002/   [Emory]  53 files
|     |          sub-I0002150000076_ses-1_caisr_annotations.edf
|     |          sub-I0002150000540_ses-2_caisr_annotations.edf
|     |          sub-I0002150000686_ses-1_caisr_annotations.edf
|     |          ... (53 files total, CAISR)
|     `-- I0006/   [Kaiser]  190 files
|                sub-I0006179000025_ses-1_caisr_annotations.edf
|                sub-I0006179000097_ses-1_caisr_annotations.edf
|                sub-I0006179000408_ses-1_caisr_annotations.edf
|                ... (190 files total, CAISR)
|
|-- human_annotations/   (232M)
|     |-- S0001/   [BIDMC]  851 files
|     |          sub-S0001111191757_ses-1_expert_annotations.edf
|     |          sub-S0001111210276_ses-2_expert_annotations.edf
|     |          sub-S0001111212184_ses-1_expert_annotations.edf
|     |          ... (851 files total, expert)
|     |-- I0002/   [Emory]  54 files
|     |          sub-I0002150000076_ses-1_expert_annotations.edf
|     |          sub-I0002150000540_ses-2_expert_annotations.edf
|     |          sub-I0002150000686_ses-1_expert_annotations.edf
|     |          ... (54 files total, expert)
|     `-- I0006/   [Kaiser]  192 files
|                sub-I0006179000025_ses-1_expert_annotations.edf
|                sub-I0006179000097_ses-1_expert_annotations.edf
|                sub-I0006179000408_ses-1_expert_annotations.edf
|                ... (192 files total, expert)
```

## 2. File-naming convention

All EDFs follow a BIDS-like pattern:

```text
sub-<PID>_ses-<N>[_<annotation-suffix>].edf
```

- **`sub-<PID>`** - subject. `PID` embeds the site code as a prefix (`S0001`/`I0002`/`I0006`) followed by the de-identified BDSP patient id, e.g. `S0001111191757`.
- **`ses-<N>`** - session / recording night (`1`, `2`, ...). A patient may have multiple overnight sessions.
- **annotation suffix** - none for raw physio; `_caisr_annotations` for algorithmic; `_expert_annotations` for human scoring.

The stem `sub-<PID>_ses-<N>` is the join key linking a physio recording to its CAISR and expert annotation files.

| Modality | Directory | Example filename |
|---|---|---|
| Raw PSG | `physiological_data/S0001/` | `sub-S0001111191757_ses-1.edf` |
| CAISR | `algorithmic_annotations/S0001/` | `sub-S0001111191757_ses-1_caisr_annotations.edf` |
| Expert | `human_annotations/S0001/` | `sub-S0001111191757_ses-1_expert_annotations.edf` |
| Raw PSG | `physiological_data/I0002/` | `sub-I0002150000076_ses-1.edf` |
| Raw PSG | `physiological_data/I0006/` | `sub-I0006179000025_ses-1.edf` |

## 3. `demographics.csv`

**1103 rows x 14 columns.** Label column: `Cognitive_Impairment`.

| Column | Dtype | % missing | Values / distribution |
|---|---|---|---|
| `SiteID` | categorical (string) | 0.0% | S0001=857, I0006=192, I0002=54 |
| `BDSPPatientID` | int (identifier) | 0.0% | 1103 distinct ids |
| `CreationTime` | datetime | 0.0% | range 2007-08-11 -> 2020-10-19 |
| `BidsFolder` | categorical (string) | 0.0% | 1103 distinct; top: sub-S0001120173185=1, sub-S0001112722303=1, sub-S0001116945514=1, sub-S0001113953169=1, sub-S0001111625329=1, sub-S0001120942410=1 |
| `SessionID` | int (identifier) | 0.0% | 5 distinct ids |
| `Age` | int | 0.0% | min=50, median=61.0, max=88 |
| `Sex` | categorical (string) | 0.0% | Male=585, Female=518 |
| `Race` | categorical (string) | 0.0% | White=849, Black=146, Others=54, Unavailable=31, Asian=23 |
| `Ethnicity` | categorical (string) | 0.0% | Not Hispanic=976, Unavailable=66, Hispanic=61 |
| `BMI` | float | 75.9% | min=18.5, median=31.4, max=63.1 |
| `Time_to_Event` | float | 92.4% | min=390.0, median=1206.5, max=2175.0 |
| `Cognitive_Impairment` | categorical (bool) | 0.0% | False=1019, True=84 |
| `Last_Known_Visit_Date` | datetime | 0.0% | range 2015-06-20 -> 2062-09-03 |
| `Time_to_Last_Visit` | int | 0.0% | min=530, median=3541.0, max=18749 |

**Sample row** (real, de-identified):

```text
SiteID                : S0001
BDSPPatientID         : 112722303
CreationTime          : 2016-05-14 22:30:01
BidsFolder            : sub-S0001112722303
SessionID             : 1
Age                   : 73
Sex                   : Female
Race                  : White
Ethnicity             : Not Hispanic
BMI                   : 24.55
Time_to_Event         : (missing)
Cognitive_Impairment  : False
Last_Known_Visit_Date : 2024-01-03 00:00:00
Time_to_Last_Visit    : 2789
```

## 4. `ICD_codes_CI.csv`

**2526 code rows x 5 columns.** Columns: `BDSPPatientID`, `SiteID`, `ICDDate`, `ICD10`, `ICD9`

- **Join key:** `BDSPPatientID` (+ `SiteID`) links each code row back to a `demographics.csv` patient.
- **84 distinct patients** carry ICD codes here; 84 of them match a `demographics.csv` patient id.
- Demographics has **84 positive** (`Cognitive_Impairment=True`) patients; 84 of these appear in this file (codes justify the positive label; `ICD9` is largely empty).

**Sample rows:**

```text
BDSPPatientID,SiteID,ICDDate,ICD10,ICD9
121854978,S0001,2022-02-10,G30.1,
118139481,S0001,2021-03-05,G30.1,
117531674,S0001,2019-03-11,G30.0,
118139481,S0001,2020-01-10,G30.1,
117531674,S0001,2018-11-24,G30.0,
118606205,S0001,2018-07-13,G30.8,
121371360,S0001,2017-02-15,G30.9,
117709491,S0001,2020-10-24,G30.1,
```

**Top 15 ICD-10 codes by frequency** (of 1605 non-empty ICD-10 values):

| Rank | ICD-10 | Count |
|---|---|---|
| 1 | `G31.84` | 345 |
| 2 | `F03.90` | 304 |
| 3 | `F02.80` | 204 |
| 4 | `G30.1` | 135 |
| 5 | `G30.9` | 115 |
| 6 | `F01.50` | 105 |
| 7 | `F02.818` | 100 |
| 8 | `G31.83` | 90 |
| 9 | `F02.81` | 67 |
| 10 | `F03.91` | 38 |
| 11 | `G30.0` | 29 |
| 12 | `G30.8` | 26 |
| 13 | `F01.51` | 19 |
| 14 | `G31.09` | 14 |
| 15 | `G31.01` | 5 |

## 5. Physiological EDF (header-only sample)

Representative recording from the **dominant BIDMC montage** (315 of 857 BIDMC recordings share this exact 19-channel layout).

- **File:** `physiological_data/S0001/sub-S0001111191757_ses-1.edf`
- **Start:** 2017-03-27 21:23:50 (de-identified)
- **Duration:** 30109 s (~8.4 h)  |  data records: 30109 x 1.0s
- **Number of signals:** 19

| # | Label | fs (Hz) | Unit | Phys min | Phys max | N samples |
|---|---|---|---|---|---|---|
| 1 | `F3-M2` | 200 | uV | -3810 | 3809.6 | 6021800 |
| 2 | `F4-M1` | 200 | uV | -3804.4 | 3805.6 | 6021800 |
| 3 | `C3-M2` | 200 | uV | -3811.4 | 3812 | 6021800 |
| 4 | `C4-M1` | 200 | uV | -3792 | 3795.4 | 6021800 |
| 5 | `O1-M2` | 200 | uV | -3611.4 | 3805.8 | 6021800 |
| 6 | `O2-M1` | 200 | uV | -3086 | 3802.8 | 6021800 |
| 7 | `E1-M2` | 200 | uV | -3803.8 | 3789 | 6021800 |
| 8 | `E2-M1` | 200 | uV | -5229.8 | 5056.6 | 6021800 |
| 9 | `CHIN1-CHIN2` | 200 | uV | -3709.61 | 3794.6 | 6021800 |
| 10 | `LAT` | 200 | uV | -1987.59 | 1999.6 | 6021800 |
| 11 | `RAT` | 200 | uV | -1983.2 | 1999.6 | 6021800 |
| 12 | `CPRES` | 200 | uV | -0.2 | 0.2 | 6021800 |
| 13 | `CFLOW` | 200 | uV | 125.6 | 128.8 | 6021800 |
| 14 | `AIRFLOW` | 200 | uV | -1983 | 1999.6 | 6021800 |
| 15 | `PTAF` | 200 | uV | -124.801 | 124.2 | 6021800 |
| 16 | `CHEST` | 200 | uV | -1983.8 | 1999.6 | 6021800 |
| 17 | `ABD` | 200 | uV | -1983.8 | 2000 | 6021800 |
| 18 | `SaO2` | 200 | uV | 0 | 98 | 6021800 |
| 19 | `EKG` | 200 | uV | -1983.4 | 1999.8 | 6021800 |

> Physio signals were read **header-only** (`lazy_load_data=True`); raw samples (~170 MB) were never loaded.

## 6. CAISR annotation EDF (sampled values)

Code maps: `stage_caisr` {1:N3, 2:N2, 3:N1, 4:REM, 5:Wake, 9:Unknown}; `resp_caisr` {1:obstructive apnea, 2:central apnea, 4:hypopnea, 5:RERA}; `limb_caisr` {1:isolated, 2:periodic}; `arousal_caisr` {1:arousal}. Value `0` = none/background. `caisr_prob_*` channels hold per-epoch/sample class probabilities.

- **File:** `algorithmic_annotations/S0001/sub-S0001111191757_ses-1_caisr_annotations.edf`
- **Signals:** 11

| Signal | fs (Hz) | Length | Value distribution (value:count -> meaning) |
|---|---|---|---|
| `arousal_caisr` | 2 | 60180 | 0:56473 ((none/background)); 1:3707 (arousal) |
| `caisr_prob_no-ar` | 2 | 60180 | 0:60180 ((none/background)) |
| `caisr_prob_arous` | 2 | 60180 | 0:60180 ((none/background)) |
| `limb_caisr` | 1 | 30090 | 0:28827 ((none/background)); 1:312 (isolated LM); 2:951 (periodic LM) |
| `resp_caisr` | 1 | 30090 | 0:25867 ((none/background)); 1:573 (obstructive apnea); 2:270 (central apnea); 3:67 (code 3 (undocumented)); 4:3189 (hypopnea); 5:124 (RERA) |
| `stage_caisr` | 0.0333333 | 1003 | 1:181 (N3); 2:368 (N2); 3:89 (N1); 4:177 (REM); 5:182 (Wake); 9:6 (Unknown) |
| `caisr_prob_n3` | 0.0333333 | 1003 | 0:997 ((none/background)); 9:6 |
| `caisr_prob_n2` | 0.0333333 | 1003 | 0:997 ((none/background)); 9:6 |
| `caisr_prob_n1` | 0.0333333 | 1003 | 0:997 ((none/background)); 9:6 |
| `caisr_prob_r` | 0.0333333 | 1003 | 0:997 ((none/background)); 9:6 |
| `caisr_prob_w` | 0.0333333 | 1003 | 0:997 ((none/background)); 9:6 |

**First 20 epochs of `stage_caisr`** (30 s each, chronological):

```text
epoch:   1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
code:    9  9  9  5  5  5  5  5  5  5  5  5  5  5  5  3  5  3  3  3
stage:  Unknown -> Unknown -> Unknown -> Wake -> Wake -> Wake -> Wake -> Wake -> Wake -> Wake -> Wake -> Wake -> Wake -> Wake -> Wake -> N1 -> Wake -> N1 -> N1 -> N1
```

## 7. Expert annotation EDF

- **File:** `human_annotations/S0001/sub-S0001111191757_ses-1_expert_annotations.edf`
- **Signals (4):** `arousal_expert`, `limb_expert`, `resp_expert`, `stage_expert`

| Signal | fs (Hz) | Length | Value distribution (value:count) |
|---|---|---|---|
| `arousal_expert` | 2 | 60180 | 0:58195; 1:1985 |
| `limb_expert` | 1 | 30090 | 0:29225; 2:861; 3:2; 4:2 |
| `resp_expert` | 1 | 30090 | 0:27798; 1:177; 2:409; 7:847; 9:859 |
| `stage_expert` | 0.0333333 | 1003 | 0:17; 1:169; 2:429; 3:87; 4:168; 5:118; 9:15 |

**Differences vs CAISR:** the expert file carries only the four manual scoring channels (`stage/resp/limb/arousal_expert`) and **omits the `caisr_prob_*` probability channels**. The event integer codes are scorer-defined and do **not** necessarily match the CAISR code maps (e.g. `stage_expert`/`resp_expert` use additional codes such as 0 and 7), so decode them with the expert convention rather than the CAISR maps above.

## 8. Modality coverage summary

Coverage counted by recording key `sub-<PID>_ses-<N>`.

| Modality | Recordings | S0001 (BIDMC) | I0002 (Emory) | I0006 (Kaiser) |
|---|---|---|---|---|
| Raw PSG (physio) | 1103 | 857 | 54 | 192 |
| CAISR (algorithmic) | 1090 | 847 | 53 | 190 |
| Expert (human) | 1097 | 851 | 54 | 192 |

- **Recordings with all three modalities:** 1085
- **Physio with CAISR:** 1090  |  **Physio with expert:** 1097
- **Physio missing CAISR:** 13  |  **Physio missing expert:** 6

---
*Generated on pdmle by `dataset_tree.py` directly from the filesystem and EDF headers; all counts and values are measured, not estimated.*
