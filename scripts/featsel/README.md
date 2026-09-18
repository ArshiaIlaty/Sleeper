# Feature-selection toolkit run + network-physiology scouting

Two tool-scouting tasks, reported together because both ask *"does an outside tool
surface anything our 436-feature champion misses?"*

1. **Feature-selection toolkit** — run a colleague's generic `FeatureSelection`
   battery (`/home/arshia/featureselection`) on our modeling matrix and see which
   features it surfaces / how it behaves at 436 features.
2. **Network physiology** — assess whether
   [`causal_networks_physiology`](https://github.com/moritz-g/causal_networks_physiology)
   (directed cross-system coupling) could help us.

Both are **descriptive / scouting** exercises. Neither changes the champion; any
feature they motivate still has to clear the existing LOSO fusion gate
(`jepa_gate_ab.py`-style paired bootstrap) to ship.

---

## 1. Feature-selection toolkit on our matrix

### What it is

The colleague `FeatureSelection` class runs a battery of feature-ranking methods
over a tabular `(features, binary target)` frame and aggregates them into a
consensus. It was demoed on a generic heart-failure CSV and had **never been run on
our data** (confirmed: no cross-reference in `Sleeper`, no PhysioNet artifacts in the
repo, git history all upstream). This run answers *how it behaves on our features and
which ones it prioritises* — **not** *does adding X improve LOSO reward* (that is the
gate's job).

### How we ran it

`run_featsel_physionet.py` (+ `featsel_finish.py`) load the cached **1103 × 436**
modeling matrix (`feature_matrix_local_plus.npz`), drop identifier/leakage columns,
median-impute NaNs so every sklearn selector runs on identical rows, attach the
binary `CI` target, and execute the battery with per-method guards. Ran on the pdmle
box (DUA: data never leaves; only the aggregate rankings below do). Target
prevalence: **84 positives (7.6 %)** — the same weak-signal regime as every other
screen.

**11 methods produced a ranking** (consensus): ANOVA-F (`F_Stats`), maximal
information coefficient (`MIC`, minepy), RF impurity (`RF_fimpo`), XGBoost split-count
& coverage (`SplitCount`, `Coverage`), RF permutation importance (`PermuImp`), Boruta
(`boruta`), SHAP (`ShapTree`), LightGBM gain & split (`importance_gain`,
`importance_split`), and mRMR (`MRMR`). LightGBM **null-importance** columns
(`NullGain`/`NullSplit`) are computed but **excluded from consensus** — they are a
shuffled-target baseline, not a relevance ranking.

### Four things had to be fixed (library bit-rot + intractability)

The toolkit is a couple of years old and three of its methods break against current
libraries; a fourth is intractable at our feature count. Fixes live in our driver /
`featsel_finish.py`; the colleague file is left untouched.

| Symptom                                                         | Cause                                                                                                 | Fix                                                                                        |
| --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| whole import dies (`ImportError: minepy`)                     | `from minepy import MINE` is module-level, so a missing optional dep kills **every** method   | `pip install --user minepy` (+ boruta / mrmr / xgboost / shap)                           |
| `lgb.train() got an unexpected keyword 'categorical_feature'` | LightGBM 4.x removed that kwarg from`lgb.train`                                                     | recompute gain/split with the current API (`booster.feature_importance("gain"/"split")`) |
| `ValueError: All arrays must be of the same length` (mRMR)    | `mrmr_classif` now returns **fewer** than `K` features; toolkit builds a length-`K` frame | map returned order → rank, unselected → NaN                                              |
| SHAP hangs                                                      | `KernelExplainer` with `max_evals=2·n+1` is intractable at 436 features                          | substitute a`TreeExplainer` on a 200-tree RF (class-1 abs-mean)                          |

### Consensus result — top 20 by mean rank (of 436)

Rank = 1-best per method, averaged across the 11 methods. `top10 / top20` = how many
of the 11 methods put the feature in their own top-10 / top-20.

| #  | feature                        | mean rank |        median | top10 | top20 | reads as                       |
| -- | ------------------------------ | --------: | ------------: | ----: | ----: | ------------------------------ |
| 1  | `arch__trans_p_wake_n1`      |      22.8 |          19.5 |     2 |     6 | sleep fragmentation (wake→N1) |
| 2  | `bmi`                        |      29.2 | **2.0** |     7 |     9 | demographic risk               |
| 3  | `rep__eeg_n1_theta_alpha`    |      33.4 |           5.0 |     7 |     9 | **EEG theta slowing**    |
| 4  | `age`                        |      37.4 | **2.0** |     9 |    10 | demographic risk               |
| 5  | `rep__eeg_n2_theta_alpha`    |      40.3 |          14.0 |     3 |     8 | **EEG theta slowing**    |
| 6  | `rep__remd_threshold_uv`     |      41.5 |          27.0 |     2 |     2 | REM EMG threshold              |
| 7  | `rep__eeg_n1_abs_theta`      |      43.2 |          38.5 |     0 |     3 | **EEG theta slowing**    |
| 8  | `arch__arousal_idx_n1`       |      43.4 |          13.0 |     3 |     7 | fragmentation (N1 arousals)    |
| 9  | `rep__eeg_wake_rel_theta`    |      51.4 |           9.0 |     6 |     7 | **EEG theta slowing**    |
| 10 | `rep__eeg_n2_rel_theta`      |      54.4 |          16.0 |     3 |     6 | **EEG theta slowing**    |
| 11 | `pct_r`                      |      55.8 |          27.0 |     0 |     4 | REM proportion                 |
| 12 | `micro__rswa_phasic_per_min` |      56.3 |          35.5 |     1 |     3 | REM-without-atonia             |
| 13 | `arch__arousal_idx_rem`      |      59.9 |          33.0 |     4 |     5 | fragmentation (REM arousals)   |
| 14 | `rep__eeg_n1_rel_alpha`      |      72.0 |          19.5 |     3 |     6 | EEG spectral (N1 alpha)        |
| 15 | `nk__hrv_sleep_n_beats`      |      72.2 |          58.5 |     1 |     2 | HRV coverage                   |
| 16 | `prob_arous_mean`            |      73.2 |          49.5 |     0 |     1 | arousal probability            |
| 17 | `micro__rswa_rem_rms_cv`     |      73.7 |          59.0 |     0 |     3 | REM EMG variability            |
| 18 | `arch__trans_p_n3_n2`        |      77.0 |          85.0 |     1 |     2 | N3→N2 transition              |
| 19 | `rep__eeg_n2_abs_theta`      |      80.4 |          29.0 |     1 |     4 | **EEG theta slowing**    |
| 20 | `nk__hrv_n3_sdnn`            |      81.8 |          64.0 |     0 |     1 | deep-sleep HRV                 |

**Redundancy-aware view (mRMR top-12):** `age`, `rep__spindle_n2_amp_mean`,
`arch__arousal_idx_n1`, `rep__eeg_n1_theta_alpha`, `nk__hrv_n1_sdsd`,
`arch__arousal_idx_rem`, `arch__trans_p_n3_n2`, `waso_min`, `rep__eeg_wake_rel_theta`,
`rep__eeg_rem_abs_delta`, `periodic_limb_idx`, `nk__hrv_n1_tp` — mRMR trades some raw
theta-power features for less-collinear complements (spindle amplitude, HRV, PLMI).

### What it tells us

- **The toolkit re-derives our own physiology.** Independently of our
  `stats_significance.py` Cohen-d ranking, the consensus surfaces the same four
  stories: **EEG theta slowing** (N1/N2/wake theta power + theta/alpha ratio — the
  canonical EEG-slowing marker of cognitive decline; it dominates the list),
  **sleep fragmentation** (wake→N1 transitions is the #1 feature; N1/REM arousal
  indices), **REM microstructure** (RSWA phasic/min, REM-EMG variability), and
  **age / bmi**. No brand-new predictor appears — the value is *validation* of the
  existing feature priorities plus a redundancy (mRMR) and stability (Boruta) lens we
  didn't have.
- **Boruta re-confirms the signal ceiling.** Of 436 features, Boruta confirms only
  **5** as reliably relevant across all 5 folds (**8** including tentative); the rest
  are indistinguishable from their shadow. That is the same **n_pos = 84 / 3-site**
  wall that produced 10 consecutive JEPA NO-SHIPs — the ceiling is the label count,
  not the feature engineering.
- **Read the two rank columns together.** `age`/`bmi` sit at median rank **2** (top by
  most methods) but mean rank ~30–37, because the tree-*split-count* methods
  (`SplitCount`, `importance_split`) under-rank low-cardinality / collinear
  demographics. Mean-rank consensus is a heuristic over heterogeneous scales, not a
  calibrated importance.

### Caveats

- **Whole-cohort, no site holdout.** This is descriptive ranking on all 1103 rows — it
  is optimistic and says nothing about LOSO generalisation. It prioritises candidates;
  the LOSO fusion gate decides lift.
- Outputs (`featsel_raw.csv`, `featsel_summary.csv`) are aggregate rankings only (feature
  names + rank numbers, no per-subject data) and are gitignored, matching the repo's
  `eda/` convention.

---

## 2. Network physiology — `causal_networks_physiology` assessment

### What it is

Moritz Günther's **Network Physiology** toolkit computes **directed cross-system
coupling** between three overnight signals — breathing rate, heart rate, and
EEG-alpha — resampled to 1 Hz, computed **per sleep stage** across several timescales,
and contrasted across **young-control / elderly-control / OSA** groups. Two estimators:

- **Granger G-causality** (`process_sleep_data.py`): statsmodels **VAR** with ADF
  stationarity screening (`checkStatCalcG`) → directed breath↔heart↔EEG influence.
- **MPRSA** (`functions.py`): multivariate phase-rectified signal averaging — a
  quasi-periodicity-aware coupling estimator robust to non-stationarity.

It is research code (no `requirements.txt`, hard-wired to its own breath/heart/EEG-alpha
data layout), not a pip-installable library.

### Could it help us?

**Plausibly, on the one axis our champion under-exploits.** Our champion is
EEG-spectral-dominated; the *directed* autonomic↔cortical↔respiratory coupling this
toolkit measures is largely absent from the 436 features (we have per-stage HRV and RSP
scalars, but no cross-system directed coupling). Crucially, this is a **different**
proposition from the failed multimodal JEPA (NO-SHIP #8, chance — the learned
multimodal embedding *diluted* EEG): these would be a **handful of low-dimensional,
interpretable coupling scalars**, not a high-dimensional learned representation, so they
attack a different failure mode.

### Caveats (why this is not a slam-dunk)

1. **Wrong target.** The toolkit validates coupling contrasts for **aging / OSA vs
   young**, not for **cognitive impairment**. The physiology is adjacent, but the
   published discriminative contrasts are not our label.
2. **Port, not plug-in.** No packaging; the estimators are wired to its own 1 Hz
   breath/heart/EEG-alpha arrays. Using it means re-implementing the G-causality + PRSA
   coupling estimators against our `CACHE`/EDF pipeline — a small feature-engineering
   project, not an import.
3. **Same wall.** The **n_pos = 84 / 3-site LOSO** ceiling that killed 10 JEPA screens
   applies unchanged. Low-dimensional, interpretable features are **not** exempt from
   the fusion gate — they must clear a Δ over the champion whose CI excludes zero.

### Recommendation

**Worth a small, strictly time-boxed experiment; do not vendor the repo.** Implement a
*compact* set of directed coupling features — e.g. breath↔HR and HR↔EEG-band
G-causality plus one PRSA coupling scalar, per NREM/REM — as ordinary tabular columns,
then run them straight through the existing LOSO fusion gate like any other candidate.
It is cheap, interpretable, and probes an under-exploited axis. Set expectations at the
n_pos wall: neutral is the likely outcome, and that is still a useful negative given how
interpretable the features are. If the fusion Δ is neutral, close it — same bar as every
other screen.

---

## Files

| File                                          | What it is                                                                                                                                  |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `run_featsel_physionet.py`                  | Load the 436-feature matrix → run the toolkit battery with per-method guards + TreeExplainer SHAP substitute → write raw + consensus CSVs |
| `featsel_finish.py`                         | Recompute the 3 bit-rotted methods (LGBM gain/split, mRMR) with current APIs and re-merge the consensus                                     |
| `featsel_raw.csv` / `featsel_summary.csv` | Per-method table + consensus ranking (gitignored; aggregate stats only)                                                                     |

The toolkit source lives at `/home/arshia/featureselection` (colleague repo, left
unmodified). The network-physiology repo is cloned at
`/home/arshia/causal_networks_physiology` (not part of this project).
