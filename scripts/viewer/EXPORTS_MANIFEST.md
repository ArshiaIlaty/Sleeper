# Feature exports — manifest

Every per-recording / per-epoch feature CSV the viewer's exporters produce, what
each column family means, how they join, and which script writes each one. On
`pdmle` these live under `/data-temp/physio-viewer/exports/` (world-readable;
teammates pull them over their own SSH). Two cohorts: `standard` (1,103, local
disk) and `large` (6,530, streamed from S3).

## Directory layout

```
exports/
├── EXPORTS_MANIFEST.md              ← this file
│
├── features_<cohort>.csv            WIDE · CAISR sleep architecture + demographics
├── nk_features_<cohort>.csv         WIDE · per-stage NeuroKit HRV / EEG-complexity / resp   (stage MEAN)
├── report_features_<cohort>.csv     WIDE · EEG spectral / spindles / SpO₂ / resp-events      (stage MEAN)
├── dispersion_features_<cohort>.csv WIDE · within-stage SPREAD of the above (SD/CV/pXX)      (NON-AVG)
│
├── feature_significance_standard.csv   ANALYSIS · univariate AUROC + MWU p + BH-FDR per feature
├── feature_significance_standard.md    ANALYSIS · presentable summary of the above
│
├── per_epoch/                       LONG · fully un-aggregated (many rows per recording)
│   ├── epoch_eeg_<cohort>.csv           one row per (recording × stage × epoch)
│   └── spindles_<cohort>.csv            one row per detected sleep spindle
│
└── shards/                          intermediate sharded outputs for the large-cohort runs
```

## The two shapes

| Shape | Grain | Rows / recording | Use |
|---|---|---|---|
| **WIDE** | one row per recording | 1 | the per-recording tabular model + significance test |
| **LONG** | one row per epoch (or per spindle) | hundreds | EDA, distribution plots, sequence / temporal models |

`features`, `nk_features`, `report_features` collapse each sleep stage to a **mean**.
`dispersion_features` is the **non-avg** wide companion — the *spread* across a
stage's epochs (SD, CV, and p10/p50/p90 + IQR for the flagship ratios). `per_epoch/`
is the **long** companion — the raw per-epoch values the means and spreads were
computed from.

## Join keys

All tables carry `(dataset, bids_folder, session)`.

- WIDE ↔ WIDE: join on `(bids_folder, session)` → one combined feature row per
  recording (this is what `build_local_feature_cache.py` and `feature_significance.py`
  do).
- LONG ↔ WIDE / demographics: join on `(bids_folder, session)`; the long table's
  extra `stage` / `epoch_index` columns give the within-night position.

## Column families

### `features_<cohort>.csv` — CAISR sleep architecture (`export_features.py`)
Stage %, sleep efficiency, WASO, latencies, stage entropy, fragmentation /
transition dynamics, preprocessing deltas, event indices (AHI, arousal, PLMI +
subtypes), demographics, and the `Cognitive_Impairment` label. Reads only the small
CAISR annotation files (no waveforms).

### `nk_features_<cohort>.csv` — per-stage NeuroKit, MEAN (`export_nk_features.py`)
Per stage (wake/n1/n2/n3/rem + pooled nrem/sleep): ECG-HRV (HR, SDNN, RMSSD, pNN50,
SDSD, CVNN, LF, HF, LF/HF, SD1/SD2), EEG sample/permutation entropy (mean + one SD),
respiratory rate (mean, SD, CV), plus cross-stage contrasts (REM/NREM ratios,
wake–sleep HR delta, HR range). Decodes one ECG + one central EEG + one effort
channel.

### `report_features_<cohort>.csv` — clinical-report signal markers, MEAN (`export_report_features.py`)
Per-stage EEG spectral (abs/rel band power, Theta/Alpha, Delta/Sigma, REM-slowing),
sleep spindles (density, amp_mean, dur_mean), SpO₂ (mean/min/T90/ODI/hypoxic
burden), respiratory events (counts, AHI/RDI, durations, post-event overshoot &
recovery), and an EOG REM-density proxy. Decodes one central EEG + one SpO₂ + one
EOG channel + the CAISR resp stream.

### `dispersion_features_<cohort>.csv` — within-stage SPREAD, NON-AVG (`export_dispersion_features.py`)
The *non-avg* companion. For every quantity the mean CSVs collapsed to one number,
this reports the spread across the stage's epochs:
- `eeg_<stage>_rel_<band>_{sd,cv}` — relative band-power spread.
- `eeg_<stage>_{theta_alpha,delta_sigma,rem_slowing}_{sd,cv,p10,p50,p90,iqr}`.
- `eegc_<stage>_{sampen,permen}_{sd,cv,p10,p50,p90,iqr}` — entropy spread.
- `rsp_<stage>_rate_{sd,cv,p10,p50,p90,iqr}` — respiratory-rate spread.
- `spindle_<stage>_{amp,dur}_{sd,cv}` — spindle amplitude / duration spread.

Reuses the exact per-epoch primitives of `eeg_spectral` / `nk_features`, so a mean
recomputed here equals the mean in the wide CSVs (verified to 5e-5). Spectral /
spindle dispersion uses the same ≤180-epoch even subsample per stage as the mean
CSVs, so the two are directly comparable; the long `per_epoch/` tables keep **every**
epoch.

### `per_epoch/epoch_eeg_<cohort>.csv` — LONG per-epoch EEG (`export_epoch_features.py`)
One row per (recording, stage, epoch): `stage`, `epoch_index`, `start_s`, then
`abs_<band>` / `rel_<band>`, `theta_alpha`, `delta_sigma`, `rem_slowing`, and
`sampen` / `permen` with `has_complexity` (entropy is a capped ≤25-epoch/stage
subsample — the O(n²) cost bound — so it is present only on those rows; band power
is on every epoch). Emitting all epochs is the point: the wide dispersion is a
summary of exactly these rows.

### `per_epoch/spindles_<cohort>.csv` — LONG spindle events (`export_epoch_features.py`)
One row per detected sleep spindle: `stage`, `start_s`, `duration_s`, `amp_uv`, and
the per-recording `threshold_uv`. This is the raw form the `report_features`
`spindle_*_amp_mean` / `dur_mean` and the `dispersion_features` spindle spreads were
computed from; per-recording spindle counts match the aggregated `n_spindles`.

## Regenerating

```bash
# on pdmle, as arshia_ilaty_physio26, from /data-temp/physio-viewer
# WIDE — CAISR only (fast, no waveforms)
python3 export_features.py            --dataset standard --out exports/features_standard.csv
# WIDE — waveform means
python3 export_nk_features.py         --dataset standard --out exports/nk_features_standard.csv
python3 export_report_features.py     --dataset standard --out exports/report_features_standard.csv
# WIDE — non-avg dispersion
python3 export_dispersion_features.py --dataset standard --out exports/dispersion_features_standard.csv
# LONG — per-epoch tables
python3 export_epoch_features.py      --dataset standard --outdir exports/per_epoch
# ANALYSIS — univariate significance across all wide families
python3 feature_significance.py       --exports exports --out exports/feature_significance_standard

# large cohort: add --resume; heavy waveform exports shard with --shard/--nshards
# (each shard needs its own PHYSIO_CACHE_DIR) and merge with merge_shards.py.
```

Add `--no-complexity` to `export_dispersion_features.py` / `export_epoch_features.py`
to skip the O(n²) per-epoch entropy pass (the dominant cost).
