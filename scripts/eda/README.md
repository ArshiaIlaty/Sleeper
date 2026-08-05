# Dataset EDA suite — PhysioNet Challenge 2026 (Team SDG)

Comprehensive descriptive statistics over the full Challenge dataset: demographics
& static data, biosignals (physiological EDF), and sleep architecture (CAISR
algorithmic annotations). Designed to run on the **pdmle** machine where the data
lives (`/data-temp/shared-physionet26-dataset/extracted`).

## Modules

| File | What it computes |
|---|---|
| `common.py` | Paths, site names, channel-role map, stage/event codes, filename parsing |
| `statutils.py` | numpy-only summary helpers (distributions, value counts, Wilson CIs) |
| `stats_demographics.py` | Cohort size, per-site counts, **label prevalence** (overall / by age band / sex / race), age & BMI distributions, missingness, follow-up fields, ICD codes |
| `stats_biosignals.py` | **Header-only** scan of physio EDFs: channel inventory, sampling rates, units, montage consistency, recording duration, coverage |
| `stats_sleep.py` | Full read of CAISR annotations: TST, sleep efficiency, WASO, stage %, latencies, AHI/arousal/PLMI, respiratory subtypes, CAISR-vs-expert agreement |
| `stats_transitions.py` | **Stage-transition matrices** P(X→Y) + fragmentation ("spikes"): awakenings, brief-wake intrusions, single-epoch stage spikes, stage-shift index, wake/sleep bouts, REM periods, named directed transition rates. Cohort-mean + CI/non-CI matrices |
| `dump_transitions.py` | Writes `eda/per_recording_dynamics.csv` (per-recording fragmentation) + `eda/transitions.json` (matrices). Runs under `sudo` (numpy+csv+edfio only, no pandas) |
| `stats_significance.py` | **Feature significance vs the CI label**: chi-square (categorical) + Welch t-test & Mann–Whitney (numeric), Cohen d / Cramér V effect sizes, Benjamini–Hochberg FDR. Writes `eda/feature_significance.csv` |
| `dump_per_recording.py` | Joins CAISR summaries + demographics + label → `eda/per_recording.csv` (drives figures + significance) |
| `make_figures.py` | Paper figures → `paper/figures/` (incl. fig7 transition heatmap, fig8 effect-size ranking) |
| `report.py` | Renders the results dict to Markdown + flat CSVs |
| `run_eda.py` | Orchestrator; writes `eda/dataset_stats.json`, CSVs, and `eda/DATASET_REPORT.md` |

## Full pipeline order (on pdmle)

```bash
sudo ... python3 run_eda.py                 # stats_* incl. transitions -> dataset_stats.json + report
sudo ... python3 dump_per_recording.py      # per_recording.csv
sudo ... python3 dump_transitions.py        # per_recording_dynamics.csv + transitions.json
# then locally (needs scipy): significance + figures + final report
python3 stats_significance.py               # feature_significance.csv
python3 make_figures.py                     # paper/figures/*.png,.pdf
```

The report's §4 (dynamics) and §5 (significance) are populated when
`transitions.json` and the per-recording CSVs are present; `run_eda.py` includes
the transition stats inline, while significance is computed at report time off
the CSVs.

## Design notes

- **Physio EDFs are never fully loaded.** `edfio.read_edf(..., lazy_load_data=True)`
  reads only the header (labels, sampling rate, units, record count → duration).
  The ~170 GB of raw samples is never materialised, so a full scan is fast.
- **Conventions mirror `team_code.py`**: stage codes `{1:N3,2:N2,3:N1,4:REM,5:Wake,9:Unknown}`,
  respiratory `{1:obstructive,2:central,4:hypopnea,5:RERA}`, limb `{1:isolated,2:periodic}`.
- Outputs land in `eda/` (gitignored), matching the repo's existing convention.

## Running (on pdmle)

The dataset is owned `root:mlusers` (mode 750); the SSH login user is not in
`mlusers`, so reads go through `sudo`. Scientific packages live in the user's
`~/.local`, so pass them to root via `PYTHONPATH`:

```bash
USP=$(python3 -c "import site;print(site.getusersitepackages())")
sudo PYTHONPATH="$USP" EDA_OUT_DIR="$HOME/eda_out" python3 run_eda.py
# quick smoke test:  add  --limit 5 --expert-sample 5
```

Requires `edfio` (`sudo python3 -m pip install edfio`), plus numpy/pandas/scipy/sklearn.

## Environment variables

- `PHYSIONET_DATA_ROOT` — dataset root (default the pdmle path above)
- `EDA_OUT_DIR` — output directory (default `./eda`)
