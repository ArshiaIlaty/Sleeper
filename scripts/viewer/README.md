# Biosignal Viewer — PhysioNet Challenge 2026

An interactive, single-page web app to browse the dataset one patient at a time:

- **Two cohorts, switchable in the header** — a **Standard (1,103)** release read
  from local disk and a **Large (6,530)** release streamed from S3. Pick either in
  the dataset dropdown; everything below re-loads for the chosen cohort. Small
  files (demographics, CAISR annotations) stream from S3 via the `aws` CLI; the
  large physiological EDFs are downloaded to a size-capped local cache on first
  view (a "streamed from S3" note appears on the signals panel for that cohort).
- **Demographics** panel (age, sex, race, BMI, follow-up fields, CI label)
- **CAISR sleep staging** — hypnogram, stage-% bar, and event indices (AHI, arousal, PLMI, apnea/hypopnea/RERA)
- **Raw vs. preprocessed staging** — the staging panel has **Compare / Raw /
  Preprocessed** tabs. Preprocessing removes biologically implausible rapid
  transitions (single-epoch stage "spikes") via **minimum-bout-duration
  smoothing** (`preprocess.py`): any interior bout shorter than the threshold
  (default 1 min) is merged into its longer neighbour; sleep-onset/offset wake is
  preserved and event indices are untouched. The Compare view overlays the raw
  hypnogram (grey) under the cleaned one (blue) and a banner reports how many
  epochs and transitions changed.
- **Sleep dynamics report** — a per-patient stage-transition matrix P(stage→next
  stage) rendered as a blue heatmap (hover any cell for this-patient vs. cohort
  comparison), **plus a transition-count heatmap with the diagonal zeroed** so
  actual stage changes carry the colour scale, plus fragmentation/"spike"
  statistics (awakenings, brief wake intrusions, stage-shift index, single-epoch
  spikes, bout counts/durations, REM periods) and named transition rates. Every
  number is shown against the cohort mean and flagged **red** when the patient is
  worse than average. The whole report has **Preprocessed / Raw tabs** (defaults
  to Preprocessed): the raw view is CAISR as scored, the preprocessed view
  recomputes every matrix and statistic on the spike-smoothed hypnogram, so you
  can see fragmentation collapse once implausible single-epoch transitions are
  merged away. Computed on demand in `dynamics.py` (same algorithm as the cohort
  EDA `stats_transitions.py`).
- **Signal by sleep stage** — pick any channel and it is split into its Wake / N1
  / N2 / N3 / REM epochs (using the *preprocessed* staging). Per stage you get
  amplitude statistics (mean ± SD, 5–95% range, minutes/epochs/percent), a
  **representative 30-second example epoch** taken from the middle of that stage's
  longest continuous bout so morphology is comparable across stages, and — for
  EEG channels — **mean relative spectral band power** (delta / theta / alpha /
  sigma / beta) per stage, the meaningful per-stage EEG "average" (delta dominates
  N3, sigma/spindles rise in N2). Below that, a **Full stage signal** viewer joins
  *every* epoch of a chosen stage end-to-end into one continuous trace (all ~90 min
  of Wake, all of N2, …) with the **same drag/scroll-to-zoom + amplitude hover** as
  the PSG plots — the server re-slices the concatenated window at full resolution,
  and dashed gold **seam lines** mark where consecutive epochs were not adjacent in
  the real night (hover reports both the concatenated-timeline position and the true
  night time). Computed on demand in `stage_signals.py` (pure numpy, no SciPy).
- **PSG signals** — pick any channels; the server downsamples each to ~2500 points before sending, so the 170 MB EDFs never hit the browser. **Scroll to zoom in/out at the cursor**, or **drag left-right to select a window** — the server re-samples just that window, so a short enough window returns *every raw sample* (e.g. a 0.1 s window on a 200 Hz EKG = 20 individual samples), making spikes and beat-to-beat morphology fully visible. **+ Zoom in** / **– Zoom out** step 2×, double-click or **Reset** returns to the whole night (down to a 0.1 s floor). The hover crosshair snaps to the nearest sample and shows its **time and amplitude value** (with units). Each plot is fully framed with min/mid/max y-ticks.
- **Per-stage physiology (NeuroKit2)** — a load-on-demand card computing HRV
  (ECG), EEG complexity, and respiratory rate *within each sleep stage* plus the
  cross-stage contrasts that capture blunted autonomic modulation. Served by
  `/api/nk_features` (`nk_features.py`).
- **Clinical report** — a load-on-demand summary card (`/api/report`,
  `clinical_report.py`) organising the team's prioritised cognitive-impairment
  markers into **Tier 1** (REM EEG slowing, N3 slow-wave activity, N2 spindle
  density, hypoxic burden, sleep-fragmentation index), **Tier 2** (stage-specific
  HRV modulation, NREM parasympathetic tone, respiratory instability, REM
  eye-movement density), a **clinical-feature table** (spindle density, SWA, REM
  slowing, hypoxic burden, stage HRV, AHI, arousal index, PLMI), **sleep-quality
  tiles** (efficiency, WASO, sleep/REM/N3 latency, awakenings, TST — abnormal
  values flagged red), and a **respiratory-event panel** (counts, event
  durations, post-event SpO₂ overshoot & recovery time). Assembled from
  `eeg_spectral.py` / `oxygenation.py` / `resp_events.py` / `nk_features.py`.
- **Hover explanations** — every event index, sleep stage, channel, and
  demographic field shows a plain-language tooltip on hover (or keyboard focus).
  Definitions live in `glossary.py` (one source of truth), served at
  `/api/glossary`.
- **Normal ranges + abnormal flagging** — event indices and stage percentages
  are compared against adult AASM/clinical reference ranges (also in
  `glossary.py`). Out-of-range values turn **red** with a severity flag
  (e.g. AHI 30+ = SEVERE), the normal range is shown under each tile, and the
  tooltip explains what the abnormal value means.
- **Edwards branding** — the logo (`edwards_logo.png`, served from `/static/`)
  sits in the header.

Pure Python **standard library** (`http.server`) + `numpy` + `edfio` — **no Flask,
no CDN, no external network**, so it cannot touch the company firewall. The
frontend is one dependency-free `index.html` (vanilla JS + inline SVG).

## Files

| File | Purpose |
|---|---|
| `app.py` | HTTP server + JSON API (`/api/datasets`, `/api/patients`, `/api/demographics`, `/api/caisr`, `/api/dynamics`, `/api/signals`, `/api/stage_signals`, `/api/nk_features`, `/api/report`, `/api/glossary`, `/static/*`); all data endpoints take `?ds=standard|large`; `/api/signals` also takes `t0`/`t1` (seconds) to zoom a window at full resolution; `/api/dynamics` returns both `raw` and `clean` (preprocessed) dynamics; `/api/stage_signals` takes `ch` to profile one channel by stage, or `ch`+`stage`(+`t0`/`t1`) to return that whole stage concatenated and zoomable; `/api/report` assembles the full clinical report |
| `stage_signals.py` | Chunk one PSG channel into its per-stage 30 s epochs → per-stage amplitude stats, a representative example epoch, EEG relative band power, and the whole-stage concatenated (zoomable) trace (pure numpy) |
| `sources.py` | Dataset abstraction: `standard` (local FS) and `large` (S3 via the `aws` CLI), with the size-capped LRU cache for large physio EDFs |
| `glossary.py` | Plain-language definitions of every metric, stage, channel role, field, and dynamics stat + clinical reference ranges (drives the hover tooltips) |
| `dynamics.py` | Per-patient stage-transition matrix + fragmentation stats, with an embedded cohort baseline (mirrors `scripts/eda/stats_transitions.py`) |
| `preprocess.py` | CAISR hypnogram preprocessing — minimum-bout-duration smoothing that removes implausible single-epoch stage spikes (raw + cleaned staging) |
| `export_features.py` | CLI: stream CAISR + demographics for a cohort → a wide per-recording feature CSV for model training (works on both datasets) |
| `nk_features.py` | Per-sleep-stage physiological features via **NeuroKit2**: heart-rate variability (ECG), EEG complexity, and respiratory rate/variability — computed *within each stage* plus cross-stage contrasts (pure numpy + neurokit2) |
| `export_nk_features.py` | CLI: stream the physiological EDFs → a wide per-recording **per-stage NeuroKit feature** CSV (companion to `export_features.py`; reads the big waveforms) |
| `export_report_features.py` | CLI: stream the physiological EDFs → a wide per-recording **clinical-report feature** CSV (EEG spectral/spindles, SpO₂/hypoxic burden, respiratory events, REM density). Decodes only one EEG/SpO₂/EOG channel + `resp_caisr`; no ECG pass, so it is cheaper than the NK export. Merge with the other two CSVs on `(bids_folder, session)` |
| `export_combined_features.py` | CLI: **single-pass driver** that opens each physio EDF once, decodes the union of channels both waveform exporters need, and writes BOTH the NK CSV and the report CSV together — halving S3 egress on the large cohort. `--shard`/`--nshards` stride-slice the cohort for parallel workers (each needs its own `PHYSIO_CACHE_DIR`). Outputs are schema-identical to the two standalone exporters |
| `eeg_spectral.py` | Per-stage EEG spectral features (scipy): absolute + relative band power, **Theta/Alpha**, **Delta/Sigma**, **REM-slowing** `(δ+θ)/(α+σ+β)`, and **sleep-spindle** detection (11–16 Hz envelope → density/amplitude/duration in N2 & N3) |
| `oxygenation.py` | SpO₂ features: scale-normalised (0–1 vs 0–100 auto-detect) mean/min/**T90**, **ODI**, desaturation depth stats, and **hypoxic burden** (Σ depth×duration per hour) |
| `resp_events.py` | Respiratory-event features from `resp_caisr` (1 Hz): apnea/hypopnea/RERA **counts**, AHI/RDI, **event durations**, and SpO₂-derived **post-event overshoot** + **recovery time** |
| `clinical_report.py` | Assembles the per-patient **clinical report** — Tier-1 / Tier-2 markers, a clinical-feature table, and sleep-quality metrics — from every signal domain (orchestrator; reads no files itself) |
| `HOW_TO_RUN.md` | Step-by-step run/connect guide for the whole team (host vs. viewer, tmux, troubleshooting) |
| `index.html` | Single-page UI (vanilla JS + SVG, validated Edwards palette, tooltip engine) |
| `edwards_logo.png` | Header logo, served from `/static/` (no CDN) |
| `run.sh` | Launch helper with a data-readability pre-check |

## Running (on pdmle)

> **Full team guide:** see **[`HOW_TO_RUN.md`](HOW_TO_RUN.md)** for a step-by-step
> host-vs-viewer walkthrough, `tmux` (keep it alive), and troubleshooting. Quick
> version below.

Run as an account **in the `mlusers` group** (e.g. `arshia_ilaty_physio26`) — it
reads the dataset directly; `ubuntu` cannot and must not use sudo for a server.

```bash
# on pdmle, as your data-capable account:
cd ~/viewer          # or wherever you copied scripts/viewer/
bash run.sh          # -> http://127.0.0.1:8050
```

From your laptop, tunnel the port over SSH and open a browser locally (the app
binds to 127.0.0.1, so it is only reachable through the tunnel — nothing is
exposed on the network):

```bash
ssh -L 8050:127.0.0.1:8050 arshia_ilaty_physio26@AWOR-PDMLEAPP01
# then, in your laptop browser:
open http://127.0.0.1:8050
```

## Configuration

- `PHYSIONET_DATA_ROOT` — standard (local) dataset root (default `/data-temp/shared-physionet26-dataset/extracted`)
- `PHYSIONET_S3_BUCKET` / `PHYSIONET_S3_PREFIX` — large dataset location
  (defaults `els-thv-nlp-sbox-input-834843060358` / `physionet26/large-dataset`)
- `PHYSIO_CACHE_DIR` — where large-dataset EDFs are cached (default `~/.cache/physio-viewer/large`)
- `PORT` / `HOST` — bind address (default `127.0.0.1:8050`)
- `POINTS` in `app.py` — max points/channel sent to the browser (default 2500)
- `CAISR_MIN_BOUT_EPOCHS` — minimum plausible stage-bout length for preprocessing
  (default `2` = 1 min; raise to smooth more aggressively, e.g. `3` = 90 s)

## Feature export (for model training)

`export_features.py` writes one wide row per recording — sleep macro-architecture
(stage %, efficiency, WASO, latencies, entropy), fragmentation / transition
dynamics, preprocessing deltas, event indices (AHI, arousal, PLMI + subtypes),
demographics, and the `Cognitive_Impairment` label — to a CSV your colleagues can
train on directly. Only the small CAISR annotation files are read (never the big
waveforms), so it streams the whole large cohort in ~1–2 h.

```bash
# on pdmle, as arshia_ilaty_physio26:
cd /data-temp/physio-viewer
python3 export_features.py --dataset standard --out exports/features_standard.csv
python3 export_features.py --dataset large    --out exports/features_large.csv --resume
```

`--resume` appends, skipping recordings already in the output (safe to re-run if
interrupted). This CSV covers sleep architecture only. Generated CSVs live in
`/data-temp/physio-viewer/exports/` on pdmle.

### Per-stage NeuroKit features (`export_nk_features.py`)

A **separate, heavier** export reads the physiological waveforms and computes, per
sleep stage (Wake / N1 / N2 / N3 / REM + pooled NREM + Sleep):

- **ECG heart-rate variability** — HR, SDNN, RMSSD, pNN50, SDSD, CVNN, LF, HF,
  LF/HF, and Poincaré SD1/SD2 *per stage*, plus **cross-stage contrasts** (REM/NREM
  RMSSD & LF/HF ratios, wake–sleep HR delta, the range of HR across stages). The
  hypothesis: cognitive impairment shows up as a *blunted autonomic swing across
  stages* more than in any single night-average number.
- **EEG complexity** — per-stage sample entropy + permutation entropy (reduced
  slow-wave-sleep complexity is a decline marker).
- **Respiratory rate & variability** per stage.

```bash
# on pdmle, as arshia_ilaty_physio26:
cd /data-temp/physio-viewer
python3 export_nk_features.py --dataset standard --out exports/nk_features_standard.csv
python3 export_nk_features.py --dataset large    --out exports/nk_features_large.csv --resume
```

~7–8 s/recording (whole-night R-peak detection + **linear** HRV; the O(n²)
`nk.hrv_nonlinear` / `fractal_higuchi` metrics are deliberately avoided — measured
unusable without `numba` on this box). Only one ECG, one (central) EEG, and one
respiratory channel are decoded per recording, never the full montage. Run under
`tmux`/`nohup` for the large cohort. Merge with `features_*.csv` on
`(bids_folder, session)` to train on the union. `team_code.py` still computes its
own whole-recording autonomic features from the raw waveforms; this is the
per-stage complement.

### Clinical-report signal features (`export_report_features.py`)

The **third** exporter takes the signal-derived markers that previously lived only
in the on-demand webapp report (`/api/report`) and writes them one row per
recording, so a model can actually train on them:

- **Per-stage EEG spectral** — absolute + relative band power (delta/theta/alpha/
  sigma/beta) per stage, **Theta/Alpha**, **Delta/Sigma**, **REM-slowing**
  `(δ+θ)/(α+σ+β)`, plus the flagship contrasts (REM slowing, N3 relative delta).
- **Sleep spindles** — N2/N3 density, amplitude, duration + the per-recording
  detection threshold.
- **Oxygenation** — scale-normalised (Emory 0–1 vs 0–100 auto-detect) mean/min/
  T90/ODI/desaturation depth and **hypoxic burden**.
- **Respiratory events** — per-type counts, AHI/RDI, event durations, and the
  SpO₂-derived post-event overshoot + recovery time.
- **REM density** — the EOG rapid-eye-movement index within REM.

```bash
# on pdmle, as arshia_ilaty_physio26:
cd /data-temp/physio-viewer
python3 export_report_features.py --dataset standard --out exports/report_features_standard.csv
python3 export_report_features.py --dataset large    --out exports/report_features_large.csv --resume
```

Only one central EEG, one SpO₂, and one EOG channel are decoded (plus the small
`resp_caisr` annotation stream) — and it **deliberately does not** recompute
ECG-HRV / EEG-complexity / respiratory rate, because those per-stage features
already live in `nk_features_*.csv`. Skipping the whole-night R-peak detection
makes it the cheaper waveform exporter (~2–5 s/recording). Merge all three CSVs
(`features_*`, `nk_features_*`, `report_features_*`) on `(bids_folder, session)`
to train on the full union.

### One pass for both waveform CSVs on the large cohort (`export_combined_features.py`)

Over the large (S3) cohort, running `export_nk_features.py` and
`export_report_features.py` separately downloads every 170–460 MB EDF **twice**.
This driver opens each EDF once, decodes the union of channels both need, and
writes both CSVs. It is compute-bound (~25 s/recording), so it **shards** across
CPU cores — each worker takes a stride of the cohort, writes its own CSV shards,
and **must use its own `PHYSIO_CACHE_DIR`** (the on-disk cache's eviction lock is
per-process):

```bash
# on pdmle, as arshia_ilaty_physio26 — launch 3 sharded workers, then merge:
cd /data-temp/physio-viewer
NSHARDS=3 DATASET=large bash large_combined_launch.sh   # -> exports/shards/*.csv
# ...wait for all shards to finish (tail /tmp/combined_large_s*.log)...
python3 merge_shards.py                                 # -> exports/{nk,report}_features_large.csv
```

The merged CSVs are schema-identical to the standalone exporters, so they stack
with the standard-cohort CSVs.

## Notes

- Signal decimation keeps a min/max envelope per bucket, so spikes/artifacts
  survive downsampling rather than being averaged away.
- The channel-list call is header-only; sample data is read only for the
  channels you actually select.
- Hypnogram "Unknown" (code 9) epochs render as gaps, not a spurious low stage.
