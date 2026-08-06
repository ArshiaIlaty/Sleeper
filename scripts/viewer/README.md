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
  worse than average. Computed on demand in `dynamics.py` (same algorithm as the
  cohort EDA `stats_transitions.py`).
- **PSG signals** — pick any channels; the server downsamples each to ~2500 points before sending, so the 170 MB EDFs never hit the browser. **Scroll to zoom in/out at the cursor**, or **drag left-right to select a window** — the server re-samples just that window, so a short enough window returns *every raw sample* (e.g. a 0.1 s window on a 200 Hz EKG = 20 individual samples), making spikes and beat-to-beat morphology fully visible. **+ Zoom in** / **– Zoom out** step 2×, double-click or **Reset** returns to the whole night (down to a 0.1 s floor). The hover crosshair snaps to the nearest sample and shows its **time and amplitude value** (with units). Each plot is fully framed with min/mid/max y-ticks.
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
| `app.py` | HTTP server + JSON API (`/api/datasets`, `/api/patients`, `/api/demographics`, `/api/caisr`, `/api/dynamics`, `/api/signals`, `/api/glossary`, `/static/*`); all data endpoints take `?ds=standard|large`; `/api/signals` also takes `t0`/`t1` (seconds) to zoom a window at full resolution |
| `sources.py` | Dataset abstraction: `standard` (local FS) and `large` (S3 via the `aws` CLI), with the size-capped LRU cache for large physio EDFs |
| `glossary.py` | Plain-language definitions of every metric, stage, channel role, field, and dynamics stat + clinical reference ranges (drives the hover tooltips) |
| `dynamics.py` | Per-patient stage-transition matrix + fragmentation stats, with an embedded cohort baseline (mirrors `scripts/eda/stats_transitions.py`) |
| `preprocess.py` | CAISR hypnogram preprocessing — minimum-bout-duration smoothing that removes implausible single-epoch stage spikes (raw + cleaned staging) |
| `export_features.py` | CLI: stream CAISR + demographics for a cohort → a wide per-recording feature CSV for model training (works on both datasets) |
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
interrupted). Signal-derived autonomic features (ECG-HRV, SpO₂) are intentionally
out of scope here — the modeling code (`team_code.py`) computes those from the raw
waveforms. Generated CSVs live in `/data-temp/physio-viewer/exports/` on pdmle.

## Notes

- Signal decimation keeps a min/max envelope per bucket, so spikes/artifacts
  survive downsampling rather than being averaged away.
- The channel-list call is header-only; sample data is read only for the
  channels you actually select.
- Hypnogram "Unknown" (code 9) epochs render as gaps, not a spurious low stage.
