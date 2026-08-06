# Biosignal Viewer — PhysioNet Challenge 2026

An interactive, single-page web app to browse the dataset one patient at a time:

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
- **PSG signals** — pick any channels; the server downsamples each to ~2500 points before sending, so the 170 MB EDFs never hit the browser
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
| `app.py` | HTTP server + JSON API (`/api/patients`, `/api/demographics`, `/api/caisr`, `/api/dynamics`, `/api/signals`, `/api/glossary`, `/static/*`) |
| `glossary.py` | Plain-language definitions of every metric, stage, channel role, field, and dynamics stat + clinical reference ranges (drives the hover tooltips) |
| `dynamics.py` | Per-patient stage-transition matrix + fragmentation stats, with an embedded cohort baseline (mirrors `scripts/eda/stats_transitions.py`) |
| `preprocess.py` | CAISR hypnogram preprocessing — minimum-bout-duration smoothing that removes implausible single-epoch stage spikes (raw + cleaned staging) |
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

- `PHYSIONET_DATA_ROOT` — dataset root (default `/data-temp/shared-physionet26-dataset/extracted`)
- `PORT` / `HOST` — bind address (default `127.0.0.1:8050`)
- `POINTS` in `app.py` — max points/channel sent to the browser (default 2500)
- `CAISR_MIN_BOUT_EPOCHS` — minimum plausible stage-bout length for preprocessing
  (default `2` = 1 min; raise to smooth more aggressively, e.g. `3` = 90 s)

## Notes

- Signal decimation keeps a min/max envelope per bucket, so spikes/artifacts
  survive downsampling rather than being averaged away.
- The channel-list call is header-only; sample data is read only for the
  channels you actually select.
- Hypnogram "Unknown" (code 9) epochs render as gaps, not a spurious low stage.
