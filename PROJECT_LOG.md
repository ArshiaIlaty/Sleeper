# Project Log — Team SDG, PhysioNet Challenge 2026

A running record of work on the George B. Moody PhysioNet Challenge 2026 entry
(predict `Cognitive_Impairment` from overnight PSG + demographics). Newest
entries first. Dates are absolute.

Legend: ✅ done & verified · 🔬 verified against data · 📌 needs follow-up ·
⚠️ known limitation.

---

## 2026-08-06 (UI polish) — Signal zoom, plot framing, hypnogram-label fix

Three UI fixes reported after using the app.

### PSG signal zoom (drag-to-zoom, server re-samples the window)
`/api/signals` now accepts `t0`/`t1` (seconds); each channel is sliced to that
sample window BEFORE decimating, so a short window returns full/near-raw
resolution instead of the whole-night min/max envelope. Frontend: **drag
left-right on any plot to zoom**, double-click / Reset to whole night, Zoom out
2×, a window readout, and a hover crosshair showing the cursor time.
- 🔬 Verified on real data: full-night EKG = 12.3 s/point (spikes invisible);
  100 s window = 0.04 s/point (~300× finer); 10 s window = 2000 pts (every
  sample); 2 s window = 400 pts (fully raw). Works for the S3 (large) cohort too.
- 🐞 Caught + fixed a bug: `_decimate`'s small-array branch returned a numpy
  array (not JSON-serialisable) → 500 on windows ≤2500 samples. Now returns a
  plain float/None list.

### Plot framing ("shape/curve cut off at the edge")
Signal plots now draw a full rectangle frame with min/mid/max y-ticks and x-time
ticks (were a lone bottom gridline, so the trace looked unbounded at the right).
The hypnogram got an enclosing frame too.

### Hypnogram-label overlap (Compare tab)
The "Raw (as scored by CAISR)" / "Preprocessed (spikes removed)" labels overlaid
the Wake segment of the stage-% bar — caused by a negative bottom margin on
`.hyp-label`. Fixed the margins so labels sit cleanly above each bar.

---

## 2026-08-06 (large dataset) — Dual-cohort viewer + feature-CSV export

Got access to the **large dataset** (6,600 demographics rows / 6,530 CAISR
recordings, 1.36 TB of physio EDFs) in S3
(`s3://els-thv-nlp-sbox-input-834843060358/physionet26/large-dataset`). Added it
to the app alongside the standard cohort and built a feature-export tool.

### Dataset source abstraction — `scripts/viewer/sources.py`
One `Dataset` interface over two backends: **standard** = local FS; **large** =
S3 via the **`aws` CLI** (chosen over boto3, which is only installed for the
`ubuntu` account — the CLI works for every account through the instance role).
Small files (demographics, CAISR EDFs) stream into memory via `aws s3 cp <key> -`;
the big physio EDFs download to a **size-capped LRU cache** (`~/.cache/physio-viewer`,
8 GB default) on first view. 🔬 Verified `arshia_ilaty_physio26` can reach S3 via
the instance role and that `edfio` reads a CAISR EDF straight from a BytesIO S3
stream.

### Viewer — dataset selector + S3 signals
`app.py` refactored so every data endpoint takes `?ds=standard|large`; added
`/api/datasets`. `index.html` has a **Dataset dropdown** in the header that
reloads the cohort; the signals panel shows a "streamed from S3 (first load
downloads the EDF)" note for the large cohort.
- 🔬 Verified end-to-end on pdmle: standard=1103 / large=6600 patients; large
  CAISR+preprocess+dynamics computed from S3-streamed files; large signals
  downloaded+cached in ~4.8 s first hit, **0.009 s cached**; channel trace
  decimated from cache; standard cohort unchanged (regression clean).

### Feature-CSV export — `scripts/viewer/export_features.py`
Streams CAISR + demographics per recording → a **60-column** per-recording CSV
for colleagues to train on: sleep macro-architecture (stage %, efficiency, WASO,
latencies, entropy), fragmentation/transition dynamics, preprocessing deltas,
event indices (AHI/arousal/PLMI + subtypes), demographics, and the label.
Feature math mirrors `scripts/eda/stats_sleep.py` + `dynamics.py` +
`preprocess.py` so the CSV matches the report/viewer. Only small CAISR files are
read (never the 1.36 TB of waveforms); autonomic signal features stay in
`team_code.py`. Resumable (`--resume`), pure stdlib+numpy+edfio (no pandas).
- 🔬 **Standard CSV generated: 1090 rows × 60 cols** (13 no-CAISR skipped),
  0 errors → `/data-temp/physio-viewer/exports/features_standard.csv`.
- 📌 **Large CSV generating** (detached `setsid nohup` run on pdmle, ~1–2 h,
  0 errors at launch) → `exports/features_large.csv`.
- ⚠️ `exports/` lives on pdmle (data-derived, not committed to git).

### Review + docs
- A code review of the source/export diff informed the design (aws-CLI over
  boto3; cache eviction; ds-race on signal cache key → keyed by dataset).
- README + HOW_TO_RUN document dataset selection, S3 env vars, and the export
  command; `sources.py`/`export_features.py` added to the files table.

---

## 2026-08-06 — CAISR hypnogram preprocessing, raw-vs-clean tab, count heatmap, run guide

Four viewer additions requested by the team.

### Hypnogram preprocessing — `scripts/viewer/preprocess.py`
Removes biologically implausible rapid stage transitions via
**minimum-bout-duration smoothing** (iterative shortest-bout merge): any
*interior* stage bout shorter than `min_bout_epochs` (default 2 = 1 min) is
relabelled to its longer neighbour (ties → preceding). Only interior bouts are
merged, so legitimate wake at sleep onset/offset is preserved; Unknown(9) and
event indices (AHI etc.) are untouched.
- **Why this over alternatives:** stage codes are categorical not ordinal, so a
  numeric median filter invents nonsensical stages; an HMM on CAISR's *hard*
  labels (no emission probabilities) collapses to a transition-smoothing prior —
  i.e. this, but opaque. The rule-based smoother states its one assumption (a
  minimum plausible bout length) explicitly and is fully reproducible.
- 🔬 Verified on 3 real patients across sites: removes ~half of all stage
  transitions (e.g. 108→56, 144→55), all single-epoch spikes; N1 (the flickery
  light stage) consistently shrinks, N2/N3/REM consolidate. `/api/caisr` now
  returns raw + cleaned hypnograms, both stage-%, and a change summary.

### Viewer — raw vs. preprocessed staging tab (`index.html`)
The CAISR card's staging panel now has **Compare / Raw / Preprocessed** tabs
(defaults to Compare). Compare stacks both hypnograms — raw drawn as a faint grey
ghost under the blue cleaned trace — with per-view stage-% bars and a banner
reporting epochs changed + transitions removed.

### Viewer — transition-count heatmap (diagonal zeroed)
Added a second heatmap in the dynamics card showing the **raw count** of each
stage change, with the **diagonal set to zero** (self-transitions removed) so the
off-diagonal moves carry the colour scale — as requested. Refactored the matrix
rendering into a reusable `matrixHeatmap(order, cellSpec)` helper. Both heatmaps
use the validated single-hue blue sequential ramp (dataviz validator: PASS).

### Run guide — `scripts/viewer/HOW_TO_RUN.md`
Step-by-step host-vs-viewer walkthrough (start server, SSH-tunnel, tmux to keep
it alive, one-liner, troubleshooting table, IT-safe rationale) since the app is
only online when someone is hosting it. README links to it + documents
`preprocess.py` and `CAISR_MIN_BOUT_EPOCHS`.

- ⚠️ Verified structurally + against live JSON on pdmle; not browser-rendered
  (no headless browser). Redeployed at `/data-temp/physio-viewer/` (perms fixed).

---

## 2026-08-05 (viewer per-patient report) — Sleep-dynamics report in the viewer

Added a **per-patient sleep-dynamics report** to the biosignal viewer so the
transition/fragmentation stats we computed cohort-wide are now available live
for whichever patient is loaded.

### New `scripts/viewer/dynamics.py` + `/api/dynamics`
Self-contained port of the cohort transition algorithm (`stats_transitions.py`)
for a single patient's CAISR stage channel. Computes the 5×5 stage-transition
probability matrix P(X→Y), fragmentation ("spikes") — awakenings, brief wake
intrusions, stage-shift index, single-epoch stage spikes, wake/sleep bout
counts+durations, REM periods — and named directed transition rates. Each number
is paired with the **cohort baseline** (pooled means over 1090 recordings,
embedded as `COHORT_FRAG`/`COHORT_MATRIX` from `eda/transitions.json`).
- 🔬 Verified on pdmle against a real patient (`sub-I0002150000076`, TST 6.98 h):
  matrix rows sum to 100%, every fragmentation + named-transition field
  populated with its cohort comparison; endpoint + glossary served correctly.

### Viewer UI — `index.html` `dynamicsCard`
New card after the CAISR panel: the transition matrix as a single-hue **blue
sequential heatmap** (darker = more likely, exact % in each cell, hover shows
this-patient vs. cohort + Δpp), then fragmentation and transition-rate tiles.
Each tile shows the cohort mean and a ▲/▼ arrow; tiles where the patient is
**worse than the cohort** turn red (color paired with a glyph, per dataviz
rules). Definitions added to `glossary.py` (`DYNAMICS_GLOSSARY`, 18 entries) so
every stat and the matrix header have hover explanations.
- ⚠️ Verified structurally + against live JSON on pdmle; not yet browser-rendered
  (no headless browser). Redeployed bundle at `/data-temp/physio-viewer/`;
  restart under `arshia_ilaty_physio26` to pick it up.

---

## 2026-08-05 (latest) — Stage dynamics, feature significance, abnormal-value flagging

Added detailed sleep-dynamics statistics, a formal significance analysis, and
clinical reference ranges in the viewer.

### Stage-transition + fragmentation stats — `scripts/eda/stats_transitions.py`
Per recording: the full 5×5 stage-transition probability matrix P(X→Y), plus
fragmentation ("spikes") — awakenings, brief single-epoch wake intrusions,
single-epoch stage spikes, stage-shift index, wake/sleep bout counts+durations,
REM-period count, and named directed transition rates. Run on pdmle via
`dump_transitions.py` → `eda/per_recording_dynamics.csv` (1090 rows, 0 errors)
+ `eda/transitions.json` (cohort-mean + CI/non-CI matrices).
- 🔬 **Key finding:** impaired patients have a **destabilized N3** (N3→N3 −5.4pp,
  N3→N2 +6.0pp) and **less stable REM** (REM→REM −4.8pp) — a mechanistic view of
  their fragmentation. Cohort matrix diagonal is high (N2→N2 92%, REM→REM 90%).

### Feature significance — `scripts/eda/stats_significance.py`
Per user request ("c or s statistics"): **chi-square** for categorical features,
**Welch's t-test** (+ Mann–Whitney robustness) for numeric, each vs the CI label,
with **Cohen d / Cramér V** effect sizes and **Benjamini–Hochberg FDR**. Runs
locally off the per-recording CSVs (scipy). Output `eda/feature_significance.csv`.
- 🔬 **14 of 48 features significant at q<0.05.** Ranked by effect: age
  (d=1.08, the confounder) ≫ periodic-limb-movement index, ↓REM, ↑WASO/wake,
  ↓efficiency/entropy, fewer REM periods, ↓N3, REM→Wake rate. **AHI is NOT
  significant** (d=0.19) — discriminative signal is in architecture/continuity,
  not respiratory load. (`time_to_last_visit` is significant but reflects outcome
  timing — treat with caution, not a clean predictor.)

### Viewer — normal ranges + red abnormal flagging (task per user request)
`glossary.py` now carries adult **AASM/clinical reference ranges** for every
event index and stage %. The viewer shows the normal range under each tile,
turns out-of-range values **red** with a severity flag (e.g. AHI 30+ = SEVERE,
N3 <10% = LOW), and the tooltip explains what the abnormal value means. Verified
on pdmle against real patients (e.g. one with AHI 28 moderate, N3 1.4% low,
Wake 26% high — all flagged correctly).

### Figures + paper + report integration
- Two new figures (`make_figures.py`): **fig7** transition-probability heatmap
  (cohort + CI−nonCI diff) and **fig8** effect-size ranking (red = significant).
- Report gained **§4 Sleep-Stage Dynamics & Fragmentation** and **§5 Feature
  Significance** with explanations; §Quality renumbered to §6. Regenerated
  `eda/DATASET_REPORT.md` (19K → 27.5K chars).
- Paper (`cinc2026_sdg.tex` + `.md`): new dynamics/significance paragraph + both
  figures; all LaTeX labels/refs balance.
- `run_eda.py` now runs transitions inline (`--skip-transitions` to opt out);
  significance computed at report time off the CSVs. eda README documents the
  full pipeline order.

---

## 2026-08-05 (later) — Dataset tree, viewer tooltips + logo, report explanations

Follow-up on the same day to make everything self-explanatory and better
organized.

### Dataset tree + samples — `eda/DATASET_TREE.md` (271 lines)
New reference doc (`scripts/eda/dataset_tree.py`, run on pdmle) giving the full
picture of *what we have*: directory tree with per-site file counts, the
`sub-<PID>_ses-<N>` naming convention, every `demographics.csv` column
(dtype/missingness/distribution) + a real sample row, `ICD_codes_CI.csv` schema
+ top-15 codes, a representative physio EDF header (19-ch BIDMC montage,
header-only), CAISR annotation value distributions + a decoded hypnogram
snippet, expert-annotation comparison, and a modality-coverage table. 🔬 All
values measured (215 GB total on disk; 0 EDF read errors).
- ⚠️ **Surfaced two data gotchas the team should know:**
  1. `resp_caisr` contains an **undocumented code `3`** (~0.2% of samples) not
     in our code map {1,2,4,5}. Our AHI currently counts {1,2,4} and ignores 3
     — defensible (rare, undocumented) but noted, not silently changed.
  2. **Expert annotations use a different code convention** than CAISR (codes
     0/7/9 appear) — decode expert files with the expert convention, not the
     CAISR maps.

### Viewer — hover explanations + Edwards logo (`scripts/viewer/`)
- `glossary.py` (new): one source of truth for plain-language definitions of
  every event index (AHI, arousal, PLMI, apnea subtypes, RERA), sleep stage,
  derived metric, channel role, and demographic field. Served at
  `/api/glossary`.
- `index.html`: added a delegated tooltip engine — hover (or keyboard-focus) any
  event tile, stage-legend item, channel chip, or demographic field to see its
  explanation. Cursor/dotted-underline affordance marks hoverable items.
- Edwards logo added to the header (`edwards_logo.png`, served from `/static/`
  with a path-traversal guard; falls back gracefully if absent).
- `app.py`: `/api/glossary` + `/static/*` routes; `channel_role()` classifier;
  `roles` map added to `/api/signals` responses (both the labels-only chip path
  and the trace path).
- ✅ Verified end-to-end **on pdmle against real data**: glossary/logo/static/
  patients/caisr/signals all return correctly; every channel in the sampled
  montages maps to a real role (0 "other"). A live-data test caught one bug —
  `roles` was missing from the labels-only response — now fixed and re-verified.
- ⚠️ Stale viewer processes from earlier sessions are still bound to 8050/8051
  running OLD code; restart the viewer under `arshia_ilaty_physio26` to pick up
  the new build. Redeployed bundle is at `/data-temp/physio-viewer/`.

### Dataset report explanations — `scripts/eda/report.py`
Each section now opens with a plain-language callout (blockquote) explaining the
terms for a mixed ML+clinical audience: what prevalence & 95% CI mean, why the
age breakdown matters (confounding), missingness, an **ICD-10 decoder table**
(every top code → its dementia/MCI meaning), biosignal/montage terms, and the
full sleep-metric glossary (stages, TST, efficiency, WASO, AHI, arousal, PLMI).
Regenerated `eda/DATASET_REPORT.md` (13.2K → 19.1K chars).

---

## 2026-08-05 — Dataset statistics, figures, paper, viewer, and deck

Goal: turn the raw dataset into (a) a comprehensive, trustworthy statistical
profile, (b) publication figures, (c) an interactive biosignal viewer, and
(d) presentation slides — so we understand the cohort we're modeling and can
communicate it.

### Data source
- Full dataset lives on the **pdmle** machine (`AWOR-PDMLEAPP01`),
  `/data-temp/shared-physionet26-dataset/extracted/`. Access notes are in
  `~/.claude` memory (`pdmle-dataset-access`). Read with the
  `arshia_ilaty_physio26` account (in `mlusers`); EDFs read header-only via
  `edfio` so the ~170 GB of samples are never loaded.
- 🔬 **Reconciliation:** 1,103 physiological EDFs = 1,103 demographics rows,
  0 orphans either direction. 1,090 have CAISR annotations, 1,097 have expert
  annotations.

### EDA suite — `scripts/eda/`
Modular, reproducible statistics over the whole cohort. Orchestrated by
`run_eda.py` → writes `eda/dataset_stats.json`, per-domain CSVs, and
`eda/DATASET_REPORT.md`. `dump_per_recording.py` joins CAISR summaries +
demographics + label into `eda/per_recording.csv` (1,090 rows).
- `common.py` — paths, site/stage/event code maps, channel-role classifier.
- `statutils.py` — numeric summaries, value counts, histograms, Wilson CIs.
- `stats_demographics.py`, `stats_biosignals.py`, `stats_sleep.py`,
  `stats_quality.py` — the four analysis domains.
- ✅ Hardened via a multi-agent code review; fixed 11 confirmed correctness
  bugs (AHI denominator = hours of *sleep* not recording; PLMI = periodic-only;
  WASO bounded by last sleep epoch; real missing-rate accounting; channel-role
  token-boundary matching; µ-sign vs Greek-mu unit handling; order-independent
  montage signatures; etc.). All fixes unit-tested locally with `edfio` stubbed.

### Key verified findings (🔬 all from `eda/DATASET_REPORT.md`)
- **Cohort:** 1,103 patients, one session each. 3 sites — BIDMC 857, Kaiser 192,
  Emory 54.
- **Prevalence:** 7.62% (84 positive / 1,019 negative, 0 missing).
  95% CI 6.19–9.33%.
- **Age confounder:** prevalence climbs 2.0% (50–59) → 7.3% → 17.5% → 36.1%
  (80+). This is exactly what the age-conditioned AUROC targets.
- **Site shift:** prevalence 6.5% (BIDMC) → 10.4% (Kaiser) → 14.8% (Emory)
  → motivates leave-one-site-out validation.
- **Biosignals:** EKG in 98.9% of files, core EEG + SaO₂ ~81%. Montages highly
  heterogeneous: BIDMC uses 42 distinct channel sets, Kaiser 9, Emory 8.
  16–88 channels/file, 20–512 Hz.
- **Sleep architecture (CAISR):** median sleep efficiency 76.8%, mean TST
  332 min, stages Wake 26% / N1 8% / N2 46% / N3 9% / REM 12%. Mean AHI 47.9/h
  of sleep, arousal index 39.7/h — a heavily sleep-disordered cohort.
  CAISR↔expert epoch agreement ~76%.
- **CI vs non-CI:** impaired patients show lower sleep efficiency, shorter TST,
  less N3 and REM; apnea/arousal indices more similar → discriminative signal is
  mainly in sleep architecture/continuity.
- **Missingness:** Age/Sex complete; BMI missing 75.9%, Time_to_Event 92.4%
  (positives only). 18 short recordings (<4 h; 9 under 1 h) flagged.

### Figures — `paper/figures/` (PNG + vector PDF)
Built with `scripts/eda/make_figures.py` + `figstyle.py` using a **validated**
Edwards palette (raw brand trio failed the dataviz colorblind/contrast checks,
so hues were snapped to passing values: red `#C8102E`, blue `#2166a0`).
1. age→prevalence gradient · 2. per-site prevalence vs cohort mean ·
3. mean sleep-stage composition · 4. CI vs non-CI sleep metrics (boxplots) ·
5. montage heterogeneity · 6. data completeness.

### Paper integration — `paper/cinc2026_sdg.{tex,md}`
Added **§2.2 "Dataset characteristics"**: real-number cohort table (per-site
CIs) + figures 1–5, prose linking the age gradient and montage heterogeneity to
design choices. All LaTeX labels/refs resolve.
- 📌 Model-performance cells in Tables 1–2 (AUROC/AUPRC/ablation) remain
  `[FILL]` — to be filled from LOSO/ablation results, **not fabricated**. Only
  Reward = 0.168 is verified.

### Biosignal viewer — `scripts/viewer/`
Single-page web app: pick a patient → demographics + CAISR hypnogram/stage-%/
event indices + downsampled PSG traces. Pure Python stdlib (`http.server`) +
numpy + edfio — **no Flask, no CDN, firewall-safe**. Server decimates each
channel to ~2,500 pts (min/max envelope) so 170 MB EDFs never reach the browser.
- Run on pdmle as `arshia_ilaty_physio26`: `cd /data-temp/physio-viewer &&
  bash run.sh` → binds 127.0.0.1:8050; tunnel with
  `ssh -L 8050:127.0.0.1:8050 arshia_ilaty_physio26@AWOR-PDMLEAPP01`.
- ⚠️ Verified structurally (HTTP responses, JSON payloads, JS syntax); not yet
  rendered in a browser — no headless browser available on either machine.

### Presentation deck — `PhysioNet Challenge 2026 - Scoring Metrics.pptx`
Appended a **"The dataset behind the metric"** section (`scripts/slides/
build_dataset_slides.py`, idempotent): 1 section header + 6 content slides
(cohort · confounders · biosignals/montages · sleep architecture · CI-vs-non-CI
signal · data quality), placed right before the "What this means for our
approach" takeaways. Agenda updated to 7 items; every slide has speaker notes.
Styling matches the existing Edwards template (fonts, palette, card/table/chip
patterns). Deck grew 18 → 25 slides.
- ✅ Verified: correct slide order, all figures embed, notes on every slide, no
  shape out of bounds, no figure↔text overlap, agenda renumbered 1–7.
- ⚠️ Structural verification only — no LibreOffice/renderer on this box.
- Original backed up to `/tmp/ScoringMetrics.backup.pptx` before editing.

---

## Earlier work (from git history, pre-2026-08-05)

- 2026-07-06 — Reward reached **0.168** (verified) on the primary metric.
- 2026-06 — Gap analyses; updates for the official phase.
- 2026-03 — PSG montage handling.
- 2026-02 — Initial commit.

Model/training code lives in `team_code.py`, `feature_prep.py`, `claude/`, and
`scripts/` (training, S3, reward optimization).
