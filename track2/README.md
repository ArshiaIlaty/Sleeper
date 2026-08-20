# Track 2 — Full-feature submission (Team SDG, Challenge 2026)

A second, independent Docker submission that ships **every beneficial handcrafted
feature we built**, on top of the Track-1 arch core. Where Track 1 is tuned for
cross-site transfer (LOSO), Track 2 is the **within-distribution champion** — it
hedges the single biggest strategic unknown: *is the hidden test set held-out
patients from the same 3 sites (→ Track 2 wins big) or entirely new sites (→
Track 1's LOSO-optimised model wins)?* Submitting both covers both cases.

## What's different from Track 1 (repo root)

Track 1 = `submit` preset (99 features: demographics-no-age + autonomic + CAISR
macro-architecture + arch transition/arousal block).

Track 2 = `submit_full` preset = Track 1's core **plus the vendored signal-feature
block** (`sig_features/`, 336 features):

| Block | Features | Source module |
|---|---|---|
| `nk__` | 183 | per-stage NeuroKit2 HRV (time+freq+Poincaré+DFA+SampEn), EEG complexity, respiration rate |
| `report__` | 122 | EEG band power / spindles (`eeg_spectral`), oxygenation (`oxygenation`), respiratory events (`resp_events`), REM-density |
| `micro__` | 31 | SO–spindle coupling (`eeg_coupling`), REM-atonia/RSWA (`emg_atonia`), CAP (`cap_events`) |

These previously existed only as **offline cache-builder scripts** in
`scripts/viewer/` that read raw EDF on the analysis box and emitted CSVs. Track 2
**ports them to run at inference** inside the frozen container.

## How the port works

- `sig_features/` vendors the 8 compute modules **verbatim** (cross-imports made
  package-relative), plus the two helpers their exporters relied on but that lived
  in the viewer's `app.py` / `preprocess.py`: `channel_role` (montage → role) and
  `smooth_stages` (the CAISR-staging preprocessing). Nothing else from the viewer.
- `sig_features.signal_feature_row(phys, phys_fs, algo, algo_fs)` reproduces the
  three offline exporters' exact per-record recipe (channel picking → per-stage
  functions → fixed-order flatten) in **one physio-EDF pass**. Returns a
  fixed-order 336-float vector; any missing stream / failed value → NaN; never
  raises.
- `team_code.extract_signal_features` calls it; `extract_all_features` loads the
  physio EDF **once** (shared with the autonomic block) and the CAISR EDF once,
  and appends the signal block under `include_signal`.
- Single pooled/site-MoE head (no separate Kaiser alt preset — that 42-feature
  head is Track-1-specific and would clash with the full vector).

## Dependencies added

`neurokit2==0.2.10` + its hard install-time deps `matplotlib==3.9.2`,
`requests==2.34.2` — the exact versions verified to install on the
`python:3.10.1-buster` base image. Everything else is unchanged from Track 1.

## Parity & verification

- `verify_sig_parity.py` — value-parity of the in-container extraction vs the
  validated offline `*_features_standard.csv` on real records (per-block
  NaN-pattern agreement = the wiring gate; small HRV drift is expected from the
  0.2.13→0.2.10 neurokit version delta and does not indicate a bug).
- Frozen-harness end-to-end: `train_model.py` / `run_model.py` on a 60-record
  `submit_test` slice must train, save `model.sav`, and score every site with no
  0.0-padding regression.

## Layout

Frozen harness files (`helper_code.py`, `evaluate_model.py`, `run_model.py`,
`train_model.py`, `create_labels.py`) and the `Dockerfile` are **byte-identical**
to the root Track-1 submission. Only `team_code.py`, `feature_presets.py`,
`requirements.txt`, and the new `sig_features/` package differ.
