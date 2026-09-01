# JEPA exploration — PhysioNet Challenge 2026 (Team SDG)

Self-supervised **Joint-Embedding Predictive Architecture** experiments for predicting
binary `Cognitive_Impairment` (CI) from overnight PSG. This branch runs **after the
Challenge deadline** — it is *pure exploration*, with no shippable / frozen-CPU-container
constraint. GPU, raw signal, and heavy models are all fair game.

**Question it answers:** does a learned SSL representation of the raw sleep signal add
anything the 436 handcrafted features (the "champion" GBM) don't already capture? Every
experiment is gated the same way (below), so the ten verdicts are directly comparable.

> **Bottom line: no.** Ten consecutive NO-SHIPs across every axis — per-epoch EEG,
> multimodal, night-dynamics, and end-to-end fine-tuning. Some reps are *genuine*
> (real label-free signal, AUROC ≈0.63) but all are **redundant** with handcrafted EEG
> under LOSO. The wall is **n_pos ≈ 84 / 3-site LOSO**, not the architecture.

## Evaluation protocol (identical across experiments)

`jepa_gate_ab.py` / `ft_eval.py` compare three arms on the **same rows and folds**, with
the same fitter (`feature_prep` GBM + BMI imputer) and metric path (`evaluate_model`
AC-AUROC / AUROC / reward) + paired bootstrap:

- **champion** — 436 handcrafted features → GBM
- **jepa-only / jepa-ft** — the SSL night representation (or fine-tuned OOF prob) alone
- **fusion** — `[436 features, JEPA representation]` → GBM

**CV:** `--cv site` = **LOSO** (leave-one-site-out; the real site-shift test, primary) and
`--cv patient` = **LOPO** (5-fold StratifiedGroupKFold by pid, for tighter CIs). Champion
LOSO headline (full cohort): **AC-AUROC 0.6352 / AUROC 0.7117 / reward +0.2738**.
Encoders are pretrained label-free on all nights (leak-free target); LOSO is enforced in
the gate head. A "SHIP" requires the fusion Δ over the champion to clear zero.

## Experiments (chronological) and measured verdicts

| # | Files | Idea | jepa-only AUROC | fusion Δ AC-AUROC (LOSO) | Verdict |
|---|---|---|---|---|---|
| — | `ts_jepa.py` | Temporal JEPA over per-epoch EEG *summary* features | 0.514 (≈chance) | −0.030 [−0.084,+0.022] | NO-SHIP (floor) |
| 6 | `raw_eeg_jepa.py`, `export_raw_eeg.py` | Raw 6-ch EEG, epoch-level data2vec, mean-pool readout | 0.5030 (chance) | −0.039 [−0.098,+0.024] | NO-SHIP |
| 7 | `raw_eeg_jepa_v2.py` | + per-channel tokens, I-JEPA block mask, CLS readout | **0.6294** (real) | −0.0113 [−0.087,+0.068] | NO-SHIP (method validated) |
| 8 | `raw_mm_jepa.py`, `export_raw_multimodal.py` | 15-channel multimodal, presence-aware I-JEPA | 0.4997 (chance) | −0.0349 [−0.086,+0.019] | NO-SHIP (dilutes EEG) |
| 9 | `night_jepa.py` | Level-2 night dynamics (5-min blocks) + Exp-C surprise | 0.6323 (real) | −0.0361 [−0.089,+0.013] | NO-SHIP |
| 10 | `finetune_v2.py`, `ft_eval.py` | End-to-end fine-tune of V2 + gated-attention MIL head | ft-only 0.6196/0.6761 | **+0.0031** [−0.008,+0.014] | NO-SHIP (least-bad) |

(The global NO-SHIP counter includes earlier non-JEPA CI-improvement screens; #6–#10 are
this branch. Fine-tuning (#10) is the only run where fusion does not *hurt* the champion —
LOSO Δ is positive, LOPO dead neutral (−0.0007) — but still no significant lift.)

## Module map

| File | What it is |
|---|---|
| `export_raw_eeg.py` | Decode 6 canonical scalp derivations → CAR → 64 Hz → robust z-score → stage-aligned 30 s epochs → per-recording npz |
| `export_raw_multimodal.py` | As above for the 15-slot montage (+ECG/EOG/EMG/airflow/effort/SpO2/legs), role-gated, presence-aware |
| `raw_eeg_jepa.py` | V1 epoch-level EEG JEPA: pack → pretrain (data2vec, EMA teacher, AMP) → embed (stage-aware pooled CLS) |
| `raw_eeg_jepa_v2.py` | V2: per-channel patch tokens + chan-emb, 2D I-JEPA block mask, narrow predictor, CLS readout |
| `raw_mm_jepa.py` | V2 extended to 15 channels with presence-aware masking (absent channels excluded from attention/readout) |
| `night_jepa.py` | Level-2: reuse frozen V2 teacher → 5-min block sequence → temporal I-JEPA over the night + Exp-C leave-one-block-out prediction-error features |
| `finetune_v2.py` | End-to-end supervised fine-tune of the V2 backbone + stage-aware gated-attention (MIL) night head → OOF probs |
| `ts_jepa.py` | Stage-1 lightweight temporal JEPA over exported per-epoch EEG summary features (CPU) |
| `jepa_gate_ab.py` | The A/B gate: champion / jepa-only / fusion, LOSO + LOPO, paired bootstrap |
| `ft_eval.py` | Same gate for the fine-tuned OOF probabilities |
| `run_*.sh` | Orchestration entry points that chain the subcommands into each full experiment |

## Run order (on the GPU box)

Experiments run as `arshia_ilaty_physio26` (owns the EDF read access + the `exports/jepa/`
work dir) on a Tesla T4 16 GB, using the isolated `gpuenv` venv (`torch==2.5.1`, cu124).
Each entry point is self-contained:

```bash
# extract + pack + pretrain + embed + gate, per experiment
./run_stage1_rest.sh          # ts_jepa (stage-1 light)
./run_v2_screen.sh 15         # raw_eeg_jepa_v2, 15 pretrain epochs
./run_mm_extract.sh && ./run_mm_screen.sh   # multimodal
./run_l2_screen.sh            # night_jepa (Level-2 + Exp-C), LOSO + LOPO
./run_ft_screen.sh 10         # finetune_v2 + ft_eval, LOSO + LOPO
```

## Design / data notes

- **Data and artifacts never enter the repo.** The DUA keeps de-identified data on the
  box; the multi-GB packs (`packed_*.npy`), encoder checkpoints (`*.pt`), embeddings
  (`emb_*.npz`), OOF probs (`ftprob_*.npz`) and logs live under
  `/data-temp/physio-viewer/exports/jepa/` (and are gitignored here anyway). Only the code
  and aggregate results (this README, `PROJECT_LOG.md`) are committed.
- **Borrowed recipe.** V2 onward folds in fixes from public JEPA repos (ECG-JEPA,
  eeg-vjepa, PhysioJEPA, signaljepa): per-channel C×T tokens + channel embeddings, I-JEPA
  context-only masking + a narrow predictor, register/CLS + attentive readout, smooth-L1 on
  EMA-teacher LN targets, AdamW betas (0.9, 0.99) with WD-exclusion.
- **Stage codes** (from `team_code.py`): `{1:N3, 2:N2, 3:N1, 4:REM, 5:Wake, 9:Unknown}`;
  CAISR epoch-pool codes 0..4 = Wake/N1/N2/N3/REM, unknown = −1.
- Full narrative with all CIs is in `PROJECT_LOG.md` (newest-first) and the
  `jepa-exploration` memory.
