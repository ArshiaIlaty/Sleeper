# Research Proposal & Roadmap

## Predicting Cognitive Impairment from Overnight Sleep Studies — George B. Moody PhysioNet Challenge 2026

**Team:** SDG · Arshia Ilaty
**Date:** July 13, 2026
**Status:** Working submission scores **Reward = 0.168** (LOSO, official metric). This document (a) states the scientific rationale for the current system, and (b) lays out a prioritized roadmap to improve the score for the official phase.

---

## 1. Problem framing

The Challenge asks us to predict, per patient, a binary **cognitive-impairment** label from a single overnight polysomnogram (PSG) plus demographics, and to emit both a binary decision and a probability. Ranking is by a **prevalence-weighted Reward** that (i) pays `1/p − 1` for catching a positive at age-specific prevalence `p`, (ii) pays a small amount for a correct negative, and (iii) charges a flat `−1` for any error. Two properties of this metric drive every design decision:

1. **It is invariant to trivial strategies.** All-positive, all-negative, and random classifiers all score ≈ 0 at any prevalence (verified by simulation). Only genuine skill scores above zero.
2. **It is age-confounder-resistant.** The companion age-conditioned and age-weighted AUROCs only compare patients of similar age, so a model that merely relearns the age→risk gradient gains nothing.

**Design consequence:** we optimize the Reward directly, protect recall on the rare positive class, and deliberately avoid letting age act as a shortcut.

---

## 2. What we have built (and why it is sound)

| Component                   | Choice                                                                                            | Rationale                                                                                                                        | Evidence                                          |
| --------------------------- | ------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| **Feature substrate** | CAISR algorithmic annotations (sleep stages, respiratory/limb/arousal events) rather than raw EEG | Portable across montages/sites; captures sleep macro- and micro-architecture that is mechanistically linked to neurodegeneration | LOSO ablation: dropping EEG/EOG/chin gave no loss |
| **Autonomic block**   | ECG-derived HRV + SpO₂ desaturation burden                                                       | Autonomic dysregulation and nocturnal hypoxaemia are independent dementia risk factors                                           | Ablation-retained                                 |
| **Demographics**      | Sex, BMI, race, ethnicity —**age excluded**                                                | Excluding age forces signal to be physiological, improving the age-aware metrics and Reward                                      | Ablation: removing age improved age-AUROC/Reward  |
| **Missing BMI**       | Site-median then global-median imputation                                                         | Robust to site-specific missingness                                                                                              | —                                                |
| **Classifier**        | `HistGradientBoostingClassifier`, isotonic-calibrated                                           | Handles missing values natively, strong tabular baseline; calibration matters because the Reward is threshold-driven             | —                                                |
| **Domain shift**      | Per-site models + global fallback (mixture of experts)                                            | Multi-site recruitment (BIDMC, Emory, Kaiser) induces distribution shift                                                         | —                                                |
| **Decision rule**     | Threshold at prevalence π                                                                        | Reward is maximized near the base rate                                                                                           | See §3 for the provably-optimal refinement       |
| **Validation**        | Leave-one-site-out CV with the**official** `evaluate_model.py`                            | Estimates generalization to unseen sites, matching the hidden test protocol                                                      | `scripts/cross_validate_s3.py`                  |

**Assessment:** this is a confounder-aware, generalization-first design with an honest validation protocol. It is a strong foundation and is publishable as-is.

---

## 3. The single highest-value improvement (theory-backed)

The current threshold uses the **global** prevalence π. For this reward, the Bayes-optimal decision is:

> **Predict positive if the calibrated probability `q > p(age)`**, the *age-specific* prevalence.

**Derivation.** For a patient with posterior `q = P(y=1|x)`, expected reward of a positive call is `q(1/p − 1) − (1−q)`; of a negative call is `−q + (1−q)(1/(1−p) − 1)`. Setting positive ≥ negative reduces to `q/p ≥ (1−q)/(1−p)`, i.e. **`q > p`**. Because the evaluation defines `p` per age (from the prevalence file, gap = 2 yr), the optimal threshold is age-varying, not constant.

**Action:** replace `_reward_optimal_threshold` with an age-indexed threshold `p(age)` estimated from the training prevalence table. Low-risk, theoretically justified, directly targets the ranking metric.

---

## 4. Prioritized roadmap

### Tier 1 — high value, low risk (do first)

1. **Age-specific optimal threshold** (§3). Expected to lift Reward with near-zero downside.
2. **Calibration audit under LOSO.** The `q > p` rule is only optimal if `q` is well-calibrated *on held-out sites*. Add reliability diagrams per fold; if miscalibrated, try Platt vs isotonic and cross-site recalibration.
3. **Threshold robustness.** Report Reward across a threshold sweep per fold (already partially in `cross_validate_s3.py`) to ensure the operating point is stable, not overfit to one split.

### Tier 2 — moderate value

4. **Feature enrichment within the CAISR substrate:** spectral EEG band-power ratios (slow-wave activity, spindle density) *if* they survive LOSO ablation; periodic-limb-movement periodicity; SpO₂ event morphology (depth × duration). Gate every addition on the ablation harness — no free-parameter creep.
5. **Site-adversarial / domain-invariant training:** encourage features whose distribution is site-invariant (e.g., site-adversarial gradient reversal, or simple per-site standardization of features). Directly attacks the LOSO generalization gap.
6. **Model diversity:** blend HGB with a logistic / linear model on standardized features for a calibrated ensemble; linear members are robust under shift.

### Tier 3 — exploratory / higher risk

7. **Survival-aware labels.** The demographics carry `Time_to_Event` and `Time_to_Last_Visit`. Right-censored patients labeled negative are noisy negatives. A survival head (e.g., discrete-time hazard) or sample-weighting by follow-up time could de-noise labels — but must be reconciled with the binary Challenge target.
8. **Sequence models on epoch-level CAISR streams** (temporal CNN/transformer over hypnogram + event sequences) instead of hand-aggregated summaries. Higher capacity, higher overfitting risk on a multi-site set — only worth it if Tier 1–2 plateau.

### Continuous discipline

- Every change is accepted **only if** it improves pooled LOSO Reward *and* does not degrade the worst-site fold (protect the weakest site — that is what the hidden test punishes).
- Keep age out of the model unless an ablation proves it helps the *age-aware* metrics (it should not).

---

## 5. Risks & mitigations

| Risk                                              | Mitigation                                                                                 |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Overfitting to 3 training sites                   | LOSO as the*only* model-selection signal; prefer simpler models; watch worst-fold Reward |
| Calibration drift across sites                    | Per-fold reliability diagrams; cross-site recalibration                                    |
| CAISR annotations unavailable/partial on test     | Global fallback + NaN-tolerant HGB; degrade gracefully                                     |
| Label noise from censoring                        | Tier-3 survival weighting; sensitivity analysis                                            |
| Metric misalignment (optimizing AUROC not Reward) | Select on Reward; treat AUROC/AUPRC as diagnostics only                                    |

---

## 6. Deliverable

The accompanying paper (`paper/cinc2026_sdg.tex` / `.md`) is written to **Computing in Cardiology** format for submission after the official phase. Numbers verified from the repository (Reward = 0.168) are stated; all other quantitative cells are marked `[FILL]` for completion from `eda/loso_cv_*.json` — **do not** submit with placeholders.
