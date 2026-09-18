# Poster talking points — Team SDG, PhysioNet Challenge 2026

Presenter's brief. For each bullet: **Explanation** (plain words) · **Assumption** (why we decided to try it)
· **Did** (what we actually implemented) · **Takeaway** (the number + what to say if pushed).
All numbers are measured; the transfer-wall gap and the age-baseline framing are the two honest anchors.

---

## 0. The 30-second opener (say this first, at the title)

- **Problem:** predict future cognitive impairment from a single overnight sleep study + demographics,
  pooled across three hospitals.
- **Two scoring twists that shaped every decision:** (1) positives are rare → we're scored on a
  *prevalence-weighted reward*, not accuracy (a trivial classifier ≈ 0); (2) the metric is *age-conditioned*
  → you can't win by relearning "older ⇒ impaired."
- **Our answer:** interpretable sleep-architecture + autonomic features feeding a calibrated gradient-boosted
  tree, with the decision threshold *derived* from the reward. Official entry: **Reward 0.168 / AC-AUROC 0.748.**
- **Headline lesson:** five very different models (foundation encoder, tabular transformer, GNN, state-space,
  our tree) cluster in a narrow band — so the bottleneck is **cross-site transfer, not model power.**

---

## 1. The submission table (one line to say per row)

Frame the table as: *"Five approaches, one leaderboard — this is what the team tried and how they compare on
the age-conditioned metric."*

- **SleepFM (0.621).** *Explanation:* a frozen large-scale sleep foundation encoder + a small trained head.
  *Assumption:* pretrained multimodal representations beat handcrafted features. *Did:* Kelvin's team, 516-d
  embedding + MLP head. *Takeaway:* highest raw ranking, but by a noise-scale margin, and it can't be
  regenerated inside the frozen submission container.
- **TabPFN (0.604).** *Explanation:* a pretrained tabular transformer ensembled with XGBoost. *Assumption:*
  in-context tabular learning generalizes on small data. *Did:* Kelvin's team, over Markov stage-transition +
  EEG features. *Takeaway:* competitive, mid-pack — foundation models didn't decisively win.
- **Ours (0.570).** *Explanation:* interpretable CAISR sleep + autonomic features → per-site gradient
  boosting, reward-derived threshold. *Assumption:* on a small, confounded, 3-site problem, interpretable +
  calibrated beats deep. *Did:* our production stack. *Takeaway:* chosen for the official entry (Reward 0.168)
  because it's regenerable, calibrated, and defensible under the reward metric.
- **Mamba (0.545).** *Explanation:* a selective state-space sequence model over per-epoch sleep tokens +
  LightGBM. *Assumption:* learning temporal dynamics end-to-end beats summary statistics. *Did:* Kelvin's team.
  *Takeaway:* underperformed both the encoder and engineered features.
- **PhysioGraph (0.533).** *Explanation:* Elahe's temporal graph neural network over an 8-node brain–body
  network (+ a population GCN). *Assumption:* modeling cross-system coupling adds signal beyond flat features.
  *Did:* SleepFM node features + LightGBM fusion. *Takeaway:* landed at the age-only baseline — the graph
  mostly transported age, which the metric removes.

**The internal-vs-hidden column is the story:** our model drops from **0.83 internal → 0.57 hidden** (the
transfer wall), while the graph model is **≈0.52 in both** (an honest null, at the age baseline). Say:
*"The gap between the two columns is the cross-site generalization penalty — that's the real problem here."*

---

## 2. Key findings (present these as the "what we learned")

1. **Stage-aware sleep biomarkers carry the CI signal.**
   *Explanation:* the predictive signal is in sleep architecture and continuity, not raw EEG or apnea load.
   *Assumption:* neurodegeneration disrupts sleep structure before diagnosis. *Did:* built CAISR
   stage/efficiency/WASO/fragmentation features. *Takeaway:* swapping a transformer for these features lifted
   AC-AUROC **0.417 → 0.677**; 14/48 features significant; **AHI was not** (d=0.19) — it's structure, not apnea.
2. **Stage-transition dynamics reveal destabilized deep sleep.**
   *Explanation:* it's not just how much N3/REM you get, but whether you can hold it. *Assumption:* CI erodes
   the stability of deep and REM sleep. *Did:* computed stage-transition (Markov) features. *Takeaway:* CI
   patients show **N3→N3 −5.4 pp, REM→REM −4.8 pp** self-persistence. *(If asked "does it add over
   percentages?" — say we see the signature descriptively; the incremental ablation is future work.)*
3. **A calibrated multi-model ensemble improved per-site stability.**
   *Explanation:* blending complementary learners steadies cross-site behavior. *Assumption:* no single model
   is robust across heterogeneous sites. *Did:* stacked LogReg + LightGBM + MLP + Cox via a logistic
   meta-learner. *Takeaway:* LOSO **reward 0.241**, trading a little ranking level for stability. *(This
   replaces the earlier "TabPFN+XGBoost" wording — that's a teammate's entry, not ours.)*
4. **Calibration is what makes the decision rule work.**
   *Explanation:* the reward rule needs probabilities on a trustworthy scale to pick who to flag. *Assumption:*
   raw tree scores aren't comparable across sites. *Did:* isotonic calibration + threshold at q>p. *Takeaway:*
   calibration lifted cross-site transfer reward from **≈−0.01 to +0.171** (≈ oracle +0.177). *(Honest note:
   it enables the threshold; it did not by itself reduce cross-site calibration error.)*

---

## 3. Other strategies investigated (why we tried it, what happened)

Frame the whole section as: *"We stress-tested the obvious ways to do better — most were neutral-to-negative,
which is itself the evidence that the ceiling is data, not modeling."*

- **SleepFM embedding fusion.** *Assumption:* foundation-model embeddings add signal to engineered features.
  *Did:* frozen encoder + head, and PCA embeddings fused into XGBoost. *Takeaway:* no robust LOSO gain over
  features alone (5 passes), and the encoder can't be regenerated in the frozen container.
- **Age handling + feature selection.** *Assumption:* dropping age removes the confound; ranking finds the
  real signal. *Did:* excluded age, added NeuroKit2 per-stage HRV/EEG/respiration features, ran Boruta.
  *Takeaway:* confirmed a small stable subset (~5 of 436) but found no signal beyond the core block.
- **Selective state-space (Mamba).** *Assumption:* end-to-end temporal modeling beats summary statistics.
  *Did:* Mamba over CAISR epoch tokens + LightGBM head. *Takeaway:* underperformed both SleepFM and the
  engineered features — deep sequence modeling didn't pay off at this scale.
- **Graph neural networks (PhysioGraph).** *Assumption:* cross-system coupling (brain–heart–respiration)
  carries CI signal. *Did:* temporal GNN over an 8-node per-patient network + population GCN. *Takeaway:*
  internal LOSO AC-AUROC ≈0.52, at the age baseline (0.537); the graph transported mostly age.
- **Network-physiology coupling.** *Assumption:* time-lagged coupling networks capture dynamics flat features
  miss. *Did:* time-delay-stability + Granger networks. *Takeaway:* recovered real signatures (EMG↔cortex
  decoupling in N2) but they were redundant with EEG+HRV and not predictive at this n.
- **Survival modeling.** *Assumption:* using diagnosis *timing* beats a binary label. *Did:* Cox/survival
  heads over the censored follow-up fields. *Takeaway:* C-index ≈0.675, SleepFM-Cox reward only 0.017 — too
  censored/sparse to beat the binary GBM on the scored metric.
- **Calibrated stacking ensemble.** *Assumption:* blending learners raises the ceiling. *Did:* LogReg + LGBM +
  MLP + Cox via a logistic meta-learner on nested OOF folds. *Takeaway:* LOSO AC-AUROC 0.582 (reward 0.241),
  only +0.045 over the age baseline and below a single shallow LightGBM.
- **Site harmonization.** *Assumption:* removing batch effects fixes cross-site transfer. *Did:* ComBat,
  per-site z-scoring, per-site rank-normalization. *Takeaway:* ComBat hurt; rank-norm was the only consistent
  lever (+0.026 AC-AUROC), but residual site shifts remained.
- **Near-term-positive training.** *Assumption:* restricting to soon-diagnosed positives sharpens a diffuse
  target. *Did:* trained only on positives diagnosed within two years. *Takeaway:* shrank the scarce positive
  class → higher variance, no net gain; reverted.
- **Data cleaning + augmentation.** *Assumption:* montage/line-noise heterogeneity is hurting us. *Did:* notch
  filtering, resampling to a common rate, synthetic missing-channel augmentation (42 montages, 200–512 Hz).
  *Takeaway:* improved robustness but not the score — annotation-level features are sampling-rate invariant.
- **Coarse age encoding.** *Assumption:* age bands are a safe middle ground between excluding and using age.
  *Did:* one-hot age-group features. *Takeaway:* still reintroduced the age→risk gradient the metric
  neutralizes, so we reverted to full exclusion.

---

## 4. The transfer wall (the honest centerpiece — expect questions here)

- *Explanation:* the same model that looks strong within a site collapses on a new site.
- *Numbers:* within-site AUROC **0.869** / AC-AUROC **0.832** / Reward **+0.474**; cross-site (LOSO) AUROC
  **≈0.66** / AC-AUROC **≈0.60**.
- *Assumption we validated:* with only three visible sites, the model can't characterize the distribution
  shift it will face on hidden sites.
- *Takeaway:* *"More patients close the within-site gap; only three sites cap new-site generalization. The
  fix is more training domains — external cohorts like SHHS/MrOS/MESA — not a bigger model."*

## 5. Limitations (say plainly)

- Binary target discards diagnosis timing (`Time_to_Event`) and treats right-censored patients as confirmed
  negatives — label noise.
- Only three visible sites; reliance on CAISR annotation quality at test time (mitigated by NaN-tolerant
  models + a global fallback); small positive count keeps confidence intervals wide.

## Quick Q&A anticipations

- *"Why not just use the foundation model — it scored highest?"* By a noise-scale margin on plain AUROC; it
  offered no age-conditioned/reward advantage and couldn't be regenerated in the frozen container.
- *"Why drop age when it's the strongest feature (d=1.08)?"* Because the metric is age-conditioned — age
  raises plain AUROC but not the score that ranks us, and it collapses cross-site transfer.
- *"Did the graph model fail?"* No — it's an honest null: it validated at the same level internally and on the
  hidden test, both at the age baseline. It tells us the ceiling is data-limited.
