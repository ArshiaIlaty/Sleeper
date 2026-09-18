# Discussion (draft) — Team SDG, PhysioNet Challenge 2026

> Drafted for `cinc2026_sdg.{md,tex}` §4. AUROC values are the team's leaderboard snapshot (plain AUROC,
> for cross-approach comparison); our official entry is better summarized by the primary metric
> (Reward 0.168 / AC-AUROC 0.748). Method attributions for the foundation-model entries were verified
> against the commit history of the team's `physionet26` repository.

## Discussion

Our entry deliberately favors **interpretable, annotation-level features and a calibrated gradient-boosted
tree** over end-to-end deep learning, and pairs that model with a decision threshold **derived from** the
Challenge reward (predict positive iff the calibrated probability exceeds the prevalence, `q > p`). It sits
inside a broader, multi-pronged team effort in which several fundamentally different inductive biases were
carried all the way to leaderboard submissions: a frozen **SleepFM** multimodal foundation-model encoder
with a trained head (AUROC 0.621), a **TabPFN** tabular-transformer ensemble over stage-transition (Markov)
and EEG features (0.604), a **Mamba** selective state-space encoder over CAISR epoch tokens with a LightGBM
head (0.545), and our handcrafted-feature system (0.570).

The most informative result is how **narrow** that band is. Four approaches spanning foundation models,
in-context tabular transformers, state-space sequence models, and hand-crafted physiology land within
roughly 0.11 AUROC of one another. The pretrained SleepFM encoder edged ahead on raw ranking, but it did so
by a margin comparable to split-level noise, could not be regenerated inside the frozen submission
container, and offered no advantage under the age-conditioned, reward-based scoring that actually ranks the
Challenge — where calibration and a defensible operating point matter more than a fraction of a point of
AUROC. Our interpretable system was therefore selected for the official reward-optimized entry
(**Reward 0.168, AC-AUROC 0.748**, large-cohort training). The convergence of such different models on a
similar ceiling is the clearest evidence that the binding constraint is **not model capacity or feature
richness but cross-site generalization**: with only three visible training sites, every approach must
extrapolate to distribution shift it has too few domains to characterize (within-site AUROC ≈ 0.87 collapses
to ≈ 0.66 under leave-one-site-out). More training *domains* — not larger models — is the lever most likely
to raise the ceiling.

### Limitations

- **Binary prediction does not explicitly model time-to-event information.** The labels carry follow-up
  fields (`Time_to_Event`, `Time_to_Last_Visit`); collapsing them to a single binary target discards the
  prognostic *timing* of diagnosis and treats right-censored, not-yet-diagnosed patients as confirmed
  negatives, injecting label noise into the majority class. A survival formulation could in principle
  exploit this structure (below), but did not improve the scored metric at the available positive count.
- **Only three visible sites** bound the diversity of distribution shift we can model, so LOSO is an
  optimistic-to-pessimistic proxy for the hidden test rather than a tight estimate.
- **Dependence on CAISR annotation availability/quality** at inference (mitigated by NaN-tolerant models and
  a global fallback), and **a small positive count** (84 in the standard release, 497 in the large cohort)
  that keeps split variance large and confidence intervals wide.

### Other strategies investigated

Each of the following was implemented and evaluated with the official scorer under leave-one-site-out
cross-validation; unless noted, none produced a robust improvement over the interpretable baseline on the
primary reward, reinforcing that the bottleneck is cross-site transfer rather than under-modeling.

- **Survival modeling for time-to-diagnosis.** Cox/survival heads over the follow-up fields (Harrell
  C-index ≈ 0.675, matching the binary C-statistic); a SleepFM-based Cox variant reached AC-AUROC 0.698 but
  a reward of only 0.017. The timing signal is too sparse and heavily censored at this scale to beat the
  binary GBM on the scored metric.
- **Site harmonization.** ComBat *removed* discriminative signal along with the site shift (worse); per-site
  z-scoring gave a small ranking gain but did not fix transfer; **per-site rank-normalization** was the one
  batch-effect correction with a consistent LOSO ranking lift. Residual site-level score shifts remained the
  core obstacle to threshold transfer.
- **Physiological feature engineering.** Sleep-stage dynamics (stage-transition Markov matrices,
  fragmentation, bout statistics), brain-age–style spectral descriptors from sleep EEG, nonlinear HRV (DFA,
  sample entropy), and respiratory/oxygenation burden. Several carry genuine univariate signal (age
  *d*=1.08; periodic limb movements, reduced REM, elevated WASO), but families added beyond the core block
  gave diminishing, largely neutral cross-site returns — notably, apnea–hypopnea index was **not**
  significant, locating the signal in sleep architecture and continuity rather than respiratory-event load.
- **Foundation-model embedding ensembles.** A frozen SleepFM encoder feeding a small trained head (the
  team's single highest-AUROC entry, 0.621), and PCA-compressed embeddings fused with engineered features
  into XGBoost. Neither fusion produced a robust LOSO gain over engineered features alone, and the encoder
  could not be regenerated inside the frozen submission container, so neither was shipped.
- **Data cleaning and augmentation.** Mains **notch filtering** (A/B-toggleable), resampling to a common
  rate, and **synthetic missing-channel augmentation** to harden the model against severe montage
  heterogeneity (42 distinct montages at one site alone, sampling rates 200–512 Hz). These improved
  robustness but not the scored metric — the annotation-level features are largely sampling-rate invariant.
- **Restricting positives to near-term diagnoses.** Training only on positives diagnosed within two years,
  to sharpen an otherwise diffuse target. Shrinking the already-scarce positive class raised variance
  without a net gain.
- **Graph-based modeling (Network Physiology / PhysioGraph).** Time-delay-stability and Granger cross-system
  coupling networks (brain–heart–respiration). Descriptive signatures were recovered (e.g., EMG↔cortex
  decoupling in N2), but they proved redundant with the EEG and HRV features and were not predictive at this
  sample size.
- **Selective state-space (Mamba-style) sequence models.** A Mamba encoder over CAISR per-epoch token
  sequences with a LightGBM head (leaderboard 0.545), learning temporal sleep dynamics end-to-end. It
  underperformed both the SleepFM encoder and the interpretable features, indicating deep sequence modeling
  did not beat engineered summaries on this cohort.
- **Age handling and feature selection.** Age was deliberately **excluded** as the confounder the
  age-conditioned metric discounts; we added NeuroKit2-derived per-stage HRV/EEG/respiration features and
  applied permutation/Boruta feature-selection ranking (which confirmed a small stable subset of the ~436
  candidates). Selection validated the feature set but did not extend the signal.
- **Coarse age encoding.** One-hot encoding of age *groups* as a middle ground between excluding age and
  using it directly still reintroduced the age→risk gradient the age-conditioned metric neutralizes, so we
  reverted to full exclusion.

---

# Concise version (poster / short-paper cut)

## Discussion

Our entry favors interpretable annotation-level features and a calibrated gradient-boosted tree with a
decision threshold *derived* from the Challenge reward (`q > p`), and sits within a broader team effort
that also carried a SleepFM foundation encoder (AUROC 0.621), a TabPFN tabular-transformer ensemble
(0.604), our handcrafted system (0.570), and a Mamba state-space model (0.545) to the leaderboard. The
telling result is how narrow that band is: four very different approaches converge within ~0.11 AUROC. The
pretrained encoder edged ahead on raw ranking but by a noise-scale margin, could not be regenerated in the
frozen submission container, and offered no advantage under the age-conditioned reward that actually ranks
the Challenge — so the interpretable model was chosen for the official entry (**Reward 0.168, AC-AUROC
0.748**). The convergence shows the binding constraint is **cross-site generalization, not model capacity**:
within-site AUROC ≈ 0.87 collapses to ≈ 0.66 leave-one-site-out with only three visible sites. More training
*domains*, not larger models, is the lever most likely to raise the ceiling.

## Limitations

Collapsing follow-up timing (`Time_to_Event`, `Time_to_Last_Visit`) into a binary target discards prognostic
timing and treats right-censored patients as confirmed negatives (label noise); only three visible sites
bound the distribution shift we can model; and the entry depends on CAISR annotation quality at test time
with a small positive count.

## Other strategies investigated

All evaluated under leave-one-site-out with the official scorer; none robustly beat the interpretable
baseline on reward, reinforcing that transfer — not under-modeling — is the bottleneck: **survival modeling**
for time-to-diagnosis (C-index ≈ 0.675; SleepFM-Cox reward 0.017); **site harmonization** (ComBat worse,
per-site rank-normalization the only consistent lever); **physiological feature families** (stage dynamics,
brain-age spectral, HRV, respiratory — AHI *not* significant); **foundation-model embedding ensembles**
(SleepFM head + PCA-fused XGBoost — no LOSO gain, not regenerable); **data cleaning/augmentation** (notch
filtering, synthetic missing-channel); **near-term-positive training** (≤ 2 yr); **graph-based modeling**
(Network Physiology / PhysioGraph — redundant with EEG+HRV); **selective state-space (Mamba)** sequence
models; and **age handling** (exclusion, NeuroKit2 features, Boruta selection, coarse age-band one-hot —
which reintroduced the confound).

---

# Key findings (with evidence)

Each claim is backed by a measured number; two of the original four are re-worded to match the evidence
(see the audit note under each).

1. **Stage-aware sleep biomarkers carry the CI signal.** Engineered sleep-architecture and continuity
   features lifted AC-AUROC from 0.417 to 0.677 over a transformer baseline (submissions 2027→2358); 14 of
   48 features were significant (reduced REM/N3, elevated WASO/wake), while AHI was not (d=0.19) — risk sits
   in *how* sleep is structured, not respiratory-event load.
2. **Stage-transition dynamics reveal destabilized deep sleep.** CI patients show lower stage
   self-persistence (N3→N3 −5.4 pp, REM→REM −4.8 pp), a signature not captured by stage percentages alone.
   *(Audit: descriptive evidence only — add an ablation delta to support the "complementary/beyond
   percentages" wording quantitatively.)*
3. *(Team)* **A calibrated multi-model ensemble improved per-site stability.** A stacking ensemble reached
   LOSO reward 0.241, trading a little ranking level for steadier cross-site behavior. *(Audit: this replaces
   the original "TabPFN + XGBoost improved robustness" claim — that is a teammate's entry, not our model, and
   its cross-site numbers are not in the accessible repo. Use TabPFN+XGBoost only if Kelvin supplies
   ensemble-vs-single LOSO numbers.)*
4. **Calibration is what makes the decision rule work.** Isotonic calibration enabled the reward-optimal
   threshold (q>p), lifting cross-site transfer reward from ≈−0.01 to +0.171 — close to the oracle ceiling
   (+0.177). *(Audit: re-worded — isotonic did not reduce cross-site ECE (pooled ECE worsened), so the
   citable win is that calibration enables the threshold, not "more reliable probabilities.")*

---

# Other strategies investigated (full, uniform)

> Correction: *Submission 5* (0.533) is Elahe's **SleepFM-PhysioGraph++ temporal GNN** (8-node per-patient
> network + population GCN), not the TDS/Granger coupling work — that is a separate, our-side exploration,
> split into its own bullet below. Hidden-test scores (0.621/0.604/0.545/0.533) are omitted here to avoid
> duplicating the results table; each bullet keeps only the exploration finding.

- **Foundation-model embedding fusion (SleepFM).** Frozen SleepFM encoder → small trained head, and
  PCA-compressed embeddings fused with engineered features in XGBoost. No robust LOSO gain over engineered
  features across five passes, and the encoder could not be regenerated inside the frozen submission
  container. Not shipped.
- **Age handling and feature selection.** Age excluded as the confounder the metric discounts; added
  NeuroKit2 per-stage HRV/EEG/respiration features, ranked by permutation/Boruta selection. Selection
  confirmed a small stable subset (~5 robust of 436 candidates) but did not extend the signal beyond the core
  feature block.
- **Selective state-space (Mamba) sequence models.** A Mamba encoder over CAISR per-epoch tokens with a
  LightGBM head, learning temporal sleep dynamics end-to-end. It underperformed both the SleepFM encoder and
  the engineered features, indicating deep sequence modeling did not beat interpretable summaries on this
  cohort.
- **Graph neural networks (PhysioGraph).** A temporal GNN over an 8-node per-patient brain–body network
  (plus a population GCN with age-banded k-NN edges), fused with LightGBM. Internal LOSO age-conditioned
  AUROC ≈ 0.52, at the age-only baseline (0.537) — its high plain AUROC (0.74) transported mostly age. A
  genuine win needs self-supervised pretraining on the large cohort.
- **Network-physiology coupling (exploratory).** Time-delay-stability and Granger cross-system coupling
  networks (brain–heart–respiration). Descriptive signatures recovered (EMG↔cortex decoupling in N2), but the
  coupling features were redundant with existing EEG + HRV features and not predictive at this sample size.
- **Survival modeling for time-to-diagnosis.** Cox/survival heads over the censored follow-up fields to
  exploit diagnosis timing. Harrell C-index ≈ 0.675 (≈ the binary C-statistic); a SleepFM-Cox variant reached
  AC-AUROC 0.698 but reward only 0.017. Too sparse/censored to beat the binary GBM on the scored metric.
- **Calibrated stacking ensemble.** LogReg + LightGBM + MLP + a Cox survival head, combined by a logistic
  meta-learner on nested out-of-fold predictions. LOSO age-conditioned AUROC 0.582 (reward 0.241), only
  +0.045 over the age baseline and below a single shallow LightGBM (~0.62). Traded ranking level for per-site
  stability.
- **Site harmonization.** ComBat, per-site z-scoring, and per-site rank-normalization to remove batch
  effects. ComBat removed discriminative signal (worse); per-site rank-normalization was the only lever with
  a consistent LOSO lift (+0.026 AC-AUROC, large cohort). Residual site-level score shifts remained the core
  obstacle.
- **Restricting positives to near-term diagnoses.** Trained only on positives diagnosed within two years, to
  sharpen an otherwise diffuse target. Shrinking the already-scarce positive class raised fold-to-fold
  variance without a net gain, so we reverted to the full positive set.
- **Data cleaning and augmentation.** Mains notch filtering (A/B-toggleable), resampling to a common rate,
  and synthetic missing-channel augmentation against montage heterogeneity (42 montages at one site,
  200–512 Hz). Improved robustness but not the scored metric — the features are largely sampling-rate
  invariant.
- **Coarse age encoding.** One-hot age-group bands as a middle ground between excluding age and using it raw.
  Still reintroduced the age→risk gradient the age-conditioned metric neutralizes, so we reverted to full age
  exclusion.

---

# Other strategies investigated (concise — one line each)

- **SleepFM embedding fusion.** Frozen encoder + head, and PCA embeddings fused with engineered features in
  XGBoost — no robust LOSO gain, and not regenerable in the frozen container.
- **Age handling + feature selection.** Age excluded; NeuroKit2 per-stage HRV/EEG/respiration features ranked
  by Boruta — confirmed ~5 stable features of 436, but no signal beyond the core block.
- **Selective state-space (Mamba).** End-to-end sequence model over CAISR epoch tokens + LightGBM head —
  underperformed both SleepFM and engineered features.
- **Graph neural networks (PhysioGraph).** Temporal GNN over an 8-node per-patient network + population GCN —
  internal LOSO AC-AUROC ≈0.52 (at the age baseline); high plain AUROC transported mostly age.
- **Network-physiology coupling.** Time-delay-stability / Granger brain–heart–respiration networks —
  signatures recovered (EMG↔cortex decoupling in N2) but redundant with EEG+HRV and not predictive.
- **Survival modeling.** Cox heads on censored follow-up — C-index ≈0.675, SleepFM-Cox reward only 0.017; too
  censored to beat the binary GBM.
- **Calibrated stacking ensemble.** LogReg + LightGBM + MLP + Cox via a logistic meta-learner — LOSO AC-AUROC
  0.582 (reward 0.241), only +0.045 over the age baseline.
- **Site harmonization.** ComBat / z-score / rank-normalization — ComBat hurt; per-site rank-norm was the only
  consistent lever (+0.026 AC-AUROC).
- **Near-term-positive training.** Restricting positives to ≤2-year diagnoses — shrank the scarce positive
  class, raising variance with no net gain.
- **Data cleaning + augmentation.** Notch filtering, resampling, synthetic missing-channel augmentation
  (42 montages, 200–512 Hz) — improved robustness but not the scored metric.
- **Coarse age encoding.** One-hot age bands — still reintroduced the age→risk gradient the metric
  neutralizes, so we reverted to full exclusion.
