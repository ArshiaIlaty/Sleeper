# Other Strategies Investigated — expanded (presenter reference)

A deeper version of the poster's "Other Strategies Investigated" section: for each item, **what the
model / foundation model / concept actually is**, **why we tried it and how it could help**, **what we
did**, and **the result + why it landed there** (with the comparative reasoning — e.g. why Mamba behaves
differently from SleepFM). Numbers are measured.

---

## A. Deep models and foundation models

### SleepFM — a sleep foundation model
- **What it is:** a large neural network *pretrained on huge volumes of polysomnography* (many thousands of
  nights), so it has already "seen" what normal and abnormal sleep physiology looks like. It reads the four
  PSG channel groups — brain activity (EEG/EOG), ECG, respiration, EMG — and compresses each night into a
  single fixed vector (in our build, a **516-dimensional embedding**). It was trained by *contrastive
  learning* (teaching the modalities to agree with each other), not on our labels.
- **Why we tried it / how it could help:** transfer learning. Our labeled cohort is small (~1,000 patients,
  84 positive); a model pretrained on outside data should carry general sleep structure our small set can't
  learn from scratch, and its embedding should be more robust to montage/equipment differences.
- **What we did:** froze the encoder, trained a small head on top, and separately fed PCA-compressed
  embeddings + our engineered features into XGBoost.
- **Result:** highest plain leaderboard ranking of the team's entries, but **no robust cross-site gain over
  engineered features**, and — decisively — the encoder **cannot be regenerated inside the frozen submission
  container** (no GPU/checkpoint), so we could not ship it.

### TabPFN — a tabular foundation model
- **What it is:** a transformer *pretrained on millions of synthetic tables* that classifies in a **single
  forward pass with no training** — you hand it the training rows and the test row together and it infers the
  answer "in context" (like a Bayesian predictor). It is state-of-the-art for *small* tabular problems and is
  well-calibrated by design.
- **Why we tried it / how it could help:** after feature extraction our problem *is* a small table; TabPFN is
  built exactly for the small-n regime where ordinary models overfit.
- **What we did:** teammate entry — TabPFN ensembled with XGBoost over Markov stage-transition + EEG features.
- **Result:** competitive but mid-pack; the tabular foundation model did not decisively beat plain
  gradient-boosted trees on this cohort.

### Mamba — a selective state-space sequence model
- **What it is:** a modern alternative to the Transformer for **long sequences**. A "state-space model" keeps
  a running memory as it walks through a sequence; Mamba makes that memory **selective** (input-dependent), so
  it chooses what to remember, and it runs in *linear* time rather than the Transformer's quadratic cost.
- **Why we tried it / how it could help:** a night of sleep is a very long sequence (thousands of epochs).
  Mamba can model the *evolution* of sleep across the whole night — the order and timing of stage changes —
  end-to-end, which flat summary features throw away.
- **What we did:** teammate entry — a Mamba encoder over CAISR per-epoch tokens with a LightGBM head.
- **Result & why it's worse than SleepFM here:** it **underperformed both SleepFM and our engineered
  features.** The key reason is *pretraining*: SleepFM imports knowledge from outside data, whereas this Mamba
  was **trained from scratch on ~1,000 patients** — it is more expressive but data-hungry, and there isn't
  enough labeled data to learn night-scale dynamics that beat handcrafted summaries. (Mamba's advantage —
  learning temporal order — only pays off with far more data or its own pretraining.)

### Graph Neural Networks — PhysioGraph (temporal GNN) and Population GCN
- **What a GNN is:** a network that operates on a **graph** (nodes connected by edges) and lets each node
  update itself using information from its neighbours ("message passing"). It's the right tool when the signal
  lives in the *relationships* between things, not the things alone.
- **Two graphs were built:**
  - **Patient-internal temporal graph (the shipped one):** nodes = physiological subsystems
    (brain, ECG, respiration, EMG + a sleep-stage node + event nodes) — **8 nodes per patient**; edges = how
    strongly those subsystems are coupled each epoch; a GNN reads the network and a temporal head (GRU) tracks
    it across the night. This is the *Network Physiology* idea — health is encoded in how organ systems
    coordinate.
  - **Population GCN:** nodes = *patients*, edges = similarity to age-matched neighbours (k-NN within ±2 y);
    message passing lets a patient "borrow" evidence from similar patients.
- **Why we tried it / how it could help:** cross-system coupling and patient-similarity structure are
  plausible early markers a flat feature vector misses.
- **Result:** internal LOSO age-conditioned AUROC **≈0.52 — at the age-only baseline (0.537)**. Its *plain*
  AUROC looked high (0.74) only because the structure it transported was essentially **age**, which the metric
  removes. Honest verdict: the graph did not beat the boosted tree; a real graph win needs self-supervised
  pretraining on the large (~6,600-patient) cohort.

> **One-line comparison for the poster:** *SleepFM and TabPFN import outside knowledge (pretrained), so they
> travel well on small data; Mamba and the GNNs learn from our ~1,000 patients alone, so they're more powerful
> in principle but starved of data here — which is why the pretrained models led and the from-scratch deep
> models landed near the baseline.*

---

## B. Physiological and statistical concepts

### Network Physiology — Time-Delay Stability (TDS) & Granger coupling
- **What it is:** a framework that measures how organ systems *coordinate*. **Time-Delay Stability** asks how
  stable the time lag is between bursts in two signals (stable lag = strong coupling); **Granger causality**
  asks whether one signal's past helps predict another's future (directed coupling).
- **Why we tried it / how it could help:** loss of brain–heart–respiration coordination is a candidate early
  signature of neurodegeneration.
- **Result:** we recovered real signatures (e.g., **EMG↔cortex decoupling in N2**), but the coupling features
  were **redundant with our existing EEG + HRV features** and not predictive at this sample size.

### Survival modeling (Cox proportional hazards)
- **What it is:** instead of predicting a yes/no label, survival models predict *time-to-event* and correctly
  handle **censoring** (patients who simply haven't been diagnosed yet). The C-index measures how well the
  model *ranks* who converts sooner.
- **Why we tried it / how it could help:** the binary label discards the follow-up timing; a survival model
  uses it and treats not-yet-diagnosed patients honestly instead of as confirmed negatives.
- **Result:** Harrell **C-index ≈0.675** (about the same as the binary model's ranking); a SleepFM-Cox variant
  hit AC-AUROC 0.698 but **reward only 0.017**. The timing signal is too sparse/censored to beat the binary
  boosted tree on the scored metric.

### Calibrated stacking ensemble
- **What it is:** train several different models (logistic regression, LightGBM, an MLP, a Cox head) and let a
  small **meta-learner** combine their predictions — using *nested out-of-fold* predictions so the combiner
  never sees a model's in-sample guesses.
- **Why we tried it / how it could help:** different models make different errors; blending them should be
  steadier across heterogeneous sites than any single model.
- **Result:** LOSO age-conditioned AUROC **0.582 (reward 0.241)** — only **+0.045** over the age baseline and
  *below* a single shallow LightGBM (~0.62). It bought per-site stability, not a higher ceiling.

### Site harmonization (ComBat, z-scoring, rank-normalization)
- **What it is:** techniques to remove "batch effects" — systematic differences between hospitals (equipment,
  populations). **ComBat** (from genomics) models each site as a shift/scale and subtracts it;
  **rank-normalization** replaces raw values with their within-site ranks so distributions line up.
- **Why we tried it / how it could help:** if features mean different things at different sites, transfer
  suffers; harmonizing should make a model trained on some sites work on a new one.
- **Result:** ComBat **removed real signal** (worse); per-site **rank-normalization was the only lever with a
  consistent lift (+0.026 AC-AUROC)**, but residual site-level score shifts remained — harmonization helped a
  little, didn't solve transfer.

### Physiological feature engineering + selection (NeuroKit2, Boruta)
- **What they are:** **NeuroKit2** is a validated open-source toolbox for physiological signal features
  (heart-rate variability, EEG band power, respiration). **Boruta** is a feature-selection method that keeps
  only features that beat randomly-shuffled "shadow" copies of themselves.
- **Why we tried it / how it could help:** add rich, standardized per-stage autonomic/EEG features, then prune
  to the ones that are genuinely predictive.
- **Result:** selection **confirmed a small stable subset (~5 of 436 candidates)** but did **not extend the
  signal** beyond the core sleep-architecture block — evidence the ceiling is data, not feature poverty.

### Restricting positives to near-term diagnoses
- **What it is / why:** train only on patients diagnosed within two years, on the assumption that far-off
  diagnoses are a "diffuse," noisier target.
- **Result:** it **shrank the already-scarce positive class**, raising fold-to-fold variance with no net gain;
  reverted to the full positive set.

### Data cleaning & augmentation (notch filter, resampling, synthetic missing channels)
- **What they are:** a **notch filter** removes mains electrical hum (50/60 Hz); **resampling** puts all
  recordings on a common sampling rate; **synthetic missing-channel augmentation** randomly drops channels
  during training so the model learns to cope when a montage lacks them.
- **Why we tried it / how it could help:** the data is severely heterogeneous (**42 distinct montages at one
  site, 200–512 Hz**); cleaning should reduce site-specific artifacts.
- **Result:** improved **robustness** but **not the scored metric** — our annotation-level features are
  largely sampling-rate invariant, so the raw-signal cleanup didn't change the answer.

### Age handling — full exclusion vs coarse age bands
- **Why age is the trap:** cognitive-impairment prevalence climbs steeply with age (2% at 50–59 → 36% at 80+),
  so a model can "cheat" by predicting age. The Challenge metric is **age-conditioned** precisely to neutralize
  that, so age boosts *plain* AUROC but not our score — and it *collapses* cross-site transfer.
- **What we tried:** excluding age entirely vs a middle-ground **one-hot age-band** encoding.
- **Result:** even coarse age bands **reintroduced the age→risk gradient** the metric removes, so we reverted
  to **full age exclusion** — vindicating the demographic-light design.
