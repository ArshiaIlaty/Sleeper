# Project Log — Team SDG, PhysioNet Challenge 2026

A running record of work on the George B. Moody PhysioNet Challenge 2026 entry
(predict `Cognitive_Impairment` from overnight PSG + demographics). Newest
entries first. Dates are absolute.

Legend: ✅ done & verified · 🔬 verified against data · 📌 needs follow-up ·
⚠️ known limitation.

---

## 2026-09-14 (Adversarial track: DANN + invariance-drop + demographic ablation; and resampling A/B) 🔬

**Why.** Two approved research directions, deadline passed → pure exploration (container frozen).
(1) Adversarial: can we *force* the encoder to learn site-invariant physiology? — batch A =
hand-crafted site-discriminative feature-drop + demographic ablation; batch B = a learned DANN with
a gradient-reversal layer. (2) Resampling: harmonize all signals to a common 200 Hz, since sites use
diverse hardware. Ran on pdmle through the production stack, aggregate metrics only. Drivers under
`scripts/adversarial/` and `scripts/resampling/`; writeup `scripts/adversarial/INVARIANCE_ABLATION_RESULTS.md`.

- 🔬 **Site-invariance feature-DROP REVERSES sign under the two-lever stack (largest ranking lever
  found).** Earlier (2026-08-18) drop-frac was rejected on RAW feats. Re-tested on **rank-norm +
  tuned-threshold**: dropping the top ~10–40% most site-discriminative features (between/within-site
  variance ratio, fit on TRAIN sites only) is now **net-positive** — large-cohort LOSO AC-AUROC
  0.6055→peak **0.6279 @40%** (stable ~0.62 at 10–20%); π-reward +0.0957→+0.1358; oracle
  +0.1765→+0.2270. Carried by Emory+BIDMC; Kaiser dips −0.010. Transfer-reward noisy (threshold
  sensitivity). The batch-effect removal (rank-norm) is what makes invariance-drop productive.
  Pairs with [[rank-norm-verdict]], [[site-invariance-drop-verdict]].
- ⚠️ **Demographic ladder — physiology-only HURTS, age-as-feature is a confound trap.** Dropping BMI
  (pure physiology) regresses transfer −0.0207. Adding **age as a feature** inflates raw AUROC
  0.6674→0.7360 but leaves **AC-AUROC flat (0.6028)** and **collapses transfer +0.1712→+0.0463** —
  the age-adjusted metric discounts exactly what age buys, and the reward threshold stops
  transferring. Vindicates the champion's demographic-light (BMI-only, no age) design.
  See [[demographic-ablation-verdict]].
- 🔬 **DANN = NO-SHIP, adversary inert.** GRL-MLP (λ∈{0,.01,.1,.3,1}, 3 seeds, 64-dim emb): the
  gradient-reversal barely dents site-decodability (z_site_acc 0.857→0.853 across all λ — the 64-dim
  bottleneck alone compresses site, not the adversary). MLP ranks fine (AC-AUROC 0.63–0.64) but
  **reward transfer is NEGATIVE, worse than the GBM's +0.1712.** Only 2 training sites → data-starved
  adversary; site is partly missingness-driven (rank+fill makes site 100% RF-decodable via the
  site-constant 0.5 fill columns). Hand-crafted feature-drop (above) beats learned soft invariance.
  Lands with the deep-net NO-SHIPs [[jepa-exploration]], [[minirocket-screen-verdict]].
  See [[dann-adversarial-verdict]].
- 🔬 **Resampling to 200 Hz = NO-SHIP (near-null, deployable metric slips).** ~97.6% of the cohort is
  already 200 Hz (BIDMC+Kaiser); only Emory's 500 Hz subset (317 rows, ~2.4%) changes → scoped
  Emory-only re-extraction at 200 Hz, spliced into native CSVs, paired LOSO A/B. **Emory's own fold
  clearly benefits** (AC-AUROC 0.6172→0.6789, AUROC 0.6322→0.6961 — consistency with the 200 Hz
  training sites), but pooled ranking barely moves (AC-AUROC +0.0076, AUROC +0.0010) and the
  **deployable transfer-reward DROPS −0.0256** (+0.1712→+0.1457); AUPRC −0.0216. Kaiser/BIDMC swings
  are pure LOSO ripple (their features unchanged; each fold trains on re-extracted Emory).
  **Confirms fs-robustness:** the pipeline extracts rate-invariant physical-Hz features at native
  rate, so harmonizing sampling rate is not a lever — papers resample because they feed *raw
  waveforms* to fixed-input-rate deep nets; we don't. See [[resampling-verdict]].

---

## 2026-09-10 (Within-night normalization lever + MiniRocket raw-EEG screen) 🔬

**Why.** Two follow-ups: (a) rank-norm removes the BETWEEN-site batch effect but not WITHIN-site,
per-NIGHT amplifier gain — so test physiology-referenced normalization on the stage-resolved
features, ON TOP OF per-site rank; (b) give the cheapest/most-deployable raw-signal method
(MiniRocket) one honest screen. Ran on pdmle through the production stack, aggregate metrics only.
Drivers `scripts/within_night/within_night_test.py` and `scripts/minirocket/minirocket_screen.py`;
writeups `scripts/within_night/WITHIN_NIGHT_RESULTS.md`, `scripts/minirocket/MINIROCKET_RESULTS.md`.

- 🔬 **Within-night ratio-over-N2 is the FIRST net-positive *feature-engineering* lever** in the
  whole exploration (every other feature lever came back neutral/harmful). Adding `f_stage / f_N2`
  for each stage-resolved metric, on top of per-site rank: large-cohort LOSO **AC-AUROC
  0.6055→0.6092, AUROC 0.6674→0.6710, reward-neutral**. Small but real and orthogonal.
- ⚠️ **The lift is non-uniform** — carried by BIDMC (+0.009 AC-AUROC, the 5075-row majority) and
  Emory (+0.010); **Kaiser dips −0.007** and pooled AUPRC slips 0.1542→0.1473. Marginal win,
  strongest on the two non-fine-tuned sites. N2 beats wake as the reference. Ships if the container
  reopens: **per-site rank + within-night ratio-over-N2**. Pairs with [[rank-norm-verdict]].
- ⚠️ **zwithin (z-across-stages)** is discrimination-neutral but lifts reward-threshold **transfer
  +0.007 / oracle ceiling +0.015** (reward-side only, optional). **abs-only zwithin wrecks transfer
  (NO)**, and **ratio+zwithin does NOT stack** — 897 feats on 497 positives drops below baseline and
  collapses transfer to +0.039. Promote ratio(N2) **alone**. Confirms within-night self-referencing
  is the productive axis, unlike further *global* normalization (z-score/INT/ComBat all tie-or-worse).
- 🔬 **MiniRocket raw-EEG screen: NO-SHIP (13th neutral/negative screen).** Central-EEG N2+N3 30s
  epochs → MiniRocket PPV (~10k feats) → ridge, LOSO-honest (params fit on train sites only). Both
  aggregate readouts miss the AUROC>0.6 gate — **pooled-LOSO 0.5552 (AC 0.5093), pooled 5-fold CV
  0.5670 (AC 0.4827, below chance)**. Only Emory (n=28, 8 pos) clears the gate = noise; Kaiser below
  chance (0.485). Raw waveform shape adds no site-transferable CI signal beyond the 436 engineered
  features — lands with the raw-signal NO-SHIPs [[jepa-exploration]], [[embedding-fusion-verdict]],
  matching the 2026-winner pattern (simple feats+GBM beats foundation-model complexity,
  [[competitor-methods-2026]]).

---

## 2026-09-10 (The other 4 "highest-value gaps": decision threshold, pairwise ranking, gain-invariance, event ablation) 🔬

**Why.** Having tested lever #1 (rank-norm) below, ran the remaining four gaps from
[[competitor-methods-2026]] as LOSO arms through the production stack on **both** cohorts.
Driver `scripts/levers/levers_test.py`; full writeup `scripts/levers/LEVERS_RESULTS.md`.
Standard cache reproduces the anchor to 4 dp; large cohort (497 pos, full 427-feat space
incl nk/micro) is the trusted read. Ran on pdmle, aggregate metrics only.

- 🔬 **Only ONE of the four converts: the *tuned* reward threshold.** On rank-norm feats the
  deployable **cross-site transfer** threshold hits **+0.1712** reward — essentially the oracle
  ceiling (+0.1765) — vs +0.0957 at π and **+0.0605** raw@π baseline. **≈2.8× the baseline
  deployable reward, no leakage.** rank-norm (normalization) + tuned threshold (decision) is the
  two-lever stack to carry if the container reopens. Pairs with [[rank-norm-verdict]].
- ⚠️ **q>pₐ decision rule (team 476) is NO-SHIP** — regresses on both cohorts (large +0.0712 <
  π +0.0957; standard +0.148 < +0.263). The Bayes-optimal form needs an *age-band-calibrated*
  posterior; ours is marginal. Verified the derivation against `compute_reward`: predict + iff
  q>p. Value is in *tuning* the threshold, not the q>pₐ rule.
- ⚠️ **Age-matched pairwise ranking loss (±2yr) is NO-SHIP** — linear pairwise (AC-AUROC 0.579)
  and linear pointwise (0.592) both trail the nonlinear HGB (0.606) on the very metric pairwise
  is meant to optimize. Model class dominates the loss function.
- ⚠️ **Drop amplitude/RMS (gain-invariance) is NO-SHIP** — ~neutral for discrimination
  (AC-AUROC +0.003) but **collapses reward-threshold transfer** (+0.171→+0.009). Per-site rank
  already neutralizes amplifier gain, so dropping µV-scale absolutes only removes signal.
- 🔬 **Event-feature ablation (476's claim): partially reproduced, marginal.** Dropping events
  *helps* discrimination on the large cohort (AC-AUROC +0.007, AUROC +0.005) but hurts reward
  transfer. Not 476's clean "AUROC↑/AC-AUROC↓" dissociation — dropping helps both. Optional micro-tweak.
- ⚠️ **Inverse-normal-transform normalization does NOT beat uniform-percentile rank** (added
  `--norm-only` arm). Ties on discrimination but reward-threshold transfer collapses (+0.072 vs
  rank +0.171) — the *bounded* [0,1] rank scale is what makes the threshold transfer across sites.
  Keep uniform-percentile rank ([[rank-norm-verdict]]).
- 🖼️ **Netphys PoC: added a combined pooled/non-CI/CI × 5-stage figure** (`render_combined_graphs`
  in `scripts/netphys/netphys_graphs.py`; `netphys_poc_graphs_combined.png`) tying the three TDS
  networks into one comparison; makes the CI EMG↔cortex decoupling readable against the pooled row.

---

## 2026-09-10 (Per-site rank-norm + reward-threshold prototype; TDS PoC report) 🔬

**Why.** Competitor review ([[competitor-methods-2026]]) flagged two untried, high-value
levers from the top teams: SIREN's (team 480, 6th) **per-site rank normalization** as a
batch-effect remover, and the 476/553 **reward-aware decision threshold**. Prototyped both
on the champion features through the exact production LOSO stack. `scripts/rank_norm/`
(`rank_norm_test.py` standard cache, `rank_norm_large.py` large cohort); ran on pdmle as
data account, aggregate metrics only. Full writeup: `scripts/rank_norm/RANK_NORM_RESULTS.md`.

- **Anchor reproduced to 4 decimals** — `moe_none` = AC-AUROC 0.6352 / AUROC 0.7117 /
  reward@π +0.2738 (matches champion). Plumbing trusted.
- 🔬 **MoE == pooled, bit-for-bit, on BOTH cohorts** — the site-MoE + Kaiser fine-tune head
  is **dead weight under LOSO** (no expert for a held-out site → global fallback). Corroborates
  [[model-architecture-findings]] "router dead / pooled≥MoE" independently.
- 🔬 **Per-site rank-norm = first batch-effect lever with a consistent positive LOSO lift.**
  On the well-powered **large cohort (497 pos)**: AC-AUROC **0.5898→0.6154 (+0.026)**, AUROC
  0.6549→0.6739 (+0.019), deployable reward +0.0882→+0.0985, reward ceiling +0.161→+0.198.
  It **beats per-site z-score** (0.6105) — z-score was previously rejected — and is
  transductive/label-free, so hidden-site-legal by construction. Validates SIREN on our data.
  (Standard 84-pos cohort *looked* like −0.03 AC-AUROC harm — that was small-n noise; the
  positive-count-limited wall again, cf. [[hdbscan-clustering-probe]].)
- 🔬 **Reward-threshold lever alone is a near-no-op** on RAW features (a naively transferred
  reward threshold is *worse* than π because the threshold is site-scale-dependent; oracle
  ceiling barely > π → confirms the global threshold was already ≈optimal). Its value is
  **conditional on rank-norm**: once features are rank-normalized the reward threshold
  transfers cleanly across sites (transfer ≈ oracle) and the oracle ceiling rises. The two
  metrics still pull opposite ways (AC-AUROC vs reward), per competitor lesson #4.
- **Status:** exploration only (deadline passed, container frozen). If the model reopens,
  rank-norm belongs as the normalization layer and is the right pairing for any future TDS block.

## 2026-09-10 (Network-physiology TDS proof-of-concept — report) 📊

**What.** Wrote `scripts/netphys/TDS_POC_REPORT.md` — an educational report on the
brain–organ Time-Delay-Stability network PoC (`scripts/netphys/`, run earlier on 18
recordings). Covers what Network Physiology / TDS is, why we built it (white space — no
competitor built brain↔organ graphs), the method, and results from
`exports/netphys_poc_{graphs,heatmaps,ci_contrast}.png`.

- ✅ **Validity check passed** — pipeline recovers the textbook stage reconfiguration
  (network densest in light sleep N1/N2 density 0.19–0.24, collapses in N3 to 0.08, REM
  intermediate) and physiologically sensible link structure (homologous cross-region
  same-band cortical coupling dominant: C·β–O·β 0.83–0.94; organ links an order weaker).
- 🔬 **Candidate CI signature (descriptive, n=18):** in N2, chin **EMG↔cortex coupling
  decouples** in CI (mean 0.198→0.086; C·σ–EMG −0.253, O·σ–EMG −0.248, C·β–EMG −0.219),
  with a mild *increase* in brain↔heart / homologous-θ coupling. Coherent and mechanistically
  plausible, but ⚠️ **not yet predictive evidence** — must clear LOSO + batch-effect + reward
  gates (pair with rank-norm above). Descriptive/publishable on its own regardless.

## 2026-09-04 (HDBSCAN unsupervised pattern probe on the 1103×436 matrix) 🔬

**Why.** Asked whether density clustering surfaces structure — by site, by CI — the champion
misses. Standardize → UMAP(10-D) → `sklearn.cluster.HDBSCAN`, then measure cluster alignment
with site vs CI. `scripts/eda/hdbscan_probe.py` (parameterized by `--cache` so it re-points at
PSG-feature / embedding matrices later); ran on pdmle as `arshia_ilaty_physio26`. Aggregate only.

- **Site is NOT the manifold driver** (corrected a prior worry about batch-effect dominance).
  Sites fully intermixed: silhouette(site, UMAP-10) ≈ **0.001**, cluster↔site **ARI ≈ 0** (−0.02),
  every cluster ~70–90 % S0001 (just mirrors S0001 = 78 % of cohort). A batch signal *exists* in
  the features (5-fold RF predicts site at **0.913** vs 0.777 majority) but doesn't shape the
  geometry — reassuring re: LOSO confounding.
- **CI is a diffuse gradient, not a cluster.** cluster↔CI **ARI ≈ 0**, NMI ≈ 0.02 — no discrete CI
  mode (expected at 7.6 %). But per-cluster prevalence spreads **1.7 % → 12.5 %** (mcs=40: c1 n=121
  CI 1.7 % / age 58 → c0 n=255 CI 12.5 % / age 65), and the enriched clusters are the **older**
  ones. The clustering carves one connected **age/severity continuum** (≈41 % noise) that CI rides
  along — i.e. it re-finds the `age` axis the champion already dominates.
- 🔬 **Verdict — no predictive help; useful as QC/exploration.** No CI cluster to exploit; the
  cluster gradient is redundant with `age`. Descriptively valuable: confirms a smooth age/severity
  manifold with sites well-mixed. Figure (`/tmp/sleeper_umap.png`, 3-panel UMAP) kept local, gitignored.
- 🔬 **Gate-confirmed NO-SHIP (Δ exactly 0).** Fed the 4 cluster indicators (mcs=40 one-hot) through
  the LOSO fusion gate (`coup_gate_ab.py` + `make_cluster_csv.py`): base+cluster is **bit-identical**
  to baseline (AC-AUROC 0.6352 / AUROC 0.7117 / reward +0.2738; Δ=+0.0000, zero-width CI). The champion
  never uses them — total redundancy with the 436 features. 12th neutral screen at the n_pos=84 wall.
- 🔬 **Same probe on the SleepFM stage embeddings (1080×2560) — batch-effect diagnosis.** The learned
  embeddings behave **oppositely** to the tabular features: site is **strongly** encoded (5-fold RF
  decodes site at **0.980** vs 0.775 baseline; silhouette(site, UMAP-10)=**0.111**; cluster↔site
  **ARI=0.29**, NMI=0.32 — HDBSCAN carves a 100%-I0006 cluster and pure-S0001 clusters, 0% noise).
  CI stays diffuse (cluster↔CI ARI≈0.03) though one small subcluster hits 19–25% CI (2.7–3.5× base),
  but it's age-elevated and site-mixed. **Mechanistic explanation for why embedding fusion was DEAD
  under LOSO** ([[embedding-fusion-verdict]]): the representation is organized primarily by
  site/montage/device, so a model trained on 2 sites keys off site structure that can't transfer to
  a held-out 3rd. Builder `build_emb_matrix.py`; figure `/tmp/emb_umap.png`.
- 🔬 **Confirmed on `embeddings_by_stage_pure` (1080×2560) — batch effect is intrinsic to SleepFM.**
  Site stays near-perfectly decodable (5-fold RF **0.961** vs 0.775 base), so the batch signal is not
  an artifact of the stage variant. Geometry is a touch *less* blob-separated than the stage set:
  silhouette(site, UMAP-10)=**−0.040** (vs +0.111), cluster↔site **ARI=0.16** (vs 0.29) — I0002 sits
  *inside* the S0001 blob and the top-right islands mix S0001+I0006, yet HDBSCAN still recovers a
  99%-I0006 cluster and 100%-S0001 clusters (7 clusters, 0.2% noise). Takeaway: site info persists but
  is carried more nonlinearly/distributed here (high RF ≠ clean UMAP separation). CI still diffuse
  (cluster↔CI ARI=0.016); one cluster (c3, n=61) reaches **21% CI** at elevated age 63.4 — the same
  age-linked gradient, not a clean CI mode. Figure `/tmp/emb_pure_umap.png`. Both embedding variants
  → same verdict: strong site decodability = LOSO batch-leakage → fusion DEAD.

---

## 2026-09-03 (Tool scouting: colleague feature-selection toolkit + network-physiology repo) 🔬

**Why.** Two outside tools worth checking against our 436-feature champion — a colleague's
generic `FeatureSelection` battery (`/home/arshia/featureselection`, never run on our data), and
the manager-mentioned `causal_networks_physiology` (directed cross-system coupling). Both are
**scouting**: neither changes the champion; anything they motivate still clears the LOSO fusion gate.
Full write-up: `scripts/featsel/README.md`.

- **Feature-selection toolkit — ran on the 1103×436 matrix (pdmle; y+=84, 7.6 %).** Drivers
  `scripts/featsel/{run_featsel_physionet.py,featsel_finish.py}`: load the cache, drop
  identifiers/leakage, median-impute, run the battery with per-method guards, write aggregate
  rankings (gitignored, aggregate-only). **11 methods** produced a consensus: ANOVA-F, MIC, RF
  impurity, XGB split-count & coverage, RF permutation, Boruta, SHAP (TreeExplainer substitute),
  LGBM gain & split, mRMR. Null-importance columns computed but excluded from consensus.
  - **Four fixes needed** (2-yr-old code vs current libs): module-level `from minepy import MINE`
    made a missing optional dep fatal to the *whole* import (installed minepy + boruta/mrmr/xgboost/
    shap); LightGBM 4.x dropped the `categorical_feature=` kwarg from `lgb.train` (recomputed via
    `booster.feature_importance`); `mrmr_classif` now returns <K features, breaking the length-K
    frame (mapped returned order→rank); SHAP `KernelExplainer` intractable at 436 feats (→ RF
    `TreeExplainer`).
  - 🔬 **Result — validates, doesn't extend.** Consensus top-20 re-derives our own
    `stats_significance.py` physiology with no new predictor: **EEG theta slowing** dominates
    (N1/N2/wake theta power + theta/alpha), plus **fragmentation** (`arch__trans_p_wake_n1` is #1;
    N1/REM arousal indices), **REM microstructure** (RSWA phasic/min), and **age/bmi**. `age`/`bmi`
    sit at median-rank 2 but mean-rank ~30–37 (tree split-count methods under-rank collinear
    demographics — mean-rank consensus is a heuristic). **Boruta confirms only 5 of 436 features**
    (8 incl. tentative) as reliably relevant — the same **n_pos=84 / 3-site** wall behind the 10
    JEPA NO-SHIPs, restated by an independent tool. Caveat: whole-cohort, no site holdout →
    prioritisation only, not LOSO lift.
- **`causal_networks_physiology` (Moritz Günther) — assessed, not ported.** Network-Physiology
  research code: **directed cross-system coupling** (breath↔heart↔EEG-alpha at 1 Hz, per sleep
  stage) via **Granger G-causality** (statsmodels VAR + ADF stationarity) and **MPRSA** (multivariate
  phase-rectified signal averaging); validated on aging/OSA-vs-young, no `requirements.txt`.
  - **Verdict — worth a small time-boxed experiment; do NOT vendor.** It hits the one axis our
    EEG-dominated champion under-exploits (directed autonomic↔cortical↔respiratory coupling), and
    unlike multimodal JEPA (NO-SHIP #8, diluted EEG) these are **low-dim interpretable scalars**, a
    different failure mode. Caveats: targets aging/OSA not CI; it's a **port** (re-implement the
    estimators against our pipeline), not a plug-in; the n_pos=84/LOSO wall still gates it.
    Suggested minimal probe: breath↔HR + HR↔EEG-band G-causality + one PRSA coupling scalar per
    NREM/REM → run through the existing fusion gate; neutral closes it.
  - 🔬 **Ran the probe — NO-SHIP (measured).** Ported the Granger-G core to `scripts/netphys/`
    (`coupling.py` estimator + `signals.py` 1-Hz breath/heart/EEG-alpha reconstruction +
    `export_coupling_features.py`; self-test recovers an injected breath→heart coupling). Exported
    **14 directed-coupling scalars** (6 ordered pairs × NREM/REM + a stationarity-coverage `statfrac`
    per group) for **1066/1103** recordings on pdmle (8 shards; 4-way after 8-way OOM-killed workers
    on the 30G box). Ran `coup_gate_ab.py` (LOSO, pooled fitter, official metrics, 2000-boot paired
    on aligned OOF) both raw and with a per-fold train-only univariate AUROC≥0.6 filter:
    | arm | AC-AUROC | AUROC | reward |
    | --- | --- | --- | --- |
    | baseline (champion) | 0.6352 | 0.7117 | +0.2738 |
    | + coupling (raw) | 0.6154 | 0.6886 | +0.2841 |
    | + coupling (filtered) | 0.6101 | 0.6909 | +0.2919 |
    Paired Δ AC-AUROC = **−0.0209** [−0.0544,+0.0131] raw / **−0.0259** [−0.0536,+0.0009] filtered;
    reward Δ +0.009/+0.019 but CIs straddle 0. **Primary metric degrades in both arms → COUP_GATE_NOSHIP.**
    The signal is real but redundant: strongest single coupling is **NREM EEG-alpha→heart-rate**
    (whole-cohort AUROC 0.637, |d| 0.31), then REM cardiac↔EEG (~0.60–0.62); breath couplings are
    near-chance — all collinear with the EEG-spectral + HRV features the champion already has. Adding
    interpretable low-dim features does **not** escape the n_pos=84/LOSO wall (11th neutral screen).
    Full write-up: `scripts/netphys/README.md`.

---

## 2026-08-28 (End-to-end supervised fine-tuning of the V2 EEG encoder) ⚠️

**Why.** All 9 prior JEPA runs used the encoder as a **frozen** feature extractor (pool CLS
descriptors → night vector → GBM) and hit the same wall: the SSL rep is redundant with the
handcrafted-EEG champion under LOSO / n_pos=84. This drops that assumption — the last untested
JEPA lever. Init the encoder from the label-free V2 pretrain and **fine-tune it end-to-end** with
the CI label through a stage-aware attentive-pooling (MIL) night head. Different failure mode; the
real risk is overfitting 84 positives across 3 sites, so heavy regularization: freeze the low-level
patch conv + chan-emb, low encoder LR (3e-5) / high head LR (5e-4), dropout 0.4, WD 0.05,
class-balanced batches (half pos / half neg), bag-resampled K=48 epochs/night as augmentation.

- **Model** — `scripts/jepa/finetune_v2.py`: V2 backbone (teacher init) → per-epoch CLS descriptor
  → Ilse-et-al. gated-attention pool over a night's epochs (+ stage embedding) → linear head.
  Produces **out-of-fold** probabilities (each night scored by a model that never trained on it)
  for both **LOSO** and **LOPO** (5-fold StratifiedGroupKFold by pid); model reinit per fold from
  the teacher (no cross-fold leak). `scripts/jepa/ft_eval.py`: three arms on the SAME folds/rows as
  the frozen screens — champion (436 feats → GBM), jepa-ft (ft OOF prob used directly), fusion
  ([436 feats, ft OOF prob] → GBM; the ft prob is a valid OOF feature) — same fitter / AC-AUROC /
  reward / paired-bootstrap protocol, so the verdict is directly comparable to all 9 prior screens.
  Training loss descended cleanly every fold (≈0.70→0.60); the encoder-init pretrain saw all nights
  label-free, so both headlines carry the same mild shared-pretrain optimism as V2.
- ⚠️ **RESULT — NO-SHIP #10** (but the *least-bad* JEPA result to date). On the 1082-night EEG-packed
  subset (y+=83); the champion anchor is lower here than the 0.6352 headline because the subset
  excludes 21 non-EEG-packed recordings — all arms are on identical rows/folds (fair paired test).
  - **LOSO:** champion AC-AUROC **0.5682** / AUROC 0.6670 / reward +0.1281; jepa-ft (alone) 0.5507 /
    0.6196 / +0.0520; fusion 0.5712 / 0.6692 / +0.1325. Δ[fusion−champion] AC-AUROC **+0.0031
    [−0.0077, +0.0135]**, reward +0.0047 [−0.0153, +0.0268]; Δ[jepa-ft−champion] AC-AUROC −0.0176
    [−0.0998, +0.0622].
  - **LOPO:** champion AC-AUROC **0.6389** / AUROC 0.7666 / reward +0.1777; jepa-ft (alone) 0.6070 /
    0.6761 / +0.1677; fusion 0.6383 / 0.7568 / +0.0606. Δ[fusion−champion] AC-AUROC **−0.0007
    [−0.0275, +0.0263]** (dead neutral), reward −0.1172 [−0.3013, +0.0104]; Δ[jepa-ft−champion]
    AC-AUROC −0.0307 [−0.1117, +0.0499].
- **What's new vs the 9 frozen screens.** For the first time fusion does **not hurt** the champion:
  LOSO Δ AC-AUROC is *positive* (+0.0031) and LOPO is dead neutral (−0.0007) — every frozen screen
  landed −0.03 to −0.04. End-to-end fine-tuning stopped the rep from degrading the champion, but it
  still adds **no significant lift** on the primary threshold-free metric under either CV (both CIs
  span 0). jepa-ft alone underperforms the champion (expected: one balanced-trained sigmoid vs the
  full 436-feature GBM). The LOPO fusion **reward** drop (−0.1172) is a thresholding artifact of the
  class-balanced ft sigmoid reshaping the GBM's ranking near the prevalence cut — AC-AUROC, the
  challenge-primary metric, is the one that's neutral.
- **Verdict.** Dropping the frozen-embedding assumption is the last distinct JEPA failure mode, and
  it produces neutral-at-best fusion. **10 consecutive NO-SHIPs across every axis** (per-epoch EEG
  V1/V2, multimodal, night-dynamics, end-to-end fine-tune) → the wall is n_pos=84 / 3-site LOSO, not
  the architecture or the frozen-vs-tuned choice. JEPA exploration is exhausted; closing it.
  Artifacts (box): `ftprob_site.npz`, `ftprob_patient.npz`, `ft_train_{site,patient}.log`,
  `ft_eval_{site,patient}.log`.

---

## 2026-08-28 (Level-2 hierarchical night-JEPA + Exp-C stage-conditioned prediction-error) ⚠️

**Why.** All 8 prior JEPA runs pooled *per-epoch* reps into a night vector and hit the same
wall: the SSL rep is redundant with (or worse than) the handcrafted-EEG champion under LOSO /
n_pos=84. This attacks the **one untested axis — night *dynamics*** (30s → 5-min → whole-night),
which is exactly where our error analysis says the CI signal lives (blunted cross-stage dynamics).
Built on the *one* representation that carried real signal: the V2 EEG per-epoch CLS descriptor
(jepa-only AUROC 0.629), not the diluted multimodal one. Gated under **LOSO** (comparable to all
prior) **and LOPO** (5-fold StratifiedGroupKFold by pid — the user's ask for tighter CIs).

- **Model** — `scripts/jepa/night_jepa.py`: (L1) reuse the frozen V2 EEG teacher to encode every
  packed epoch → `l1_epemb` (974,482×128); (L2) pool 10 epochs = one 5-min block + majority stage
  → per-night sequence of block tokens (median 91 blocks/night); a small night-transformer
  (d128/depth4) + **temporal I-JEPA** — mask a contiguous span of blocks, EMA teacher, narrow
  cross-attn predictor fills masked block latents from mask-token + block-pos/stage embeddings,
  smooth-L1 on EMA-teacher LN reps. Label-free single all-nights pretrain (60 ep, 51 s on T4,
  loss 0.33→0.008 — night block-sequences are far more predictable than raw epochs).
- **Exp-C (the actual new hypothesis)** — at embed time, mask each block one-at-a-time and predict
  it from the rest → a per-block *surprise*; aggregate glob {mean,max,cv} + per-stage mean (5) +
  cross-stage modulation → stage-conditioned prediction-error features. Hypothesis: CI = blunted /
  disorganized stage-conditioned surprise.
- **Gate change** — added `--cv patient` (LOPO-style StratifiedGroupKFold by pid) to
  `jepa_gate_ab.py`; default `--cv site` (LOSO) unchanged, so all 8 prior verdicts stay comparable.
- ⚠️ **RESULT — NO-SHIP #9.** champion (LOSO) AC-AUROC **0.6352** / AUROC 0.7117 / reward +0.2738.
  - **LOSO:** jepa-only AC-AUROC 0.5757 / **AUROC 0.6323** / reward +0.0209 (real label-free signal,
    ≈ V2's 0.629); fusion 0.5994 / 0.7003 / +0.2115. Δ[fusion−champion] AC-AUROC **−0.0361
    [−0.0893, +0.0126]** (spans 0, leans neg); Δ[jepa-only−champion] −0.0579 [−0.1301, +0.0116].
  - **LOPO** (tighter CIs, as asked): champion AC-AUROC 0.6249 / AUROC 0.7610 / reward +0.2195;
    fusion 0.5912 / 0.7338 / +0.2899. Δ[fusion−champion] AC-AUROC **−0.0344 [−0.0810, +0.0111]**
    (tighter than LOSO but lower bound still negative); reward Δ +0.0719 [−0.0769, +0.2598] (spans 0);
    Δ[jepa-only−champion] AC-AUROC −0.0906 [−0.1647, −0.0186] (sig worse).
  - **Exp-C features carry only weak signal** — best `expc_glob_mean` whole-cohort (optimistic, no CV)
    AUROC 0.594, below the in-fold 0.6 filter; per-stage means 0.52–0.57. The "blunted stage-conditioned
    surprise" hypothesis gets weak-at-best support and does not survive CV.
- **Verdict.** Night-dynamics is the last untested JEPA axis, and it reproduces the exact pattern of
  the other 8: a genuine label-free rep that adds **nothing** over handcrafted EEG on the primary
  AC-AUROC under either CV. **9 consecutive NO-SHIPs** → the wall is n_pos=84 / 3-site LOSO, not the
  architecture. Recommend closing the JEPA exploration. Artifacts (box): `encoder_l2_all.pt`,
  `emb_l2_all.npz` (1082×905 = 128×7 pools + 9 Exp-C), `l1_epemb.npy`, `l2_blocks.npz`,
  `l2_gate_{loso,lopo}.log`.

---

## 2026-08-27 (Multimodal PSG-JEPA — 15-channel extractor + presence-aware I-JEPA) 🔬📌

**Direction:** challenge deadline passed → pure exploration, NO shippable/frozen-container
constraint. Eval is **LOSO** (leave-one-site-out) primary, compared against the champion
handcrafted model (LOSO **AC-AUROC 0.6352 / AUROC 0.7117 / reward +0.2738**). GPU authorized
and fixed: box has a Tesla T4 16 GB; the user-site torch was cu130 (too new → CUDA unavailable),
so an isolated venv `exports/jepa/gpuenv` runs `torch==2.5.1` (cu124 ≤ the T4's CUDA 12.5).

**Why multimodal.** The raw-EEG JEPA (V2, below) learned a *genuine* self-supervised
representation but it was **redundant** with the 436 handcrafted EEG features — no LOSO lift.
The one remaining orthogonal-signal lever is the non-EEG PSG the champion under-exploits:
ECG/HRV, respiration, SpO2, EOG/EMG. This experiment feeds the validated V2 recipe the full
15-channel montage so cross-modal structure (apnea→desaturation, arousal↔HRV, REM EOG/EMG
atonia) can surface as signal handcrafted per-signal features flatten out.

- **Extractor** — `scripts/jepa/export_raw_multimodal.py`: 15 canonical slots (6 EEG
  F3/F4/C3/C4/O1/O2 + ECG + 2 EOG + chin EMG + airflow + chest/abd effort + SpO2 + leg EMG),
  role-gated by `app.channel_role` with cue disambiguation; CAR over present EEG only,
  per-channel robust z-score (±8, also fixes the 854 `uV`-mislabeled SpO2), require ≥5/6 EEG.
- **Cross-site canonicalization VERIFIED** 🔬 — the montage channel *names* differ by site
  (Kaiser unreferenced). Full pack: **974,482 epochs / 1082 nights**, sites S0001 845 /
  I0006 190 / I0002 47, 83 positive nights; **every one of the 15 slots present ≥99.4%**
  across all three sites; sample epoch shows real per-slot variance (not zero-fill).
- **Model** — `scripts/jepa/raw_mm_jepa.py`: V2 extended to N_CH=15 (450 tokens/epoch) with
  **presence-aware I-JEPA** — absent channels are zero-filled but excluded everywhere (padded
  out of encoder attention via `src_key_padding_mask`, never sampled as I-JEPA context/target,
  dropped from the CLS readout); `night_chan_mask` stored in the pack index, looked up per epoch.
  Same emb.npz schema + LOSO gate (`jepa_gate_ab.py`) as V1/V2 for a directly comparable verdict.
- ⚠️ **RESULT — NO-SHIP #8, WORSE than EEG-only.** champion AC-AUROC 0.6352 / AUROC 0.7117;
  multimodal jepa-only 0.5113 / **0.4997 (literal chance)**; fusion 0.6003 / 0.6838.
  Δ[fusion−champion] AC-AUROC −0.0349 [−0.0864,+0.0189] (leans neg, spans 0);
  Δ[jepa-only−champion] −0.1242 [−0.2370, **−0.0084**] (significantly worse). Pretrain ran
  ~7 h on the T4 (loss plateaued ~0.131). **Adding the 9 non-EEG channels *diluted* the
  representation** — EEG-only V2 jepa-only was AUROC 0.629 (real signal), multimodal fell to
  0.4997 (chance). Diagnosis: the single CLS readout over 15 heterogeneous modalities washes
  out the EEG structure (where all 8 experiments locate the CI signal); SpO₂ is near-flat within
  a 30 s epoch, effort/EMG noisy. The cross-modal-coupling hypothesis (apnea↔desat, arousal↔HRV
  the champion under-exploits) is **rejected at chance level**. Consistent finding across all 8
  JEPA NO-SHIPs: the wall is the data (83 positives / 3-site LOSO), not the representation.
  Cheap unrun diagnostic: re-pool the trained encoder's per-token reps by modality group
  (embed-only, ~15 min) to separate readout-dilution from no-signal — but a rescue's ceiling is
  "neutral fusion" since the EEG group is redundant (V2). Remaining lever = Level-2 temporal/
  hierarchical night-JEPA + Exp-C stage-conditioned prediction-error, ideally under LOPO.

## 2026-08-26 (Raw-signal JEPA on GPU — V1 chance, borrowed fixes → V2 genuine but redundant) 🔬⚠️

Escalation from a Stage-1 light TS-JEPA screen (data2vec over per-epoch EEG *summary* features:
jepa-only AC-AUROC 0.514, fusion 0.6046 — encoder compressed away signal, NO-SHIP) to raw
multichannel EEG on the GPU. Pipeline: `scripts/jepa/export_raw_eeg.py` decodes 6 scalp
derivations → CAR → 64 Hz → robust z-score → stage-aligned 30 s epochs → per-recording npz;
`raw_eeg_jepa.py`/`_v2.py` pack → memmap, pretrain (label-free, all nights), embed
(stage-aware night pooling, 768-d), gate under LOSO. Encoder is label-free so a single
all-nights pretrain doesn't leak the target; LOSO is enforced in the GBM head.

- **V1 (data2vec-style): NO-SHIP.** jepa-only AUROC **0.5030 (literal chance)**; fusion 0.6734;
  Δ[fusion−champion] AC-AUROC −0.039 [−0.098,+0.024] (spans 0). Root causes: channel-mixed
  `Conv1d(6→d)` erased topography, in-place masking too easy, mean-of-LayerNorm'd tokens washed
  out the night vector.
- **Borrowed-repos escalation** (user unblocked `git clone`). Cloned ECG-JEPA (kweimann),
  eeg-vjepa (amir-hojjati), PhysioJEPA (benmfox), signaljepa (eugenehp) — code only (HF weights
  blocked). Three convergent fixes folded onto V1: (1) per-channel (C×T) patch tokens + learned
  channel-embeddings (topography); (2) **I-JEPA masking** — context encoder sees only kept tokens,
  narrow predictor (dim/2) fills masked targets from mask-token + target pos-emb, smooth-L1 on
  EMA-teacher LayerNorm'd latents; (3) CLS/register-token readout. AdamW betas (0.9,0.99),
  WD-exclude bias/norm, EMA momentum 0.998→0.9999.
- **V2 (`raw_eeg_jepa_v2.py`): NO-SHIP #7, but METHOD VALIDATED.** champion AC-AUROC 0.6352 /
  AUROC 0.7117; jepa-only 0.5963 / **0.6294**; fusion 0.6253 / 0.6801. Δ[fusion−champion]
  AC-AUROC **−0.0113 [−0.0870,+0.0680]** (spans 0, ~neutral); Δ[jepa-only−champion] −0.0380
  [−0.1389,+0.0638]. The borrowed recipe **worked at the representation level** — jepa-only AUROC
  went 0.5030 (chance) → 0.6294 (real label-free SSL signal), and fusion went from *hurting* the
  champion (V1 −0.039) to ~neutral. But the V2 embedding is **redundant** with the 436 handcrafted
  EEG features → no LOSO lift. ⚠️ Ceiling is the data: **83 positive nights across 3 sites** under
  LOSO — the same wall as the 7 prior NO-SHIPs. Conclusion: single-modality raw-EEG JEPA is genuine
  but adds nothing over handcrafted EEG; next lever = multimodal (2026-08-27 entry).

## 2026-08-20 (TRACK 2 — full-feature submission built: signal block ported into the container) ✅🔬

**Two-submission strategy.** Track 1 (repo root, `submit` preset, 99 features) is the LOSO-tuned
cross-site entry — done, committed, audited. Track 2 (`track2/`, new `submit_full` preset, 336
signal features on top of the T1 core) is the **within-distribution champion**, hedging the
same-3-sites-vs-new-sites unknown: if the hidden test set is held-out patients from the same 3
sites, the full feature set wins big (438-feat 80/20 AC-AUROC ~0.83 vs LOSO ~0.635). Two dockers,
one per hypothesis.

**The port.** The nk/report/micro feature families lived ONLY as offline cache-builder scripts in
`scripts/viewer/` that read raw EDF on the box and wrote CSVs — never wired into the frozen
container. Track 2 ports them to run at INFERENCE. New `track2/sig_features/` package vendors the 8
compute modules **verbatim** (only cross-imports made package-relative — verified byte-identical to
the viewer originals otherwise) + the two helpers their exporters relied on from the viewer's
`app.py`/`preprocess.py`: `channel_role` (montage→role) and `smooth_stages` (CAISR-staging
preprocessing). `sig_features.signal_feature_row()` reproduces the three exporters' exact per-record
recipe (channel picking → per-stage functions → fixed-order flatten) in one physio pass → 336
floats (183 nk + 122 report + 31 micro), NaN on any miss, never raises. Wired into `team_code.py`
via `extract_signal_features` + `include_signal` preset flag; physio EDF loaded ONCE (shared with
autonomic block). Single pooled/site-MoE head — `KAISER_ALT_PRESET=None` for T2 (the 42-feat alt is
T1-specific, would clash with the full vector). **Stage-code convention verified identical**
(5=W,4=REM,3=N1,2=N2,1=N3) between team_code and the viewer modules → per-stage features don't
scramble.

**Deps.** `neurokit2==0.2.10` + hard deps `matplotlib==3.9.2`/`requests==2.34.2` — exact versions a
teammate's submission proved install on `python:3.10.1-buster`. Container uses numpy 2.0.2, so the
`np.trapezoid` gotcha (frequency-HRV blanking on numpy<2) is MOOT. Dockerfile byte-identical to T1
(DO-NOT-EDIT lines intact); `.dockerignore` doesn't exclude `sig_features/`.

**Verification (on box, 20 real records):** value-parity (`track2/verify_sig_parity.py`) vs the
validated offline `*_features_standard.csv` — box neurokit is 0.2.13 (matches how the CSVs were
built) so it's a clean recipe/wiring check. **`nk__` (183 feats) and `report__` (122 feats) are
BYTE-PERFECT** — 0 NaN-pattern disagreements, 0 value mismatches, max rel-diff 5.9e-08 (float32
rounding). **`micro__` (31): RSWA + CAP byte-perfect; only the SO-spindle `couple_*` family drifts
~1-5%** (80 cells >2%). Root cause is benign + explained: coupling is the ONLY feature that runs a
full-signal Hilbert transform (slow-osc phase at each spindle peak), which is length-sensitive, and
the offline export decoded with `edfio ... lazy_load_data=True` while the container's
`helper_code.load_signal_data` uses `lazy_load_data=False` → a ULP/1-sample length delta shifts
every phase slightly (the 510× "max rel-diff" is a `preferred_phase_sin` sign-flip near zero, abs
diff ~0.29). PROOF it's not a wiring bug: report's per-epoch spindle detection matched byte-perfect
(same detector, reproducible), and CAP/RSWA (envelope/RMS, length-robust) matched perfectly.
**Train and inference share the same in-container path → the model is self-consistent; the drift is
only vs the offline CSV, in the lowest-ROI micro sub-family. PASS.** Frozen-harness train on the
60-record `submit_test_t2` slice trained + saved a valid `model.sav` at **435 features**
(demo+autonomic+caisr+arch+336 signal) — the wiring end-to-end works. `site_models=[]` on this slice
is expected small-slice fallback (Track 1 shows the same on 60 recs; per-site MoE fits on the real
~1090). **BUG FOUND + FIXED via the harness:** the first train logged 15× `channel_table.csv not
found` — I'd scaffolded track2 without copying `channel_table.csv` (the montage-rename table the
autonomic block needs; `DEFAULT_CSV_PATH = <team_code dir>/channel_table.csv`), so autonomic HRV/SpO2
silently degraded. Copied it into `track2/` (byte-identical, not dockerignored) and retrained clean (0 errors).
**Frozen-harness END-TO-END PASS:** retrain → `run_model.py` on the 60-record holdout slice scored
all 60 records, all 3 sites (I0002/I0006/S0001, 20 each), **0 exact-0.0 probs → no Kaiser padding
regression**, ~15s/record (well within budget). Every prob = 0.400 (base rate) — a tiny-slice
calibration artifact, NOT a bug: Track 1 produces the identical flat 0.400 on this same slice (the
real ~1090-record set gives the proper spread). Local wiring smoke test (synthetic signals,
scipy-only) confirmed report/micro compute. **Track 2 is submission-ready.** See
[[challenge-2026-approach]], [[track2-signal-port]].

## 2026-08-20 (ARCH BLOCK PORTED into submission + harmonization tested/rejected) ✅🔬

Two levers explored to lift results, decided by a fixed-cohort feature ablation (the large-cohort
0.87-AUROC win was mostly the 6× positives, NOT the features — standard-cohort 438-feat 80/20
AC-AUROC 0.618 ≈ ~55-feat LOSO 0.635, so at fixed cohort the extra features barely move the
*within-distribution* metric). The honest gate is: **does a block add signal OVER the shipped
autonomic+CAISR core on cross-site LOSO?** Ran `scripts/ablation_on_prodcore.py` on the plus cache
(core already includes whole-night autonomic HRV/SpO2 — the earlier CSV ablation lacked it, so its
arch/micro gains were partly just *recovering* autonomic). Verdict vs prod_core (LOSO AC-AUROC):
`+arch +0.034` (reward@pi +0.145→+0.233), `+micro −0.003` (recovers autonomic, dilutes arch),
`+nk −0.022` (hurts, needs neurokit2), `+rep +0.055`, `ALL +0.062`. micro helped 80/20 (+0.036)
but hurt LOSO — overfits-to-distribution; for a cross-site challenge, LOSO is trusted.

**Ported the arch block only** (surgical, low-risk): vendored a self-contained `arch_features.py`
at repo root (pure numpy; inlines the 3 `dynamics.py` symbols; CAISR-annotation-only, NO neurokit2,
NO waveform decode; NOT dockerignored → ships in the container). Added `extract_arch_features` to
`team_code.py`, captured `arousal_caisr` sampling rate from the fs-dict the caisr load already
returns (was discarded), gated on a new `include_arch` preset flag, added it to `submit`. Kaiser
alt head extracts with `caisr_autonomic` (no arch) so it stays 42 features — the 0.0 fix untouched.
`submit` now **54 → 99 features**. rep NOT ported: 122 features for a similar LOSO delta but with
container-dependency cost and no in-memory-annotation shortcut — a reasonable fast-follow, not a
deadline-day change.

**Verified two ways.** (1) **Value-parity** (`scripts/verify_arch_parity.py`): team_code's arch
extraction on 40 real records × 45 features = **0 mismatches** vs the validated `arch_features_
standard.csv` block (max abs diff 1.46e-05, float32 rounding) → the ablation's measured lift
transfers to the submission unchanged, no 9h waveform LOSO rerun needed. (2) **Frozen harness**
end-to-end (`train_model.py`→`run_model.py`, 60-record subset): trains clean at 99 features, all 60
records score across 3 sites, **I0006 scores normally (0.40, no 0.0 regression)**, no errors,
infer well within time limit, valid output CSV (`Cognitive_Impairment` bool + `_Probability`),
model.sav 181KB. (Uniform 0.40 = isotonic calibrator saturating on the degenerate balanced subset;
spreads out on the real cohort.)

**Harmonization tested and REJECTED** (`scripts/harmonize_loso.py`): per-site robust z-score
(median/IQR) to attack the flat LOSO transfer. baseline pooled AC-AUROC 0.635 (matches anchor).
`adaptive` (each site scaled by its OWN stats incl. held-out — non-deployable ceiling): pooled
−0.022 but lifts minority sites (Emory +0.05, Kaiser +0.08). `deployable` (train-site stats;
unseen held-out site → global fallback): pooled −0.039, and HURTS Kaiser (−0.048, reward
+0.412→+0.117) exactly because the held-out site gets mismatched global stats. Pooled is 78% BIDMC,
which harmonization slightly hurts. No robust shippable win → NOT adopted. ⚠️ The minority-site
ceiling lift suggests real transferable location/scale structure, but we can't exploit it per-record
at inference. 📌 Possible follow-up: harmonize only when the test site IS in training (multi-site
holdouts), or learn a site-invariant representation — out of scope for deadline day.

---

## 2026-08-20 (Random 80/20 within-distribution eval — LOSO counter-check) 🔬✅

User concern: is LOSO unfairly pessimistic vs an in-distribution test set? Ran `scripts/
random_split_eval.py` (50 seeds; train .68/val .12/**test .20** — val carved from the 80% dev so
threshold-free primary metrics are a clean 80/20 while reward thresholds still get a val slice).
STANDARD (n=1090): AC-AUROC **0.618±0.073**, AUROC 0.672, AUPRC 0.200 — lands right on the LOSO
anchor (0.635), so LOSO is NOT being pessimistic at fixed cohort. LARGE (n=6530): AC-AUROC
**0.828±0.023**, AUROC 0.864, AUPRC 0.418 — the +0.21 jump is the 6× positives (same 438 features),
with ~2× variance collapse. Confirms: positives fix within-distribution; features (esp. arch) are
what move cross-site.

---

## 2026-08-20 (SUBMISSION BUG FIXED — Kaiser head silently scored 0.0 for every I0006 record) 🐛✅

End-to-end submission-harness verification (`train_model.py` → `run_model.py`, the frozen scripts
the organizers run) on a balanced 60-record subset (3 sites, real EDFs) caught a **real shipping
bug** in `team_code.predict_from_features`. Symptom: every Kaiser (I0006) recording scored exactly
`prob=0.0`; other sites fine. Root cause = an `if/elif` width-padding bug:
```
infer_n = kaiser_alt_n_features if use_alt else n_features   # 42 for Kaiser (alt head)
if infer_n and feats.size != infer_n:      # 42==42 → skip (no pad needed)
elif n_features and feats.size != n_features:  # 42!=54 → FIRES → re-pads to 54!
```
When a Kaiser record produced its normal 42 alt-features, the `if` correctly skipped, then the
`elif` re-padded the vector to 54 → the 42-feature Kaiser `HistGradientBoostingClassifier` raised
`ValueError: X has 54 features, expecting 42` → swallowed by the try/except → `prob=0.0`. **Fix:**
collapse to one block targeting `target_n = infer_n or n_features` (the width the chosen model
actually expects). Verified post-fix: I0006 scores normally (0.40 on the degenerate subset, same
as other sites; spreads out on the real cohort). **Why never caught before:** all LOSO/eval scripts
call `predict_with_kaiser_override` on the full matrix — they never exercise the single-record
`predict_from_features` path; only the real harness does. Impact if unfixed: every I0006 test
recording scores 0.0 (1138/6530 in large cohort); doesn't fire on new-site-only test sets
(use_alt=False) but silently broken regardless. ✅ **Current submission now verified train→save→
load→predict→valid-output-table end-to-end.** Model.sav 140KB; output CSV has `Cognitive_Impairment`
bool + `Cognitive_Impairment_Probability`. Aggregate stats only left the box.

---

## 2026-08-20 (Large-cohort full-feature training eval — 6530 recordings / 497 positives) 🔬✅⚠️

Deadline-day payoff test: ran the FULL feature stack (all 5 families + demographics) through
the exact production model on the large cohort, standard-vs-large on identical footing.
`scripts/large_cohort_eval.py` + `run_local_cv.{loso,tvt}`. Built from CSVs (the 436-feat .npz
needs a ~9 h raw-EDF waveform pass, infeasible today) — extends the Phase-1 CSV-assembly to ALL
FIVE families + one-hot demographics (matching `team_code.extract_demographic_features`), and
builds STANDARD THE SAME WAY as a calibration anchor. Micro shards merged →
`micro_features_large.csv` (6530×40). Both cohorts aligned to **438 common features, 0 dropped**
(perfect schema parity). Standard reproduced its known anchor (LOSO AC-AUROC 0.601 / AUROC 0.656),
so the pipeline is verified faithful and the large numbers are trustworthy on the same footing.

**Within-distribution (70/15/15, 30-seed mean±SD) — DECISIVE WIN:**
| metric | standard (84 pos) | large (497 pos) |
|---|---|---|
| AUROC | 0.676 ± 0.052 | **0.869 ± 0.021** |
| AC-AUROC | 0.632 ± 0.077 | **0.832 ± 0.027** |
| reward | +0.288 ± 0.594 | **+0.474 ± 0.103** |
| reward@pi | +0.244 ± 0.401 | **+0.553 ± 0.133** |
| AUPRC | 0.179 ± 0.059 | **0.417 ± 0.049** |
AUPRC more than doubles (0.18→0.42) and reward SD collapses ~6× (0.594→0.103) — the model goes
from wildly unstable at 84 positives to reliable at 497. Exactly what the learning curve predicted
(positive-count-limited, not saturated). **All 438 features — CAISR arch, per-stage NK HRV,
report spectral/spindle/oxygenation, and the advanced batch (C1 nonlinear HRV, F1 transition
Markov, SO–spindle coupling, RSWA, CAP) — carried onto 6530 recordings with zero column drift.**

**Cross-site (LOSO — honest new-site proxy) — FLAT** ⚠️: AUROC 0.656→0.658, AC-AUROC 0.601→0.592,
reward@pi +0.154→+0.057. Per fold: Emory reward +1.24 (small n, prevalence artifact), Kaiser
+0.205, **BIDMC (n=5075) reward +0.025 / AC-AUROC 0.565** — the big fold dominates pooled and drags
reward@pi down (a threshold/prevalence artifact on the huge fold; ranking AUROC is fine). More
PATIENTS from the same 3 sites does NOT fix new-site transfer — that needs more SITES (external
cohorts SHHS/MrOS/MESA), the known domain-generalization wall. **So the large win is real & large
IF the hidden test set is held-out patients from the same 3 sites; if it's NEW sites, transfer is
still the ceiling (~0.66 AUROC).** Test-set composition remains the decisive strategic unknown.
`large_cohort_eval_summary.json` on box. Aggregate stats only left the box.

---

## 2026-08-19 (End-to-end fused SleepFM+GNN → UMAP pipeline — the "new submission" recipe) 🔬⚠️

Ran the pipeline the request described in full: fuse handcrafted(436) + SleepFM-pure(2560-d)
+ PhysioGraph GNN(16-d), apply UMAP on the fused representation, feature-select, train the
production stack, prepare a submission. Built as an HONEST LOSO experiment FIRST (`scripts/
eda/fuse_umap_e2e.py`) because embeddings **cannot be regenerated inside the frozen challenge
container** — the image installs only edfio/joblib/numpy/pandas/sklearn/scipy/tqdm (no torch,
no torch-geometric, no umap), `.dockerignore` drops `*.pt/checkpoints/`, and there is no
EDF→embedding code path in `team_code.py`. Our SleepFM/GNN `.npy`/`.npz` were precomputed
offline keyed to our cohort's subject IDs; on the hidden test set they don't exist (every row
→ `has_emb=0`). So a fused model is only worth the (heavy, risky) container work if it clearly
beats the deployable handcrafted baseline here. It does not.

Five arms, all leakage-safe (StandardScaler / median-imputer-for-UMAP / PCA / UMAP / AUC>0.6
filter / BMI imputer ALL fit on train rows only; UMAP unsupervised; missing emb → 0 + has-flag):
**A** baseline 436 · **B** concat-append 436+PCA32(SleepFM)+GNN16 · **C** the *literal recipe*
(fuse all three → UMAP-32 the whole matrix → train on UMAP dims alone) · **D** late-fusion
(UMAP-32 the fused embedding block, keep raw tabular, append) · **E** = D + AUC>0.6 select.
This is NEW vs prior passes (they only APPENDED one block and UMAP'd the embedding block alone;
here SleepFM+GNN are fused and the select-before/after/none ordering is isolated). 10 seeds,
1500× bootstrap, ncomp=32.

**Paired 70/15/15 Δreward vs baseline (identical splits — the CI that can exclude 0):**
B −0.037 [−0.264,+0.053] wins 4/10; C −0.000 [−0.167,+0.394] wins 3/10 (Δage-AUROC −0.073);
D −0.012 [−0.245,+0.328] wins 3/10; E −0.020 [−0.120,+0.042] wins 5/10. **Every Δ CI spans 0,
every point ≤ 0, no arm wins a majority.**

**LOSO reward@π (1500× boot, the new-site proxy):** A +0.221 [−0.010,+0.493] auroc 0.721;
B +0.268 [+0.030,+0.552] auroc 0.719; **C −0.028 [−0.318,+0.307] auroc 0.536, age-AUROC 0.516
(near chance)**; D +0.268 [+0.008,+0.564] auroc 0.711; E +0.221 auroc 0.712. B/D's LOSO point
edges baseline but their CIs overlap it almost entirely AND they LOSE the paired 70/15/15 test
→ split-variance mirage (same pattern the earlier GNN pass flagged), not signal.

**Verdict** ⚠️: **NO robust lift — 5th independent pass agreeing.** The literal "UMAP the whole
fused thing" recipe (arm C) is actively HARMFUL — compressing 3016 fused dims into 32
unsupervised components discards the signal, which SHAP shows lives in a few specific
handcrafted features (age, HRV, architecture), not a global manifold. The n=84-positive ceiling
is not rescued by fusion, UMAP, or any feature-select ordering. Combined with the can't-deploy
container fact → **do NOT ship an embedding-fused model.** The evidence-backed submission is a
refreshed HANDCRAFTED model (collapse the inert MoE+Kaiser head → single pooled model; keep
label-free per-site z-score — both already DECIDED). Aggregate stats only left the box.

---

## 2026-08-19 (Model architecture A/B — pooled vs site-MoE vs router + kNN personalization) 🔬

Answered "site-agnostic vs patient-specific vs the current 3-site MoE, does a router help,
and does anything crash on an unseen site?" `scripts/eda/site_arch_ab.py` +
`scripts/eda/arch_confirm.py` (bootstrapped CIs). **Structural insight that drives the
design:** under strict LOSO the held-out site has no expert, so the MoE falls back to
`global` → **site-MoE ≡ pooled model on unseen sites, by construction** (confirmed:
identical 0.635/0.712/+0.274 to the digit). So specialization/routing can only pay off
IN-DISTRIBUTION. Evaluated two regimes.

**Regime A — unseen site (LOSO):** pooled == site_moe (as predicted); **pooled_knn**
(pooled blended 0.5 with a leakage-safe kNN local positive-rate — "patient-specific" for a
single-session cohort) is the only arm that moves up: AC-AUROC 0.649 vs 0.635, AUROC 0.732
vs 0.712, reward +0.354 vs +0.274.

**Regime B — known site (10-seed in-dist split):** pooled BEATS the MoE on AC-AUROC
(0.647 vs 0.609, MoE wins 3/10) and AUROC; router 0.596 and **oracle_router (routing
ceiling) 0.609 — below pooled 0.647.** Since the ceiling loses to pooled, no real router
can win. **Site separability = 0.995** (feature→site classifier; baseline 0.777) so routing
works mechanically — routing was never the problem; the destination (a positive-starved
per-site expert) is. **Router: dead, ceiling-tested, cheaply.**

**Bootstrapped confirmation (`arch_confirm.py`, paired bootstrap 2000×):**
- kNN unseen-site: Δ vs pooled AC-AUROC **+0.014 [−0.026,+0.056]**, AUROC +0.020
  [−0.008,+0.052], reward +0.080 [−0.026,+0.227] — all point UP, all CIs span 0. Effect is
  monotone/stable in blend (0.636→0.649→0.648) and the **physiology-only kNN variant matches
  (+0.016)** → not an age-prevalence lookup, genuinely uses neighborhood physiology.
  📌 Promising lead, not yet significant; kNN gains with sample size → revisit on 497-cohort.
- pooled vs MoE in-dist (8× 5-fold OOF): AC-AUROC Δ −0.009 [−0.063,+0.050] (tie), **AUROC Δ
  −0.039 [−0.075, +0.000]** (whole interval ≤0 → pooled significantly better), reward +0.004
  (threshold noise). **No regime where the 3-site MoE beats a single pooled model.**

**Robustness:** fault-injected a bogus `"ZZZZ_unseen_site"` + NaN ages through MoE/router/
oracle → all finite via global fallback. **PASS — never crashes on unseen sites.**

**Recommendations:** (1) 📌 collapse the site-MoE + Kaiser head to ONE pooled model —
equal-or-better on both regimes, simpler, robust by construction; (2) drop the router;
(3) develop kNN personalization, best tested on the 497-positive cohort. Aggregate stats
only left box.

---

## 2026-08-19 (SHAP attribution + LOSO learning curve) 🔬

`scripts/eda/shap_analysis.py` (TreeExplainer on the raw production-hyperparam HistGBM,
isotonic wrapper stripped so the tree is introspectable; LightGBM fallback available) and
`scripts/eda/learning_curve.py` (subsample TRAIN positives under LOSO; test site always
full). box has shap 0.49.1 / numpy 2.2.6 / sklearn 1.7.2 — no install (air-gap OK).

**SHAP (mean|SHAP|, in-sample over 1103):** age 0.354 ≫ bmi 0.127 > nk__hrv_n1_hf 0.089 >
arch__arousal_idx_rem 0.079 > rep__eeg_n1_abs_theta 0.072 > nk__hrv_rem_cvnn 0.054 …
**Age dominates (~3× next) — this is WHY AC-AUROC (0.635) < plain AUROC (0.712):** the model
leans on age and AC-AUROC neutralizes it. Under age/bmi, the handcrafted physiology carries
real weight — HRV (autonomic), REM arousal index, N1 spectral θ, RSWA. Artifacts in
`exports/shap/`: shap_importance.csv, shap_bar/beeswarm/dependence_top6.png.

**Learning curve (LOSO, 8 seeds, TRAIN positives subsampled):** AC-AUROC by n_pos —
21→0.535, 34→0.559, 46→0.555, 59→0.575, 71→0.589, **84→0.635**; reward 0.024→0.072→0.113→
0.173→0.196→**0.274**. **Still climbing steeply at the right edge — last step (71→84) is the
STEEPEST (+0.046 AC-AUROC), no saturation.** f=1.0 reproduces the 0.635 anchor exactly.
**Strongest evidence yet that finishing the 497-positive large cohort is the right
investment** — more positives is the one lever that reliably moves the metric (embeddings,
extreme-filter, soft-weight were all flat/negative). ⚠️ caveat: subsamples within the 3
existing sites → pure count effect; real 497 gains may be tempered by new-site transfer.
Artifacts: `exports/learning_curve/learning_curve.{csv,png}`.

---

## 2026-08-19 (Extreme-case training A/B — hard filter vs soft confidence weight) 🔬

Tested the idea "train only on extremes (fast converters ≤2y vs CI-free >7y) so labels and
predictions don't overlap." `scripts/eda/extreme_vs_soft_ab.py`: baseline (all data) vs
extreme_hard (827 rows: 21 pos + 806 neg, the notebook's §5 cohort) vs soft_w{0.5,0.3,0.2}
(down-weight the ambiguous middle by outcome confidence via `sample_weight`; extreme_hard is
the floor→0 special case). Survival (tte/ttlv) joined from `features_standard.csv` by
bids_folder. **ALL evaluated LOSO on the FULL population** (never filtered), decision policy
+ prevalence held fixed so only the training data varies.

**Verdict: BOTH hurt on the full population — baseline wins on every metric.**
- extreme_hard LOSO: **AUROC 0.477 (below chance), AC-AUROC 0.473, reward ~0** vs baseline
  AUROC 0.712 / AC-AUROC 0.635 / reward@prod 0.324. Ranking literally inverts — reproduces
  the notebook's §9 collapse (held-out middle AUC ~0.50) at population scale + on the reward.
- soft weighting: AC-AUROC 0.55–0.59 (all < 0.635), reward roughly halved — gentler, same
  direction, still harmful.
- **The trap, made explicit:** on 70/15/15 random splits extreme_hard looked GOOD
  (+0.060 AC-AUROC, 7–8/10 wins) — same-distribution ~12-positive test + within-site
  structure reward the cleaner training cohort. LOSO removes both crutches and the mirage
  flips to below-chance. Clean demonstration of why we insist on LOSO.
- Root cause: extremes teach "about to convert within 2y," which doesn't transfer to a
  general population; the overlap IS the real difficulty the Challenge grades. Bottleneck
  remains positives + site transfer, not label noise. Aggregate stats only left box.

---

## 2026-08-19 (Embeddings — multivariate value test, NOT univariate significance) 🔬

Answered "do Kingson pure (2560-d) + PhysioGraph GNN (16-d) embeddings add PREDICTIVE
lift beyond the 436 handcrafted features?" the correct way — multivariate added-value,
not univariate significance on stochastic t-SNE axes (which would be non-reproducible).
`scripts/eda/emb_value_test.py` adapts `umap_fuse.py`: baseline=436 feats; append each
embedding block (gnn raw16, pure PCA32/PCA64, pure UMAP16); leakage-safe (reducer +
AUC>0.6 filter fit on TRAIN rows only, .transform held-out; missing emb → zero + has_emb
flag). Matched to cache pids by SUBJECT (strip _ses-N; single-session): pure 1080/1103,
gnn 1090/1103. 10 seeds 70/15/15 (paired deltas) + LOSO pooled reward@π boot 1500×.

**Verdict: NO robust predictive lift — same n=84 fragility as the earlier PCA/UMAP fusion
pass.** Paired 70/15/15 Δreward vs baseline: gnn +0.122 [−0.031,+0.942] wins 2/10; pure
PCA32 +0.001 wins 4/10; PCA64 +0.004 wins 6/10; UMAP16 −0.007 wins 4/10 — **every CI spans
0, every Δage_auroc ≤ 0.** LOSO reward@π: gnn +0.305 [+0.035,+0.615] vs baseline +0.221
[−0.010,+0.493] — looks like a gain BUT the two CIs overlap heavily and gnn wins only
2/10 paired seeds, so it's split-variance noise, not signal (also contradicts gnn's own
README: age-only baseline ~0.50–0.54 on LOSO age-AUROC). AUROC essentially flat across all
(0.71–0.72). **Confirms for a THIRD representation family: the bottleneck is
positives/variance + site transfer, not the feature/representation richness — learned
embeddings don't rescue it.** De-prioritize embedding fusion. Aggregate stats only left box.

---

## 2026-08-19 (Phase-2 large NK+C1 done → significance re-run on 497 positives) 🔬

**Phase-2 NK+C1 large re-extraction COMPLETE.** All 3 shards hit 2200/2200 and
exited clean (2164+2180+2186 = 6530 recs, 0 errors, ~63 "no-file/staging" missing
EDFs across shards). Merged via new `scripts/merge_nkc1_shards.py` (asserts 3
byte-identical headers, dedups keys, backs up prior): `nk_features_large.csv` = 6530
× **194 cols WITH 34 C1 dfa/sampen cols**; `report_features_large.csv` re-merged 132
cols. (micro_large still NOT extracted — remaining gap before full 458-col parity.)

**Univariate significance re-run on the LARGE cohort (`scripts/eda/significance_large.py`
→ reuses `stats_significance.py` UNCHANGED: Welch t + Mann-Whitney, χ², Cohen d /
Cramér V, BH-FDR q<0.05).** Joined features_+arch_+nk_+report_ (6530/6530 all
families), label True/False→1/0. **217 / 401 features significant at q<0.05** — vs
**14/48 in the power-limited n=84 standard run.** The extra power surfaced the new
handcrafted families as REAL: **C1 nonlinear HRV 25/34 sig** (`nk__hrv_wake_dfa_a2`
d=−0.37, DFA/SampEn per stage), **arch dynamics 26/45 sig** (`arch__arousal_idx_n1`
d=−0.40, `arch__trans_p_wake_n1`, `arch__trans_entropy_rem`), nk 122/183, rep 44/122,
base 25/51. Top effects unchanged in rank: **age d=1.10** (confounder) ≫ wake θ (0.54),
N1 θ/α (0.47), bmi (−0.44), ↓REM%/REM-epochs, PLMI (0.42), ↓stage_entropy (−0.37).
CSV: `exports/feature_significance_large.csv`. ⚠️ Significance ≠ predictive lift
(n=84 batch-1+2 reward CIs still spanned 0) — this vets the features + powers the
paper table, doesn't itself move the primary metric. Aggregate stats only left the box.

---

## 2026-08-18 (Learned-embedding viz — Kingson "pure" + PhysioGraph GNN) 🔬

Ran the same PCA/UMAP/t-SNE 3×3 panel (`scripts/eda/embed_viz.py`, two new loaders:
`load_from_embnpz` for a self-contained embedding npz; `load_from_embdir` for a dir
of per-recording `.npy` joined to a meta npz) on two teammate-supplied LEARNED
embeddings, to see whether a trained representation separates CI where the
hand-crafted features don't.

**Sources.** (1) Kingson's `/data-temp/embeddings/embeddings_by_stage_pure/` — 1080
per-recording `.npy`, each **2560-d** (per-stage concatenation), keyed
`sub-<PID>_ses-<n>`; **joined 1080/1080** to the GNN meta npz for label/site/age (77
CI+). (2) `physiograph_last_layer_embeddings.tar.gz` — PhysioGraph++ C8 temporal-GNN
last layer, self-contained npz `embedding (1090×16)` + patient_id/site/age/label/split
/head_risk (84 CI+). → `exports/embed_viz_pure.png`, `embed_viz_gnn.png` (+ `_coords.npz`).

**Result — same verdict as hand-crafted features, sharper: NO embedding separates
CI+/CI−.** In both sources, across PCA/UMAP/t-SNE, CI+ (orange) is diffused through
the CI− cloud — no cluster, no boundary. This is now three independent feature
families (337-col engineered, 2560-d pure, 16-d GNN) all agreeing the CI signal is a
weak multivariate gradient, not a geometric split. GNN README's own HONEST NOTE
concurs: under LOSO age-AUROC these sit at the **age-only baseline ~0.50–0.54** (meta
test_ac_auroc=0.493) → "features, not a validated risk score."

**But site structure is even STRONGER in the learned embeddings than in the
hand-crafted features** — the pure 2560-d t-SNE splits into crisp per-site islands
(Kaiser=pink corner, Emory=vermillion sub-cluster, BIDMC=green bulk); GNN 16-d shows
Kaiser as its own arm. The learned encoders soaked up site identity at least as much
as CI. Age remains a smooth gradient. **Takeaway: swapping hand-crafted → learned
representations does NOT fix the binding constraints (weak CI separability + site
shift); if anything the embeddings encode site more crisply, reinforcing that the
lever is more training DOMAINS / harmonization, not a richer single-cohort encoder.**
Aggregate/plot only — no data left the box.

---

## 2026-08-18 (PCA / UMAP / t-SNE visualization) — t-SNE added, embeddings visualized 🔬

Added t-SNE alongside PCA + UMAP and rendered a 3×3 embedding panel
(`scripts/eda/embed_viz.py` → `exports/embed_viz_standard.png` + `_coords.npz`).
Rows = PCA/UMAP/t-SNE; cols = colored by CI label / site / age. Okabe-Ito
colorblind-safe categorical + viridis for age; CI+ drawn on top.

**tsne-cuda (the requested GPU lib) CANNOT run on this box** — its pip wheel is
built for CUDA 10.2 (needs `libcudart.so.10.2`); the server has CUDA 12.2/12.5 only
→ hard ABI gap. Also needed Intel MKL (installed to physio-user `.local` +
bare-name symlinks) but the cudart mismatch is fatal; building from source vs CUDA
12 needs a matching FAISS-GPU + toolchain (multi-hour, likely-fail). Moot anyway:
at 1103×436 (or 6530×213) **sklearn Barnes-Hut CPU t-SNE finishes in <1 min**; GPU
t-SNE only pays off at 1e5–1e6 points. Uninstalled tsnecuda after probing.

**What the figure shows (reinforces the whole arc):** (1) **No CI+/CI− separation**
in any of the three embeddings — CI+ (orange) is diffused through the CI− cloud, not
a cluster. Consistent with our modest AUROC: the signal is a weak multivariate
gradient, not a manifold split — no reducer will magic out a boundary. (2) **Site
structure IS visible** (esp. UMAP/t-SNE: Kaiser=pink and BIDMC=green occupy different
regions) → the between-site shift is real and geometric, exactly the transfer problem
LOSO exposed. (3) **Age shows a smooth gradient** (viridis) — the confounder is a
dominant axis of variation, as expected. Nets: unsupervised embeddings won't find a
CI cluster (matches the earlier UMAP/PCA fusion pass that didn't robustly earn a
place), but they visually CONFIRM the site-shift diagnosis. Aggregate/plot only —
no data left the box.

---

## 2026-08-18 (Site-agnostic strategy for new-site test set) — MoE is inert, decision made 🔬

Team decision: assume the WORST — hidden test set is NEW sites → LOSO is the real
metric, 70/15/15 is vanity. Tested site-agnostic variants under LOSO on the large
cohort (`scripts/site_agnostic_test.py`, all LOSO-honest):

| variant | reward@π | AUROC | age-AUROC | AUPRC | held-out BIDMC reward |
|---|---|---|---|---|---|
| A moe_baseline (site experts + Kaiser) | +0.086 | 0.656 | 0.591 | 0.143 | +0.021 |
| B pooled (single global model) | **+0.086** | **0.656** | **0.591** | **0.143** | +0.021 |
| C pooled + site_z | **+0.092** | 0.664 | 0.609 | 0.146 | +0.012 |
| D pooled + site_z + drop 25% site-discrim | +0.072 | 0.663 | **0.620** | **0.148** | **+0.046** |

**HEADLINE: A == B byte-for-byte.** The site-MoE + Kaiser fine-tune (the machinery
behind the 0.168 submission) contributes **exactly nothing** under LOSO — an unseen
site has no expert (falls back to global) and the Kaiser head only fires on Kaiser
rows, never present in a held-out-Kaiser test. **So for a new-site test set the
entire site-specialization stack is inert → replace with a single pooled model
(simpler, identical LOSO, less overfit).** Clean win.

**site_z (C):** best primary reward (+0.086→+0.092) AND better ranking (age-AUROC
+0.018) — keep as free, label-free preprocessing.

**Feature-invariance drop (D):** dropping the 25% most site-discriminative features
(between/within-site variance ratio, train-only) improves ranking (age-AUROC
0.591→0.620) and the biggest/hardest fold (BIDMC 5075: reward +0.021→+0.046,
AUROC too) but LOWERS pooled reward (small Emory/Kaiser folds drop).

**Drop-frac sweep (`run_dropfrac_sweep.sh`) — verdict: feature-drop does NOT help
the primary reward.** reward@π by drop-frac: 0%(=site_z) **+0.092** > 5% +0.088 >
10% +0.090 > 15% +0.081 > 20% +0.071 > 30% +0.039. site_z with NO drop is the reward
optimum; every drop level is worse on reward (monotonic collapse past 15%). Dropping
DOES modestly help ranking metrics (AUROC peaks ~0.672 at 5%, age-AUROC ~0.622 at
20%, AUPRC ~0.155 at 20%), but reward is the scored metric → **skip invariance-drop,
keep site_z only.** The site-discriminability ratio is too blunt a selector; a
supervised invariance method (site-adversarial) might do better but isn't worth it
at this ceiling.
Note large-LOSO reward (~0.09) < standard (~0.15) because it's dominated by the
78%-of-cohort BIDMC hold-out (train on 2 small sites → predict the giant one).

### Decisions
1. **Drop site-MoE + Kaiser → single pooled model** for the new-site world (proven inert).
2. **Adopt site_z** preprocessing (small clean gain, test-legal).
3. **Sweep drop-frac** for invariance selection (10/15/20/25%) — promising but untuned.
4. ⚠️ **Hard ceiling: only 3 sites.** LOSO = train 2 domains, predict 1. The real
   ceiling-raiser is MORE TRAINING DOMAINS → external public sleep cohorts
   (SHHS/MrOS/MESA/CFS, all have EEG/ECG/resp + our signals). This is the long-game.

---

## 2026-08-18 (Cross-site harmonization) — TESTED, marginal ⚠️

Tested whether harmonizing the feature distribution across sites lifts LOSO
transfer on the cheap 213-col large cache (`scripts/harmonize_test.py`, all
LOSO-honest: held-out site standardized by its OWN feature distribution, never its
labels). Pooled LOSO reward@π / AUROC / age-AUROC:

| transform | reward@π | AUROC | age-AUROC | held-out BIDMC reward |
|---|---|---|---|---|
| none | +0.086 | 0.656 | 0.591 | +0.021 |
| site_z (per-site z-score) | +0.092 | **0.664** | **0.609** | +0.012 |
| combat (self-contained EB) | +0.040 | 0.645 | 0.590 | +0.001 |

**ComBat hurt** (removed signal with the site shift). **site_z gives a small real
ranking gain** (age-AUROC +0.017, AUROC +0.008; age-AUROC is scored → worth keeping
as near-free preprocessing) **but does NOT fix transfer** — pooled reward flat,
held-out BIDMC reward slightly worse. `neuroCombat`/`neuroHarmonize` not installed
(pip→internal mirror); ComBat hand-rolled. Strategic takeaway below.

### The real picture (two tests + Phase 1 together) 📌
The **70/15/15 vs LOSO gap IS the transfer problem**: mixing all sites → AUROC 0.86;
holding out a site → AUROC 0.66. More positives (Phase 1) crushed the *within-
distribution* variance but barely moved *held-out-site* performance, and neither the
age-threshold nor harmonization closes it. **The single most important unknown is
whether the challenge's hidden test set is NEW sites (LOSO-like, we're at ~0.66) or
held-out patients from the SAME sites (70/15/15-like, ~0.86).** That determines
which number is real. Action items: (1) confirm the test-set composition from the
challenge rules; (2) if new-site, the lever is domain generalization (site-
adversarial training / a single pooled model with site-invariant features), not more
data or post-hoc harmonization; (3) keep site_z as free preprocessing regardless.

---

## 2026-08-18 (Large-cohort pivot, Phase 1) — variance collapses, gate PASSED 🔬

The pivot everything pointed to: extract features on the 6530-recording large cohort
(**497 positives vs 84**, 5.9×). Phase 1 = the cheap already-extracted families only
(features_ + arch_ + report_, 213 cols), built IDENTICALLY for both cohorts so
**positives is the only variable** (`scripts/phase1_large_variance.py`; no multi-session
leakage — every bids_folder unique in both). 50-seed 70/15/15 sweep + LOSO through the
production stack:

| | standard (84 pos) | large (497 pos) |
|---|---|---|
| 70/15/15 reward | +0.262 **± 0.495** | +0.494 **± 0.130** |
| reward_at_pi | +0.341 ± 0.544 | +0.550 ± 0.116 |
| age_auroc | 0.642 ± 0.102 | **0.827 ± 0.027** |
| auroc | 0.674 ± 0.075 | **0.861 ± 0.019** |

**Variance collapses ~4× (reward SD 0.495→0.130), ranking jumps hard (AUROC .67→.86,
age-AUROC .64→.83).** This is exactly the diagnosed bottleneck (positives/variance)
being fixed — GATE PASSED, proceed to Phase 2 (re-extract NK+C1 large + micro_large
for full 458-col parity). ⚠️ **Caveat — cross-site transfer NOT fixed:** LOSO pooled
reward@π stays low (large +0.086, dominated by held-out BIDMC n=5075 at +0.021 —
worse than standard's +0.099). More data helps WITHIN-distribution enormously but the
held-out-SITE problem needs harmonization (ComBat / site-invariant) separately. The
challenge's hidden test set is new sites → LOSO is the honest proxy, so Phase 2 alone
won't win; harmonization is the natural next lever after parity. Note 70/15/15
reward is inflated vs LOSO because splits mix all sites.

---

## 2026-08-15 (Age-conditioned Bayes threshold) — TESTED, negative result ⚠️

The long-standing "top improvement": the reward's Bayes-optimal rule is *predict
positive iff q > p(age)* (age-specific prevalence, not global). Derivation is
correct (payoffs TP=1/p−1, FP/FN=−1, TN=1/(1−p)−1 → q>p) **but assumes q is the
age-CONDITIONAL calibrated posterior.** Our model deliberately excludes age, so
q=P(CI|sleep) marginal over age — the premise is violated. `scripts/age_threshold_ab.py`
applies rival decision rules to the SAME LOSO OOF probs (model fixed):

| rule | reward (NEW) | n_pos_pred | sens | spec |
|---|---|---|---|---|
| global_pi (q>0.076, current) | **+0.274** | 560 | 0.786 | 0.515 |
| global_grid (oracle best single thr=0.074) | +0.283 | 565 | 0.798 | 0.511 |
| ageBayes_oracle (q>p_full(age)) | +0.179 | 717 | 0.548 | 0.342 |
| ageBayes_train (q>p_train(age), deployable) | +0.183 | 610 | 0.452 | 0.439 |

Bootstrap ageBayes−global_pi Δ=**−0.095**, P(Δ>0)=0.14 → **it REGRESSES.**
Mechanism: true positives cluster at older ages → high p(age) → high threshold →
they get missed (sens 0.79→0.55), while low-p young bands flag mostly negatives
(n_pos_pred 560→717). `global_grid`≈current → **current threshold is already ~optimal.**
Confirms the bottleneck is **sample size / calibration, not the decision rule.**
Would only help if the model INCLUDED age (q age-conditional) — but age is the
confounder the metric discounts (circular). **Model swap answer:** xgboost NOT on
box; lightgbm 4.7.0 IS, but AUROC is flat across the seed sweep so a GBDT-library
swap can't beat the ±0.45 split variance. Real levers now: more positives (large
6530 cohort / external OSA-cognition data), split-ensembling, site harmonization.

---

## 2026-08-15 (Advanced features batch 1+2 A/B) — measured LOSO lift 🔬

Full cohort re-extracted (nk 194 cols incl. C1; micro 1090/1090, 0 errors), cache
rebuilt **337 → 436 features**, A/B run through the exact production stack
(`scripts/ab_batch1.py`, per-family leave-one-out). **Pooled LOSO, reward@π
primary:**

| metric | OLD (337) | NEW (436) | Δ |
|---|---|---|---|
| **reward@π** | +0.1759 | **+0.2738** | **+0.098** |
| AUROC | 0.678 | 0.712 | +0.034 |
| age-AUROC (AC-AUROC) | 0.580 | 0.635 | +0.055 |
| AUPRC | 0.197 | 0.180 | −0.016 |

**Clean control:** masking all 99 new cols reproduces OLD *exactly* (+0.1759) →
gain is the features, not base-nk re-extraction drift. Gain concentrated in the
**largest / most stable LOSO fold** (BIDMC n=857: +0.051 → +0.238); Emory (n=54)
and Kaiser (n=192) ~flat. **Per-family marginal (NEW − NEW-without-family):**
reward@π — c1_hrv **+0.073**, micro **+0.046**, arch **+0.032**; age-AUROC — micro
**+0.049** biggest (the expensive signal-microstructure features carry real decline
signal — reverses the prior "low ROI at n=84" assumption). All three families help
reward@π. Marginals sum > total (0.15 vs 0.098) → correlated, non-additive.
⚠️ AUPRC dipped −0.016 (reward-threshold trade favours the prevalence-weighted
objective).

**Robustness (`scripts/robust_boot.py`, harness point-estimates ASSERTED == pipeline):**
- *Paired subject bootstrap on pooled LOSO OOF (B=10k):* reward@π Δ=+0.098, 95% CI
  **[+0.0006, +0.213]**, P(Δ>0)=0.976 — positive but lower bound grazes zero.
  age-AUROC Δ=+0.055, 95% CI **[+0.005, +0.108]**, P=0.985. Per-family marginal CIs
  ALL span 0 (c1_hrv P=0.88, micro 0.77, arch 0.71) → shared correlated signal, no
  single family individually significant.
- *70/15/15 seed sweep (50 random splits, paired):* the sobering angle. reward@π
  meanΔ=+0.057 win=0.62; age-AUROC meanΔ=+0.016 win=0.62; AUROC meanΔ≈0 win=0.58;
  AUPRC meanΔ=−0.009 win=0.38. Fitted-threshold `reward` (site_decade) meanΔ=**−0.060
  win=0.32** (NEW *loses* — the fitted-threshold path overfits tiny val sets). Split
  variance is enormous (reward ±0.45–0.62).

**Verdict 🔬📌:** features are real signal and help the primary LOSO metric on point
estimate + subject-bootstrap (borderline-significant), but split-choice variance at
n=84 positives dominates — NOT a locked win. Keep the features (principled, don't
hurt reward@π), but the bottleneck is **positives/variance, not features** → **Batch 3
DE-PRIORITIZED** (adding heavy multi-channel cols to a p≫n problem raises variance).
Next levers: age-conditioned Bayes threshold (free, metric-aligned), large-cohort
positives, split-ensembling, site harmonization. See [[challenge-2026-approach]].

---

## 2026-08-14 (Advanced features batch 2) — sleep-microstructure extractors

Batch-2 of the roadmap: the "new but single-channel EEG / chin-EMG, moderate
effort" set. All three are single-channel waveform features built on `eeg_spectral`
primitives (Butterworth→filtfilt→Hilbert, `_detect_bursts`, `_align_epochs`) — no
multi-channel decode. Unit-tested on synthetic signals AND probed on real
recordings across all three sites (channel-role matching handles `CHIN1-CHIN2`
[BIDMC] / `CHIN` [Emory] / `ChinA` [Kaiser]; EEG `C3-M2` / `C3`; arousal 2 Hz).

### B3 — SO–spindle coupling · `scripts/viewer/eeg_coupling.py` ✅ 🔬
For each detected spindle (reusing `eeg_spectral`'s sigma envelope + N2/N3-calibrated
threshold), read the slow-oscillation (0.5–1.25 Hz) instantaneous phase at the
spindle peak; summarise the circular distribution: **coupling_strength** = mean
resultant vector length (near-independent of spindle count), preferred phase
(cos/sin), Rayleigh-z modulation, per N2/N3/pooled-NREM (15 cols). Declining
SO–spindle coupling is a leading memory-consolidation/cognitive-decline EEG
biomarker. **Synthetic check:** coupled EEG → strength 0.98 / z 279; uncoupled →
0.03 / z 0.28 (spindle counts equal — it captures phase, not count).

### E2 — RSWA (chin-EMG atonia) · `scripts/viewer/emg_atonia.py` ✅ 🔬
REM-sleep-without-atonia = prodromal α-synuclein-neurodegeneration marker. All
metrics normalised to the recording's OWN atonia floor (20th-pct REM EMG RMS) so
they survive the montage heterogeneity: `rswa_rem_nrem_ratio` (>1 = REM not the
quietest stage), `rswa_tonic_fraction`, `rswa_phasic_per_min`, tone ratio, RMS CV
(5 cols). **Real-data signal:** one BIDMC subject showed rem/nrem 1.09 + 47
phasic/min (strong RSWA), another 0.36 + 1/min (normal atonia) — genuine
between-subject variation.

### A2 — CAP (approximate) · `scripts/viewer/cap_events.py` ✅ 🔬 ⚠️
Cyclic Alternating Pattern = NREM instability. Transparent surrogate for Terzano
scoring (NOT clinical): 2 s band-power activation vs a moving-median background →
A-phases (2–60 s), subtyped A1/A2/A3 by fast-power fraction + arousal overlap,
assembled into CAP sequences by 2–60 s B-gaps. `cap_rate` + A-indices + subtype %
+ sequence stats (11 cols). ⚠️ **Calibrated on the box:** `ACT_FRAC=0.75` lands
`cap_rate` at ~0.48 (literature adult 0.3–0.5); 0.5 gave ~0.78 (too high). A1
dominant (~75–99%), as expected.

### Wiring
`scripts/viewer/export_micro_features.py` (one physio pass: EEG + chin-EMG + CAISR
arousal, shardable/resumable) → `micro_features_standard.csv` (31 feature cols).
`micro__` merge block added to `build_local_feature_cache.py`. `scripts/ab_batch1.py`
extended to isolate the `micro__` block too. Cache after both batches ~405 → **~436
features**. **Code review** found + fixed: a `--resume` SessionID-default dedup risk;
a coupling threshold calibrated on N1+N2+N3 vs the detector's N2+N3 (now matched).

### Status 📌
Extractors built, tested, deployed. Full `micro` extraction deferred until the
batch-1 NK re-extraction finishes (avoid two physio-EDF passes contending). Then:
extract micro → rebuild cache → A/B each family (`arch__`, `micro__`, C1 nk cols)
separately vs the pre-batch snapshot. Batch 3 (B1 connectivity, B2 microstates)
NOT built — lowest ROI at n=84, deferred pending batch-1/2 signal.

---

## 2026-08-14 (Advanced features batch 1) — dynamic/instability extractors

From the manager's advanced-feature proposal (`ADVANCED_FEATURES_COVERAGE.md`),
built the batch-1 "cheap + high-value, no new signal decoding" set. All three
target the axis our error analysis says we're weak on — blunted cross-stage
*dynamics*, not night averages — and reuse streams we already decode.

### C1 — nonlinear HRV (DFA + sample entropy) in `scripts/viewer/nk_features.py` ✅ 🔬
Added `_dfa_alpha` (detrended fluctuation analysis; **α1** short-term 4–16-beat,
**α2** long-term 16–64-beat fractal scaling) and `_sampen` (Richman-Moorman sample
entropy) computed **per stage/pool on the SAME clean RR arrays `ecg_hrv_by_stage`
already builds** — so zero extra R-peak detection. 23 new columns
(`hrv_<stage>_{dfa_a1,dfa_a2,sampen}` × 7 pools + `rem_nrem_{dfa_a1,sampen}_ratio`
contrasts). We had Poincaré SD1/SD2 but no fractal/entropy HRV.
- **Compute care:** DFA per-box detrend is closed-form vectorised (no per-box
  `polyfit` loop — that made it 30 s/rec; back to ~9–13 s/rec); SampEn length-capped
  to 1200 beats. The O(n²) `nk.hrv_nonlinear`/`fractal_*` remain avoided.
- Verified vs reference implementations: DFA matches a per-box-`polyfit` reference
  to ~1e-15; SampEn matches a double-loop Richman-Moorman reference exactly (a
  first cut had an m+1 off-by-one that biased entropy high — caught in review, fixed).

### F1 + A1 — `scripts/viewer/arch_dynamics.py` + `export_arch_features.py` ✅ 🔬
New CAISR-annotation-only module + exporter (no waveforms → 1090 recordings in 8 s):
- **F1 stage-transition Markov:** full 5×5 row-normalised `trans_p_<from>_<to>`,
  per-row conditional entropy `trans_entropy_<from>`, occupancy-weighted
  `trans_entropy_rate` (disorder) and `trans_stability_index` (persistence). We had
  `stage_entropy` + 6 named per-hour rates but not the normalised matrix or its
  entropies. Reuses `dynamics.transition_matrix`/`_row_normalise` for consistency.
- **A1 microarousals:** arousal duration + inter-arousal-interval distributions
  (`arousal_iai_cv` = clustering) + per-stage arousal index. Reads arousal `fs` from
  the EDF (**confirmed 2 Hz** on this cohort) so durations are in real seconds.
- Wired an `arch__`-prefixed block into `build_local_feature_cache.py` (optional —
  cache still builds if the CSV is absent/empty; StopIteration guard added in review).

### Status 📌
NK re-extraction (adds C1 to `nk_features_standard.csv`) running on the box
(~3.5 h, background); `arch_features_standard.csv` done. Next: rebuild `_plus`
cache and LOSO A/B (reward + AC-AUROC) vs the pre-batch-1 cache, then subgroup/
error re-check. Numbers appended here when measured.

---

## 2026-08-14 (Subgroup + error analysis) — where/why the model fails

Manager ask: subgroup analysis of predictions + per-subject drill-down into why a
case failed; feature-distribution and TTE-by-outcome figures; "entropy/skewness".
All from the SAME leakage-safe LOSO out-of-fold pass (site-MoE + Kaiser + BMI,
per-fold prevalence threshold). Row-level outputs stayed on the box.

### `scripts/eda/subgroup_analysis.py` ✅ 🔬 — subgroup metrics + error case cards
Two artifacts: (1) **subgroup performance** table/CSV (recall/spec/precision/AUROC/
reward by sex, age-bin, site, BMI cat, AHI severity), (2) **case cards** CSV — one
row per misclassified subject with the top-8 features where it most deviates from
the CI− baseline (robust z), so you can open a specific failed subject and see the
driving physiology. Categorical one-hot dummies + age excluded from the "why" rank.
- **Fairness/robustness gaps:** recall by site BIDMC **0.43** / Emory 0.88 / Kaiser
  0.95 (but Kaiser spec only 0.18 — over-flagging); recall by age 50-59 **0.40** →
  80+ 0.77. Young converters and the biggest site (BIDMC) are where recall is lost.
- Overall LOSO: recall 0.60, spec 0.67, AUROC 0.68, reward 0.179 (matches anchor).

### `scripts/eda/make_feature_dist_figure.py` → `paper/figures/fig_feature_dist_by_outcome.png` ✅ 🔬
Robust-z (median/MAD vs CI− baseline) strip plots of the 6 most outcome-separating
features, split TP/FN/FP/TN, with per-group skewness. **Headline finding:** caught
converters (TP) sit well off z=0 (elevated HRV/EEG power, **reduced stage entropy =
blunted cross-stage modulation**), while **missed converters (FN) cluster at z=0,
overlapping the negatives (TN)** — the model misses converters that don't yet show
the signature. Directly validates the blunted-modulation hypothesis. (First cut
used mean/SD z and abs-power features → outlier-dominated, unreadable; switched to
robust z + median selection.)

### `scripts/eda/make_subgroup_figure.py` → `paper/figures/fig_subgroup_performance.png` ✅
Grouped recall+specificity bars faceted by subgroup axis, n_pos annotated, cohort
recall reference line. Renders locally from the aggregate `subgroup_metrics.csv`.
(TTE-by-outcome was already delivered 2026-08-13 as `fig_tte_confusion.png`.)

---

## 2026-08-14 (UMAP fusion + concordance) — UMAP vs PCA on SleepFM, Harrell C-index

Two manager-driven asks: (1) list the feature significance/selection tests + add
the C-statistic if missing; (2) try **UMAP** instead of PCA-32 on the SleepFM
embedding block.

### UMAP vs PCA-32 as the embedding reducer — `scripts/eda/umap_fuse.py` 🔬
Same leakage-safe protocol as `fuse_confirm.py` (site-MoE + Kaiser + BMI-impute;
scaler/PCA/UMAP **and** AUROC>0.6 filter fit in-fold on train rows only). Reducers:
PCA-32, unsupervised UMAP-16/UMAP-32, supervised UMAP-16 (fit with `y[train]`).
UMAP installed via the box's internal pip mirror (no external egress).

**70/15/15 balanced, reward mean [95% CI], 10 seeds:**
- `AUC>.6 only [ISOLATION]`   → +0.090 [−0.094,+0.311]  auroc 0.670
- `emb_PCA32 + AUC>.6`        → +0.041 [−0.114,+0.301]  auroc 0.653
- `emb_UMAP16 + AUC>.6`       → **+0.139 [−0.037,+0.453]  auroc 0.682**  ← best 70/15/15
- `emb_UMAP32 + AUC>.6`       → +0.075 [−0.172,+0.317]  auroc 0.683
- `emb_UMAPsup16 + AUC>.6`    → +0.047 [−0.061,+0.281]  auroc 0.654

**Paired per-seed Δreward (UMAP − PCA32), identical splits:** UMAP16 +0.098
[−0.064,+0.368] wins 7/10; UMAP32 +0.034 wins 7/10; UMAPsup16 +0.006 wins 6/10.

**LOSO pooled reward@π, bootstrap 95% CI (1500×):**
- `AUC>.6 only [ISOLATION]`   → +0.160 [−0.086,+0.456]  auroc 0.641
- `emb_PCA32 + AUC>.6`        → **+0.253 [−0.009,+0.557]**  auroc 0.631  ← best LOSO (matches prior)
- `emb_UMAP16 + AUC>.6`       → +0.156 [−0.094,+0.450]  auroc 0.620
- `emb_UMAP32 + AUC>.6`       → +0.114 [−0.132,+0.411]  auroc 0.625
- `emb_UMAPsup16 + AUC>.6`    → +0.105 [−0.058,+0.330]  auroc 0.610

**Verdict** ⚠️: **UMAP does NOT clearly beat PCA.** UMAP-16 is best under 70/15/15
(and beats the no-embedding isolation, which PCA does not), but PCA-32 is best
under LOSO — the two protocols disagree, every CI still spans 0, and supervised
UMAP is the weakest despite using labels. This is the same n=84 fragility as the
PCA pass: embeddings still don't earn a robust place. Do NOT quote either the
+0.253 (PCA/LOSO) or +0.139 (UMAP/70-15-15) as a win. UMAP-16's edge under the
protocol that matches the challenge's within-cohort split is worth one more look
(late-fusion/stacking, per-stage reduction) before discarding embeddings.

### Feature-test inventory + C-statistic — `FEATURE_TESTS.md`, `scripts/eda/harrell_cindex.py` ✅ 🔬
Cataloged every significance/selection test in the repo (Mann–Whitney U + AUROC,
Welch t + Cohen's d, χ² + Cramér's V, all BH-FDR; in-fold AUROC>0.6 selection;
permutation importance; leave-family-out ablation). Clarified the **binary
C-statistic == AUROC** (already reported everywhere; model LOSO C = **0.678**).
**Added Harrell's survival C-index** using `Time_to_Event` + `Time_to_Last_Visit`
(censoring) on the LOSO OOF risk scores: **C = 0.675 [95% CI 0.611, 0.736]**
(89,082 comparable pairs, bootstrap 1000×). Confirms the risk ranking is
time-concordant, not just class-concordant. Implementation validated against a
brute-force O(n²) reference (exact match, 200 randomized censored trials).

---

## 2026-08-12 (SleepFM embedding fusion) — QC + leakage-safe fuse into the model

A teammate (Kingson) produced **SleepFM per-stage PSG embeddings** →
`/data-temp/embeddings_by_stage/` (1080 `.npy`, one per recording, world-readable).
Task: QC them and, if clean, fuse with our NK2+report features and re-benchmark.

### QC ✅ 🔬 (`/tmp/qc_embeddings*.py`, run on box)
- **Structure:** 1080 files, all uniform `(2560,)` float; **2560 = 5 stages × 512-d**
  (Wake/N1/N2/N3/REM). Missing-stage blocks are **zeroed** — on 93/120 spot-checked
  recordings the all-zero 512-blocks *exactly* match the stages absent from that
  night's hypnogram; zero-block frequency highest for N3 (12%) / N1 (9%), the rarest
  stages. 🔬
- **Numerics:** 0 NaN / 0 inf / 0 all-zero / 0 constant / 0 dead dims; **1080/1080
  unique rows** (no dup/copy-paste); roughly standardized (global mean≈0, std≈0.78).
- **Alignment:** all 1080 map to a cohort recording (subject id); 23 of our 1103 have
  no embedding → get a zero vector + `has_emb=0` flag. One embedding per subject (no
  session ambiguity).
- **Provenance:** SleepFM = self-supervised contrastive foundation model, **not**
  trained on CI labels and not on this cohort → no label leakage from the embedder.

### Fusion benchmark (`/tmp/fuse_eval.py`) — leakage-safe ✅
Approach 1 (chosen over "concat-then-PCA", which lets 2560 emb dims drown the ~200
features). **PCA(32) on the embedding block AND the AUROC>0.6 univariate filter are
fit IN-FOLD on training rows only** — never on val/test (avoids selection leakage,
which at 84 positives biases test AUROC +0.05–0.10). **Age dropped from features**
(per user; challenge scoring is age-conditioned) but kept for scoring/stratification.
Balanced 70/15/15 stratified on **label × site × sex × age-bin** (label-first is the
critical stratifier at 7.6% prevalence); scored through the **exact production stack**
(feature_prep site-MoE + Kaiser fine-tune + BMI imputer; `evaluate_model.compute_*`),
so numbers are comparable to the 0.176 LOSO anchor.

Initial 5-seed result (LOSO gate): **`plus_noage + emb_pca32 + AUC>0.6` → pooled LOSO
reward +0.253** (vs 0.176 anchor, +44%), improving every held-out site incl. the hard
BIDMC (+0.05→+0.21). BUT: `emb_pca32 only` is **LOSO AUROC 0.498 (chance)** — embeddings
carry in-distribution structure but ~no *site-transferable* CI signal alone (random-CV
0.66 was in-distribution flattery). And 70/15/15 disagreed with LOSO, all dominated by
variance (~13 test positives; reward std ≈ mean). 📌

### Confirmatory pass (`/tmp/fuse_confirm.py`) ✅ 🔬 — verdict: NOT a win (yet)
To separate "embeddings help" from "AUC>0.6 selection helps": added an **isolation
config** `plus_noage + AUC>0.6` (no embeddings), **paired per-seed deltas** on identical
splits, **20 seeds** for a 70/15/15 CI, and a **bootstrap 95% CI on pooled LOSO reward**.
Ran to completion (`/tmp/fuse_confirm.log`, X=1103, y+=84 7.6%, emb 1080/1103).

**70/15/15 reward, mean [95% CI] over 20 seeds** — CIs are enormous (~13 test
positives; reward std ≫ mean), so none of these separate:
- `plus_noage (336)`            → +0.195 [−0.153, +1.298]  auroc 0.657
- `plus_noage + AUC>0.6`        → **+0.237** [−0.184, +1.567]  auroc 0.676  ← best point
- `plus_noage + emb_pca32`      → +0.203 [−0.121, +1.289]  auroc 0.657
- `plus_noage + emb_pca32+AUC`  → +0.137 [−0.275, +1.431]  auroc 0.666

**Paired per-seed deltas (identical splits) — the direct test:**
- (emb+AUC) − (AUC only) [**pure embedding value**]: **Δ = −0.100 [−1.319, +0.660], wins 9/20** → CI straddles 0; adding embeddings on top of the AUC>0.6 filter does *not* help (slightly hurts the point estimate).
- (AUC only) − (no select) [pure selection value]: Δ = +0.042 [−0.628, +1.149], wins 9/20 → selection alone also not significant here.
- (emb+AUC) − (no select) [combined]: Δ = −0.058 [−0.307, +0.348], wins 6/20.

**LOSO pooled reward@π, bootstrap 95% CI (2000× resample of recordings):**
- `plus_noage (336)`            → +0.145 [−0.102, +0.445]  auroc 0.627
- `plus_noage + AUC>0.6`        → +0.160 [−0.086, +0.447]  auroc 0.641  ← selection alone
- `plus_noage + emb_pca32`      → +0.128 [−0.131, +0.420]  auroc 0.619
- `plus_noage + emb_pca32+AUC`  → **+0.253 [−0.009, +0.561]**  auroc 0.631  ← best point, **CI still includes 0**

**Verdict** ⚠️: the earlier +0.253 was **real as a point estimate but not statistically
distinguishable** from either the no-embedding baseline (+0.145) or the selection-only
config (+0.160) — every CI includes 0, and the *paired* embedding delta is slightly
negative. **The SleepFM embeddings do NOT yet earn a place in the model** on this
evidence; the apparent lift is dominated by variance + the AUC>0.6 regularization, not
the embeddings. Do NOT quote +0.253 as a win. Next steps to revisit before discarding:
richer embedding pooling (per-stage PCA rather than one 32-d block), late-fusion/stacking,
or a larger labeled cohort to shrink the CIs (84 positives is the binding constraint).

---

## 2026-08-12 (Signal-feature artifacts) — significance, dispersion, per-epoch, quality, non-avg HRV

A run of work making the signal-feature outputs durable, complete, and presentable,
plus a same-harness benchmark of the NK2+report features vs the verified baseline.
All exports under `/data-temp/physio-viewer/exports/` (world-readable; teammates pull
over their own SSH), cataloged in `scripts/viewer/EXPORTS_MANIFEST.md`.

### Benchmark: NK2+report features vs baseline (bench harness) 🔬
Same folds, only the feature set changes (`bench/run_local_cv.py`, production model +
scoring). **Pooled LOSO reward 0.098 (baseline, 55 feat) → 0.176 (plus NK+report, 337
feat)** — nearly 2×, improving every held-out site (Kaiser flips −0.10→+0.42). ⚠️ The
baseline landed at **0.098, not the 0.168 anchor** — this harness ("production site-MoE
+ Kaiser + BMI-impute") is not identical to whatever produced 0.168; the *relative* lift
is trustworthy, the *absolute* number needs reconciling before quoting vs leaderboard.
70/15/15 was noisy (337-feat reward std > mean, AUROC slightly down) → the raw union
overfits; **feature selection is the next step** (borne out by the embedding pass above). 📌

### Univariate feature significance — `scripts/viewer/feature_significance.py` ✅ (committed ddb01dd)
Durable replacement for the ad-hoc `feat_auroc.py` (which only printed). Per feature:
AUROC, |AUROC−0.5| effect, direction, Mann-Whitney U + tie-corrected z + two-sided p
(via `math.erf`, no scipy), BH-FDR q. Grouped demographic / baseline / nk / report;
writes `.csv` (all) + `.md` (presentable per-group tables). 🔬 331 features on the
1090-rec cohort: age AUROC **0.772** (top), plmi 0.642, nk `hrv_sleep_sd1sd2` 0.637,
report `eeg_n1_theta_alpha` 0.636. Caught & fixed a baseline-contamination bug (the
nk/rep join mutated shared row dicts → snapshot `base_header` before the join).

### Non-avg / long / quality exports ✅ (committed 983d82c, dd3a2be, 6e4200d)
- `stage_dispersion.py` + `export_dispersion_features.py` — within-stage **SPREAD**
  (SD/CV/pXX) of the mean-collapsed per-epoch values. **1090 × 265 cols.** Recomputed
  mean matches the mean CSVs to 5e-5. 🔬
- `epoch_features.py` + `export_epoch_features.py` — **LONG** per-epoch tables in
  `exports/per_epoch/`: `epoch_eeg_standard.csv` **970,662 rows × 25** (one per
  recording×stage×epoch), `spindles_standard.csv` **222,111 rows × 11** (one per
  spindle). Verified the long table = averaged extractor under the same 180-epoch
  subsample (4.77e-5). 🔬
- `signal_quality.py` + `export_quality.py` — **NeuroKit signal-quality** scores +
  diagnostic plots. **1090 × 35 cols + 2,155 PNGs** (`nk.ecg_plot`/`nk.rsp_plot`).
  Cohort QC 🔬: ECG averageQRS mean **0.825** (71% ≥0.8, 0% unusable), RSP 0.753,
  ECG quality stable across stages; EEG/EOG electrically clean (≈0% NaN/clip).

### Fixes from the quality-CSV review ✅ (committed 40195ab)
- **`ecg_zhao_verdict` empty for all 1090:** NeuroKit's zhao2018 path calls
  `np.trapezoid` (NumPy ≥2.0), absent on the box's NumPy <2.0 → raised & swallowed.
  **Same bug silently blanks frequency-domain HRV** (lf/hf/lfhf/lfn/hfn/tp) in any
  fresh `nk_features` re-run. Fix: alias `np.trapezoid=np.trapz` at import in
  `signal_quality.py` + `nk_features.py`. 🔬 verified verdicts + 727/730 HRV windows
  now carry lfhf. (See memory `neurokit-numpy-trapezoid-gotcha`.)
- **EEG `flat_frac` over-sensitive:** old per-sample `|diff|<1e-6·MAD` collapsed to
  "consecutive samples byte-identical" → Emory's ~2 µV EEG quantizer read as ~70%
  flat. Fix: **windowed** SD-vs-active-level (1 s windows, flat if SD < **2% of p90**).
  Calibrated on cohort 🔬: live channels (Emory-quantized incl.) ≤0.005, a 4.5 h frozen
  Kaiser channel 0.68. p90 (not median) so a mostly-dead channel still flags.

### Non-avg per-stage HRV — `scripts/viewer/hrv_windows.py` + `export_hrv_windows.py` ✅ (committed 40195ab)
`nk_features` gives one *pooled* HRV number per stage; the dispersion family never
covered HRV. HRV is undefined per 30 s epoch, so the un-aggregated grain is a **~120 s
window of beats within a stage**. Long `hrv_windows_<cohort>.csv` (one row per
recording×stage×window) + wide `hrv_dispersion_<cohort>.csv` (per-stage SD/CV/pXX).
Reuses `nk_features`' exact R-peak detection + `_hrv_from_rr`, so a window's HRV
matches the pooled per-stage definition. 🔬 dispersion mean == long-row mean to 1e-4;
smoke ~230 windows/rec, 0 errors, ~4 s/rec. Full standard-cohort run + a quality
re-export (fixed zhao + flat_frac, scores only — plots unaffected) chained on the box.

---

## 2026-08-10 (Clinical report) — EEG spectral/spindles, hypoxic burden, resp events, Tier-1/2 report

Built the sleep-quality + clinical-marker feature families the team prioritised,
on top of the per-stage physiology below. Four new pure modules in
`scripts/viewer/` (scipy-based, degrade to None, never raise):

- `eeg_spectral.py` — per-stage **absolute + relative band power** (Welch PSD),
  **Theta/Alpha**, **Delta/Sigma**, **REM-slowing** `(δ+θ)/(α+σ+β)`, and
  **sleep-spindle** detection (11–16 Hz Butterworth→Hilbert→smoothed envelope,
  mean+2.5·SD threshold on pooled NREM, 0.5–3 s bursts with 0.3 s gap-merge →
  density/amp/dur in N2 & N3). 🔬 N2 density > N3 for all 3 test sites; rel_delta
  peaks in N3; REM slowing < N3 slowing — physiologically correct.
- `oxygenation.py` — **hypoxic burden** (Σ desat depth×duration per hour), ODI,
  T90, min/mean SpO₂, desat depth stats. ⚠️ **SpO₂ scale differs by site** —
  Emory stores a 0–1 fraction, BIDMC/Kaiser 0–100; `_normalise_spo2` auto-detects
  from the median and rescales, else a third of the cohort silently breaks. 🔬
  Emory's 0–1 SaO₂ correctly reads mean 95%.
- `resp_events.py` — apnea/hypopnea/RERA **counts**, AHI/RDI, **event durations**
  (mean/median/max from contiguous `resp_caisr`@1Hz runs), **post-event SpO₂
  overshoot** + **respiratory recovery time** (from the aligned SpO₂). 🔬 event
  durations 12–16 s mean, recovery 4–8 s, overshoot ~1.3–1.7 %.
- `clinical_report.py` — orchestrator (reads no files): assembles **Tier 1** (REM
  slowing, N3 SWA, N2 spindle density, hypoxic burden, fragmentation index),
  **Tier 2** (stage-HRV CV, NREM RMSSD, respiratory instability, REM
  eye-movement density via a filtered-EOG proxy), a clinical-feature table, and
  sleep-quality metrics (SE, WASO, sleep/REM/N3 latency, awakenings, TST).

Wired: `app.py` `/api/report` (lazy, decodes only ECG/EEG/EOG/SpO₂/effort + the
small resp/arousal/limb CAISR channels, ~8–14 s); a load-on-demand **Clinical
report** card at the end of the patient view in `index.html` (Tier tiles + table +
sleep-quality tiles with abnormal flagging + respiratory-event panel); 27 new
`glossary.py` entries under `GLOSSARY["report"]`. ✅ Verified end-to-end through the
live HTTP server on BIDMC/Emory/Kaiser; graceful degradation with missing channels;
bad bids → clean error.

⚠️ Known: Emory ECG peak detection sometimes yields nonsensical RMSSD (e.g. 636 ms)
— pre-existing `nk_features` behaviour, unchanged here. Spindle detector is
single-channel and undercounts vs expert scoring; *relative* differences are the
usable signal. 📌 Not yet: fold these into an enriched export CSV + benchmark the
model on them.

**Standard NK export (per-stage physiology) finished** ✅ →
`/data-temp/physio-viewer/exports/nk_features_standard.csv`, **1090 rows × 171
cols, 0 errors** (13 records had no physio file). World-readable for teammates.

---

## 2026-08-10 (NeuroKit per-stage physiology) — HRV / EEG complexity / respiration

New exploratory feature family: per-sleep-stage physiology from the raw waveforms
via **NeuroKit2** (already installed on pdmle; no pip / no firewall). Rationale:
CI is hypothesized to show as *blunted modulation across stages* more than in any
night-average — so features are stage-resolved and the flagship ones are
cross-stage contrasts.

### Timing constraints (measured on the box, no `numba`) — drove the whole design 🔬
- `hrv_time`/`hrv_frequency`: cheap (~0.03s/0.5s even at 8000 beats). USE.
- `hrv_nonlinear` and umbrella `nk.hrv()`: **O(n²)** — 15s @2k beats, times out
  past ~4k. A night's stage has 15k+ beats → NEVER call on full-night RR.
- `fractal_higuchi`: ~268s without numba → AVOID all numba-JIT complexity metrics.
- `entropy_sample` on the RR *interval* series is fast; on raw *signal* samples it
  stalls → EEG epochs are decimated to ≤1024 samples, ≤25 epochs/stage.
- ⇒ HRV uses only linear time+frequency; Poincaré SD1/SD2 added in closed form.

### Code (all in `scripts/viewer/`) ✅
- `nk_features.py` — `nk_stage_features()` → per-stage HRV (ECG), EEG sample/
  permutation entropy, respiratory rate/variability, + cross-stage contrasts
  (`rem_nrem_rmssd_ratio`, `wake_sleep_hr_delta`, `stage_hr_range`, …). Degrades to
  NaN/None, never raises. `flatten_features()`/`NK_FEATURE_COLUMNS` = 160 stable cols.
- `export_nk_features.py` — CLI mirroring `export_features.py` but reads the physio
  EDFs (one ECG + one central EEG + one effort channel only). 171 CSV cols. ~7-8
  s/recording. Resumable (`--resume`). Merge with `features_*.csv` on
  `(bids_folder, session)`.
- `app.py` — `/api/nk_features?bids=` (decodes only the 3 needed channels, ~5-10s).
- `glossary.py` — `NK_GLOSSARY` (19 entries), served under `GLOSSARY["nk"]`.
- `index.html` — **Per-stage physiology** card (lazy load-on-click button), per-stage
  tables (stages=columns) + cross-stage contrast tiles; **browser-tab favicon** added
  (`<link rel=icon>` → `/static/edwards_logo.png`).

### Verified 🔬
- End-to-end on real patients (BIDMC + Emory): HRV LF/HF rises Wake→REM, EEG SampEn
  lowest in N3 — physiologically sensible. 6/6 export rows: 0 errors, all 171 cols
  filled, stable schema (declared==produced), graceful degradation when ECG missing.
- Live server on a fresh port: favicon serves 200 image/png; `/api/nk_features` OK
  in ~5-10s; glossary serves the nk block.

### Running / follow-up 📌
- **Standard-cohort export running** (detached `setsid`) →
  `/data-temp/physio-viewer/exports/nk_features_standard.csv`, log
  `/tmp/nk_export_standard.log`. ~1103 recs × ~8s ≈ 2-2.5 h. `--resume` safe.
- Not yet: large-cohort export; benchmarking the traditional model on these features.

---

## 2026-08-06 (full-stage signal + AHI note) — Whole-stage zoomable trace

Follow-up asks: (Q1) do event indices change under preprocessing? (Q2) show the
*whole* of each stage as a zoomable signal, not just one example epoch.

### Q1 — why AHI/arousal/PLMI don't move (documented, not changed)
CAISR scores respiratory/arousal/limb events on their OWN channels (`resp_caisr`
etc.), independent of the stage channel, so preprocessing (which only smooths
stages) leaves event *counts* untouched. The viewer's index divides by *recording
hours* (staging-independent) → identical raw vs preprocessed, as observed.
Empirically confirmed on 3 patients: only if you use the clinical definition
(sleep-gated events ÷ TST) does it nudge (~2%, e.g. AHI 21.00→21.41) because a few
epochs move between Wake and sleep. Also noted a latent inconsistency: the viewer
(÷recording-hr) and export_features.py (÷TST) use different AHI denominators —
flagged to the user; left as-is pending their call (colleagues may train on the
CSV convention).

### Q2 — full-stage concatenated signal — `stage_signals.stage_concat_signal`
Joins every epoch of one stage end-to-end into a single trace on its own 0..(total
stage minutes) timeline; `/api/stage_signals?ch=&stage=(&t0=&t1=)` returns it
windowed at full resolution (same server-side re-slice zoom as `/api/signals`).
Returns a bout position map (concat position + real night time + epoch count) so
the UI draws dashed **seam** lines at night-discontinuities and the hover reports
both concat-time and true night-time.
- Frontend: **generalized `attachZoom`** to take onWindow/onReset/onZoom callbacks
  (PSG plots and the new full-stage plot now share one zoom controller; PSG call
  site rewired, no regression). New `fullStageSection`/`drawFullStage`/`fsSetWindow`
  /`fsZoom`/`fullStageSVG` with their own FS_STATE + a race guard, separate button
  ids (#fszin/#fszout/#fsreset) from the PSG bar.
- 🔬 Verified on a real patient: Wake 85.5 min/8 bouts/171 epochs → 2500-pt
  overview; 10 s zoom → 2000 raw samples; bout map night-times correct; zero
  NaN/Inf; valid JSON. Unit-tested seam/clamp/absent-stage/NaN edge cases.

---

## 2026-08-06 (dynamics + per-stage signals) — Preprocessed dynamics & signal-by-stage

Two requests: (1) show the sleep-dynamics matrices/stats for the **preprocessed**
staging too, not just raw; (2) chunk each patient's biosignals by sleep stage and
show per-stage detail + averages.

### Sleep dynamics on preprocessed staging
`/api/dynamics` now returns both `raw` and `clean` dynamics (`clean` computed on
the spike-smoothed hypnogram from `preprocess.smooth_stages`, same cohort
baseline). The dynamics card gained **Preprocessed / Raw tabs** (defaults to
Preprocessed) that re-render both heatmaps + all fragmentation/transition tiles
for the chosen view; top-level fields stay == raw for back-compat.
- 🔬 Verified on a real patient: raw single-epoch spikes 15 → 0 after smoothing;
  stage-shift index 15.18 → 7.63 /h.

### Signal by sleep stage — `scripts/viewer/stage_signals.py` (new)
Aligns one channel to the preprocessed per-epoch staging (epoch i =
samples[i·spe:(i+1)·spe], spe = round(fs·30); shorter of signal/staging wins) and
per stage produces: amplitude stats (mean±SD, 5–95%, minutes/epochs/%), a
**representative 30 s example epoch** (middle of that stage's longest bout,
min/max-decimated), and — EEG only — **mean relative band power** (delta/theta/
alpha/sigma/beta via per-epoch rFFT, gated to fs ≥ 60 Hz). New endpoint
`/api/stage_signals?ch=…` (header-only channel list when `ch` omitted; reads the
big physio EDF lazily, one channel). New card `stageSignalCard` with per-stage
stat tiles, band-power bars, and framed example-epoch plots.
- Suggested + added the EEG band-power panel as the meaningful per-stage EEG
  "average" (a time-domain average of unlocked oscillations cancels to ~0).
- 🔬 Verified: synthetic-signal band dominance is correct per stage; real C4-M1
  gives 5 stages, delta-dominant deep sleep, N3 highest EEG variance, zero
  NaN/Inf, valid JSON; non-EEG channels (EKG/SaO2) correctly omit band power.
- Pure numpy (no SciPy), so it runs in the same minimal viewer environment.

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
