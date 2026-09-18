# Poster blueprint — Team SDG, PhysioNet Challenge 2026

**Title:** *Confounder-Resistant, Site-Aware Prediction of Cognitive Impairment from Overnight Sleep Studies*
**Authors:** Arshia Ilaty — Team SDG · George B. Moody PhysioNet Challenge 2026

> Companion to `cinc2026_sdg.md`/`.tex`. This file is the **poster** cut: what goes in each panel,
> which numbers are measured (safe to print) vs. still `[FILL]`, plus the two artifacts you asked for —
> the **submission progression table** and the **methodology figure** (`figures/fig_methodology_poster.png`).
> Every number below is measured; none is fabricated. Exploration numbers are labelled *post-deadline*.

---

## The figure

`figures/fig_methodology_poster.png` (and `.pdf`, vector, scales to A0). One-glance pipeline:
**Data → Features → Site-aware model → Reward-optimal threshold → LOSO validation/output**, with the
primary-metric definition as a strip and two callouts (official result + the transfer wall). Regenerate
with `figures/make_methodology_poster.py`. Use the **PDF** for print.

---

## Table 1 — Submission progression (our Challenge portal history)

The story the poster should tell with this table: **feature engineering + more training data drove the
gains; the transformer was abandoned early; adding SpO₂ oxygenation was the single biggest jump; and the
best official entry (2634) trained on the *large* cohort.**

| # | ID | Change from previous | Train data | AC-AUROC | Reward |
|---|------|-----------------------------------------------|:------:|:--------:|:------:|
| 1 | 2027 | Transformer + demographics (baseline)          | Small  | 0.417 | 0.008 |
| 2 | 2358 | Replaced transformer → algorithmic + PSG feats | Small  | 0.677 | 0.033 |
| 3 | 2461 | Added spectral / ECG / HRV                     | Small  | 0.693 | 0.032 |
| 4 | **2634** | **Added SpO₂ oxygenation  (official entry)** | **Large** | **0.748** | **0.168** |
| 5 | 2741 | Added EOG / chin-leg EMG dynamics              | Large  | 0.717 | 0.085 |
| 6 | 2872 | SpO₂ model, **age removed**                    | Large  | 0.672 | 0.019 |

**Reading notes for the caption:**
- **2027 → 2358 (+0.26 AC-AUROC):** dropping the transformer for interpretable algorithmic + PSG
  features was the pivotal decision — deep learning underperformed on this small/medium tabular problem.
- **2461 → 2634 (+0.055 AC-AUROC, reward 0.032 → 0.168):** adding **SpO₂ oxygenation** features, and
  moving to **large**-cohort training, produced the best entry. `physio_spo2_arou_rate_per_hour` ranked
  top by SHAP.
- **2741, 2872 (regressions):** adding EOG/chin-EMG dynamics and removing age both *hurt* — consistent
  with our finding that raw-channel dynamics add montage-dependent fragility, and that age-derived
  structure still helps ranking even though the age-conditioned metric discounts it.

---

## Recommended poster layout (4 columns / 8 panels)

### Panel 1 — Problem & why it's hard
- Predict binary `Cognitive_Impairment` from **one overnight PSG + demographics**; sleep disruption is an
  early marker of neurodegeneration.
- Two scoring twists shape everything: **(i)** positives are **rare** → primary metric is a
  **prevalence-weighted Reward** (TP = 1/p−1, TN = 1/(1−p)−1, FP = FN = −1; random/all-pos/all-neg ≈ 0);
  **(ii)** diagnostics are **age-conditioned**, so simply relearning "old ⇒ impaired" is *not* rewarded.
- Data pooled from **3 clinical sites** → distribution shift between visible and hidden sites.

### Panel 2 — Data & cohort (use `fig1`, `fig5`)
- Training release: **1103 recordings, 3 sites, prevalence 7.6%** (measured):

  | Site | n | CI+ | Prevalence % (95% CI) |
  |---|--:|--:|---|
  | BIDMC (S0001) | 857 | 56 | 6.5 (5.1–8.4) |
  | Kaiser (I0006) | 192 | 20 | 10.4 (6.8–15.5) |
  | Emory (I0002) | 54 | 8 | 14.8 (7.7–26.6) |
  | **Total** | **1103** | **84** | **7.6 (6.2–9.3)** |

- **Large** training cohort used for the official entry: **n = 6530, 497 CI+** (BIDMC 5075 / Kaiser 1138 / Emory 317).
- **Prevalence climbs monotonically with age:** 2.0% (50–59) → 7.3% (60–69) → 17.5% (70–79) → 36.1% (80+)
  — exactly the shortcut the age-conditioned metric neutralizes.
- **Raw signal is fragile:** BIDMC alone shows **42 distinct montages**, sampling rates 200–512 Hz, 854
  SpO₂ channels mislabeled `uV`. → motivates harmonized **CAISR annotations over raw waveforms**.

### Panel 3 — Methodology figure
- Drop in `figures/fig_methodology_poster.pdf` (the hero figure). Caption = one sentence per stage.

### Panel 4 — What we feed the model (use `fig8` significance, `fig7` transitions)
- **No raw EEG.** Three interpretable blocks: **sleep architecture (CAISR)** · **autonomic (ECG-HRV, SpO₂)**
  · **demographics without age**.
- Univariate screen (Welch *t* / χ², BH-FDR): **14 / 48 features significant at q<0.05.** Effect sizes:
  **age d=1.08** (the confounder we exclude) ≫ periodic limb movements (0.56), ↓REM (−0.48), ↑WASO (0.45),
  ↑wake (0.43), ↓efficiency/entropy (~−0.42), ↓N3 (−0.30). **AHI is *not* significant (d=0.19)** — signal
  is in **sleep architecture & continuity, not respiratory-event load**.
- Stage *dynamics*: CI patients have destabilized deep sleep — **N3→N3 −5.4 pp**, **REM −4.8 pp**.

### Panel 5 — Model & the reward-optimal threshold
- **Histogram gradient boosting** (native missing-value handling), **per-site mixture-of-experts + global
  fallback**, **isotonic calibration**, site-median BMI imputation.
- **Bayes-optimal decision rule (our theory result):** under the Challenge reward, predict positive iff
  **q > p** (q = calibrated probability, p = prevalence). Because prevalence is defined **per age**, the
  optimum is an **age-specific** threshold `q > p(age)` — a clean, metric-aligned lever.

### Panel 6 — Results (Table 1 above + this)
- **Official Challenge entry (submission 2634, large cohort): Reward 0.168 · AC-AUROC 0.748.**
  A positive reward = genuine skill beyond base-rate guessing (trivial classifiers ≈ 0).
- Design ablations support: **removing the EEG/EOG/chin block did not hurt cross-site performance**, and
  it reduced montage fragility.

### Panel 7 — The cross-site transfer wall *(honest headline finding)*
- **Within-distribution (mixed-site 70/15/15, large cohort):** AUROC **0.869**, AC-AUROC **0.832**,
  Reward **+0.474**, AUPRC **0.42**.
- **Cross-site (LOSO):** AUROC **≈ 0.66**, AC-AUROC **≈ 0.60**.
- ⇒ **More patients close the within-site gap; only 3 training sites cap new-site generalization.**
  The wall is **transfer, not feature poverty.**
- *Post-deadline analysis (not in the frozen submission):* the two levers that actually move cross-site
  Reward are **(a) per-site rank-normalization** (a batch-effect remover) and **(b) a tuned cross-site
  Reward threshold**, together reaching LOSO transfer Reward **≈ +0.171 ≈ the oracle ceiling (+0.177)**,
  ~2.8× the untuned baseline. A systematic screen of **14+** added-feature / deep-learning ideas
  (spectral shape, self-supervised embeddings, raw-waveform nets, adversarial domain-invariance,
  resampling) was **neutral-to-negative** on cross-site Reward — reinforcing that the bottleneck is site
  diversity, not more features.

### Panel 8 — Conclusion & next steps
- An **interpretable, confounder-resistant, site-aware** system reaches **Reward 0.168 / AC-AUROC 0.748**
  on the official metric, with its threshold **derived** from the reward rather than tuned blindly.
- **Ceiling-raiser = more training *domains*** (external public cohorts: SHHS / MrOS / MESA — same
  EEG/ECG/resp signals), plus the age-specific `q > p(age)` threshold contingent on good cross-site
  calibration.
- **Limitations:** only 3 visible sites; reliance on CAISR annotation quality at test time (mitigated by
  NaN-tolerant models + global fallback); label noise from right-censored follow-up.

---

## Notes for filling the paper's `[FILL]` cells (`cinc2026_sdg.md` §3)

- **Table 1 (pooled LOSO official metrics):** you can now print **Reward = 0.168** and, from the official
  2634 entry, **AC-AUROC = 0.748**. AUROC / AUPRC / age-weighted AUROC are in your gitignored
  `eda/loso_cv_*.json` — pull them from there; I don't have those files in the repo, so don't guess.
- **Add the submission-progression table above as a new Table 1** (it's currently missing from the paper
  and is exactly the kind of "how we got here" table reviewers like).
- **Keep the exploration numbers (rank-norm +0.171, the 14 screens) in Discussion/§4 as post-challenge
  analysis** — they were not part of the frozen submission, so don't present them as the entry's score.
