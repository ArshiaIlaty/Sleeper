"""Append an "Our approach & results" section to the Scoring Metrics deck.

Adds one section-header slide plus five content slides:
  1. Pipeline overview  — the methodology figure (fig9_methodology)
  2. Feature engineering — the 3-family feature stack (fig_slide_features)
  3. The model          — site mixture-of-experts + Kaiser fine-tune + BMI impute
  4. Internal results   — LOSO reward baseline vs +engineered (fig_slide_benchmark)
  5. Foundation model   — honest SleepFM fusion forest plot (fig_slide_forest)

Reuses the styling helpers in build_dataset_slides.py so the new slides match the
Edwards template exactly, with speaker notes on every slide. Inserts the section
right after "What this means for our approach" (the takeaways bridge), before the
References slide. Idempotent: refuses to double-insert. Backs the deck up first.

All numbers are verified (see PROJECT_LOG.md 2026-08-12 + git anchor Reward 0.168);
figures are the vetted PNGs in paper/figures/.

    python3 scripts/slides/build_approach_slides.py
"""
import os
import shutil
import argparse

from pptx import Presentation
from pptx.util import Inches
from pptx.enum.text import PP_ALIGN

import build_dataset_slides as B   # same directory: styling + helpers


SECTION_TITLE = "Our approach & results"


# ---- slide builders -----------------------------------------------------------
def slide_section(prs):
    s = prs.slides.add_slide(prs.slide_layouts[B.LAYOUT_SECTION])
    B.set_title(s, SECTION_TITLE)
    B.add_notes(s,
        "We've seen the metric and the data it scores. Now here is what Team SDG "
        "actually built to win on that metric — the full pipeline from a raw "
        "overnight recording to a calibrated impairment probability, the features "
        "we engineered, the model, and where our numbers stand today, including "
        "one experiment that hasn't paid off yet.")
    return s


def slide_pipeline(prs, figdir):
    s = prs.slides.add_slide(prs.slide_layouts[B.LAYOUT_CONTENT])
    B.set_title(s, "From raw PSG to a calibrated risk score")
    B.add_eyebrow(s, "APPROACH · END-TO-END PIPELINE")
    # wide landscape pipeline diagram across the content area
    B.fit_picture(s, os.path.join(figdir, "fig_slide_pipeline.png"),
                  0.55, 1.55, 7.1, 2.9, halign="center", valign="top")
    B.add_text(s, 0.6, 4.55, 7.1, 0.85,
               [[("Two feature streams fuse leakage-safe", True, B.RED, 12),
                 (" (PCA & selection fit in-fold), feed a ", False, B.INK2, 12),
                 ("per-site mixture-of-experts", True, B.INK, 12),
                 (" with a Kaiser fine-tune, and a Reward-optimal threshold; "
                  "validated leave-one-site-out.", False, B.INK2, 12)]],
               base=12)
    B.add_footer(s)
    B.add_notes(s,
        "This is the whole system on one slide. We start from the raw overnight "
        "recording — EEG, ECG, respiration, oxygen — decode every channel with "
        "NeuroKit2 in a site-aware way, and run quality checks. Then two parallel "
        "feature streams: our hand-crafted sleep physiology, and self-supervised "
        "embeddings from a foundation model. Everything downstream of the split — "
        "the dimensionality reduction and feature selection — is fit only on "
        "training rows; with just 84 positives, fitting those on the whole set "
        "would inflate our scores by five to ten points, so we are strict about "
        "it. The model is a per-site mixture of experts with a fine-tuned head for "
        "Kaiser. Finally we tune the decision threshold to maximize the Reward, by "
        "site and age decade, and we validate leave-one-site-out so our numbers "
        "reflect a genuinely unseen hospital.")
    return s


def slide_features(prs, figdir):
    s = prs.slides.add_slide(prs.slide_layouts[B.LAYOUT_CONTENT])
    B.set_title(s, "337 engineered features across three families")
    B.add_eyebrow(s, "APPROACH · FEATURE ENGINEERING")
    B.fit_picture(s, os.path.join(figdir, "fig_slide_features.png"),
                  0.55, 1.75, 7.1, 2.55, halign="center", valign="top")
    B.add_text(s, 0.6, 4.4, 7.1, 0.95,
               [[("Stage-resolved by design", True, B.RED, 12),
                 (" — cognitive impairment shows as blunted modulation across "
                  "sleep stages, so features are computed per stage and the "
                  "flagship ones are cross-stage contrasts. ", False, B.INK2, 12),
                 ("Age is excluded from training.", True, B.INK, 12)]],
               base=12)
    B.add_footer(s)
    B.add_notes(s,
        "Our features come in three families. The baseline set — 55 features — is "
        "the CAISR sleep architecture and demographics: how the night is "
        "structured, fragmentation, event indices. On top of that we added 160 "
        "NeuroKit2 features: per-stage heart-rate variability, EEG complexity, and "
        "respiration, computed from the raw waveforms. And 122 clinical-report "
        "features: EEG spectral power and spindles, oxygenation and hypoxic "
        "burden, respiratory events. The design principle is that impairment "
        "shows up as blunted modulation across sleep stages, so almost everything "
        "is stage-resolved, and the strongest features are contrasts between "
        "stages. Crucially, we exclude age from training — the metric strips the "
        "age shortcut, so our features have to carry signal within an age band.")
    return s


def slide_model(prs):
    s = prs.slides.add_slide(prs.slide_layouts[B.LAYOUT_CONTENT])
    B.set_title(s, "A model built for site shift and class imbalance")
    B.add_eyebrow(s, "APPROACH · THE MODEL")
    cards = [
        ("Site mixture-of-experts",
         "A HistGradientBoosting classifier per site (BIDMC · Kaiser · Emory) "
         "with a global fallback — so each hospital's montage & population is "
         "modeled on its own terms."),
        ("Kaiser fine-tune",
         "A dedicated fine-tuned head for the Kaiser site, the hardest to "
         "generalize to, layered on top of the shared model."),
        ("BMI imputation",
         "BMI is missing for 76% of patients, so it is imputed per site rather "
         "than dropped — no feature is allowed to silently vanish."),
        ("Reward-optimal threshold",
         "Predict positive when the probability clears the age-specific "
         "prevalence — the Bayes-optimal rule for the prevalence-weighted "
         "Reward, tuned by site & decade on validation."),
    ]
    x0, y0, cw, ch, gapx, gapy = 0.6, 1.65, 3.45, 1.6, 0.2, 0.22
    for i, (head, body) in enumerate(cards):
        cx = x0 + (i % 2) * (cw + gapx)
        cy = y0 + (i // 2) * (ch + gapy)
        B.add_card(s, cx, cy, cw, ch, B.CARD)
        B.add_text(s, cx + 0.18, cy + 0.12, cw - 0.36, ch - 0.24,
                   [[(head, True, B.BLUE, 13)],
                    [(body, False, B.INK2, 10.5)]], base=10.5)
    B.add_footer(s)
    B.add_notes(s,
        "The model is designed around the two problems we keep coming back to: "
        "site shift and class imbalance. First, it is a mixture of experts — one "
        "gradient-boosted classifier per site, with a global fallback — because "
        "the three hospitals have such different montages and populations that a "
        "single model blurs them. Second, a dedicated fine-tuned head for Kaiser, "
        "which is the hardest site to generalize to. Third, because BMI is missing "
        "for three-quarters of patients, we impute it per site instead of dropping "
        "it. And the piece that directly targets the metric: instead of the usual "
        "0.5 cutoff, we predict positive when the probability exceeds the "
        "age-specific prevalence — that's the Bayes-optimal threshold for the "
        "Reward — tuned per site and age decade on held-out validation.")
    return s


def slide_results(prs, figdir):
    s = prs.slides.add_slide(prs.slide_layouts[B.LAYOUT_CONTENT])
    B.set_title(s, "Engineered features nearly doubled the internal reward")
    B.add_eyebrow(s, "RESULTS · INTERNAL LOSO BENCHMARK")
    B.fit_picture(s, os.path.join(figdir, "fig_slide_benchmark.png"),
                  0.5, 1.6, 4.0, 3.55, halign="left", valign="top")
    bullets = [
        [("+ NeuroKit2 & report features", True, B.INK, 12),
         (" lift pooled LOSO reward ", False, B.INK2, 12),
         ("0.098 → 0.176", True, B.RED, 12),
         (" on the same harness — improving every held-out site.",
          False, B.INK2, 12)],
        [("Reward 0.168 is the verified submission", True, B.INK, 12),
         (" (official evaluate_model.py).", False, B.INK2, 12)],
        [("The raw 337-feature union overfits", True, B.RED, 12),
         (" — feature selection is the next lever (below).",
          False, B.INK2, 12)],
        [("Caveat:", True, B.INK, 12),
         (" this internal harness scores the baseline at 0.098, not 0.168 — the "
          "relative lift is trustworthy; the absolute number needs reconciling "
          "before quoting vs the leaderboard.", False, B.INK2, 11)],
    ]
    y = 1.7
    for b in bullets:
        B.add_text(s, 4.7, y, 3.0, 0.95, [b], base=12)
        y += 0.85
    B.add_footer(s)
    B.add_notes(s,
        "Here is where the engineered features get us. On our internal "
        "leave-one-site-out harness — same folds, same model, only the feature "
        "set changes — adding the NeuroKit2 and clinical-report features lifts the "
        "pooled reward from 0.098 to 0.176, nearly double, and it improves every "
        "single held-out site. For reference, 0.168 is our verified submission "
        "score from the official scoring code. One honest caveat: this internal "
        "harness scores the baseline at 0.098, not at the 0.168 we submitted, so "
        "the two aren't the identical pipeline — the relative lift is what we "
        "trust, and the absolute number still needs reconciling before we quote it "
        "against the leaderboard. The other lesson: throwing all 337 features in "
        "raw actually overfits, which points straight at feature selection as the "
        "next lever — and that's the last slide.")
    return s


def slide_foundation(prs, figdir):
    s = prs.slides.add_slide(prs.slide_layouts[B.LAYOUT_CONTENT])
    B.set_title(s, "Can a foundation model add more? Not yet.")
    B.add_eyebrow(s, "RESULTS · SLEEPFM EMBEDDING FUSION (in progress)")
    B.fit_picture(s, os.path.join(figdir, "fig_slide_forest.png"),
                  0.5, 1.58, 4.45, 3.2, halign="left", valign="top")
    bullets = [
        [("SleepFM embeddings", True, B.INK, 12),
         (" (self-supervised, no CI labels) fused leakage-safe with our features.",
          False, B.INK2, 12)],
        [("Best point estimate 0.253", True, B.RED, 12),
         (" LOSO — but its 95% CI still includes 0.", False, B.INK2, 12)],
        [("Paired test is the verdict:", True, B.INK, 12),
         (" adding embeddings on top of feature selection gives ",
          False, B.INK2, 12),
         ("Δ = −0.10 [−1.32, +0.66]", True, B.INK, 12),
         (" — not distinguishable from selection alone.", False, B.INK2, 12)],
        [("Next:", True, B.BLUE, 12),
         (" per-stage PCA, late-fusion/stacking, and more labeled data to "
          "shrink the CIs (84 positives is the binding constraint).",
          False, B.INK2, 12)],
    ]
    y = 1.66
    for b in bullets:
        B.add_text(s, 5.1, y, 2.6, 1.0, [b], base=11.5)
        y += 0.86
    B.add_footer(s)
    B.add_notes(s,
        "I want to end with an experiment that is still open, because it shows how "
        "we work. A collaborator produced embeddings from SleepFM, a self-"
        "supervised foundation model that was never trained on impairment labels, "
        "so there's no leakage. We fused them with our features the leakage-safe "
        "way and benchmarked. The best configuration reaches a reward of 0.253 "
        "leave-one-site-out — which looks like a big jump. But look at the "
        "confidence intervals: every one of them still includes zero. And the "
        "decisive test is the paired comparison — on identical splits, adding the "
        "embeddings on top of plain feature selection changes the reward by "
        "minus 0.10, with a confidence interval from minus 1.3 to plus 0.66. In "
        "other words, we cannot yet distinguish the embeddings' contribution from "
        "the feature selection alone. So we are not claiming this as a win. The "
        "binding constraint is that 84 positives makes every interval wide; the "
        "next steps are richer per-stage pooling, stacking, and more labeled data.")
    return s


# ---- insertion ----------------------------------------------------------------
def already_present(prs):
    for s in prs.slides:
        if s.shapes.title is not None and \
           (s.shapes.title.text or "").strip() == SECTION_TITLE:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pptx", default="PhysioNet Challenge 2026 - Scoring Metrics.pptx")
    ap.add_argument("--figures", default="paper/figures")
    ap.add_argument("--backup", default="/tmp/ScoringMetrics.before_approach.pptx")
    args = ap.parse_args()

    prs = Presentation(args.pptx)
    if already_present(prs):
        raise SystemExit("Approach section already present — nothing to do.")

    anchor = B._title_index(prs, "What this means for our approach") \
        if hasattr(B, "_title_index") else None
    if anchor is None:
        # fall back to a local title scan
        anchor = None
        for i, s in enumerate(prs.slides):
            if s.shapes.title is not None and \
               (s.shapes.title.text or "").strip() == "What this means for our approach":
                anchor = i
                break
    if anchor is None:
        raise SystemExit("Could not find the takeaways anchor slide.")

    shutil.copyfile(args.pptx, args.backup)
    print(f"backed up original -> {args.backup}")

    figdir = args.figures
    builders = [
        lambda: slide_section(prs),
        lambda: slide_pipeline(prs, figdir),
        lambda: slide_features(prs, figdir),
        lambda: slide_model(prs),
        lambda: slide_results(prs, figdir),
        lambda: slide_foundation(prs, figdir),
    ]
    for b in builders:
        b()

    B.reorder_after(prs, len(builders), anchor)
    prs.save(args.pptx)
    print(f"Added {len(builders)} slides after index {anchor}; "
          f"deck now has {len(prs.slides)} slides.")


if __name__ == "__main__":
    main()
