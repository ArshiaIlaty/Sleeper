"""Render the end-to-end methodology figure for the paper / onboarding.

A schematic (not a data chart) of Team SDG's full pipeline, from raw PSG to the
prevalence-weighted Reward, drawn with the validated figstyle palette. It is a
teaching diagram: every stage is annotated with the concrete artefact it produces
(cohort sizes, feature counts, the 2560-d embedding block) and the leakage-safe
fold boundary is called out explicitly.

Numbers shown are the ones verified this cycle:
  cohort 1103 recordings, 84 CI+ (7.6%), 3 sites; plus-cache 337 cols (336 w/o age);
  SleepFM embeddings 2560-d = 5 stages x 512, 1080/1103 aligned;
  quality 1090x35, hrv_windows 236100x27, hrv_dispersion 1090x183;
  LOSO reward baseline(no age) 0.145 -> +AUC>0.6 0.160 -> +SleepFM+AUC 0.253,
  paired (emb+AUC - AUC-only) delta = -0.10 [-1.32,+0.66], 9/20 wins  (NOT significant).

Pure numpy + matplotlib; depends on figstyle for the palette + rc.
"""
import os
import figstyle as fs
from figstyle import (RED, BLUE, GOLD, AQUA, VIOLET, INK, INK2, MUTED, GRID,
                      SURFACE)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIGDIR = os.path.join(REPO, "paper", "figures")


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), bbox_inches="tight",
                    dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


def _lighten(hex_color, f=0.88):
    """Blend a hex color toward white by fraction f (0=color, 1=white)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * f); g = int(g + (255 - g) * f); b = int(b + (255 - b) * f)
    return f"#{r:02x}{g:02x}{b:02x}"


def box(ax, x, y, w, h, title, body, accent, body_size=8.4, title_size=9.6):
    """A stage card: light accent fill, a solid accent left-rule, bold title, body."""
    from matplotlib.patches import FancyBboxPatch, Rectangle
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.9",
        linewidth=1.0, edgecolor=accent, facecolor=_lighten(accent, 0.90),
        mutation_aspect=1.0, zorder=3))
    ax.add_patch(Rectangle((x, y), 0.7, h, facecolor=accent, edgecolor="none",
                           zorder=4))
    ax.text(x + 1.6, y + h - 1.9, title, fontsize=title_size, fontweight="bold",
            color=INK, va="top", ha="left", zorder=5)
    if body:
        ax.text(x + 1.6, y + h - 4.6, body, fontsize=body_size, color=INK2,
                va="top", ha="left", zorder=5, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=1.6, style="-|>"):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=13,
        linewidth=lw, color=color, zorder=2,
        shrinkA=0, shrinkB=0))


def main():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fs.apply_style()

    fig, ax = plt.subplots(figsize=(12.4, 15.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(0, 99.2, "Team SDG — Cognitive-Impairment prediction from overnight PSG",
            fontsize=15, fontweight="bold", color=INK, va="top", ha="left")
    ax.text(0, 96.4, "PhysioNet/CinC Challenge 2026  ·  end-to-end methodology  ·  "
            "primary metric = prevalence-weighted Reward",
            fontsize=10, color=MUTED, va="top", ha="left")

    # ---------------------------------------------------------------- (1) DATA
    box(ax, 2, 88.3, 96, 7.0, "①  DATA   —   1103 overnight PSG recordings · 84 CI-positive (7.6%) · 3 sites",
        "Per recording:  EEG · EOG · EMG · ECG · respiration · SpO₂   +   demographics (age, sex, BMI)   +   CAISR stages (W/N1/N2/N3/REM).\n"
        "Sites:  BIDMC (S0001)   ·   Emory (I0002, coarse ~2µV EEG ADC, SpO₂ as 0–1 fraction)   ·   Kaiser (I0006).",
        BLUE, body_size=8.6)

    # ---------------------------------------------------------------- (2) DECODE + QC
    box(ax, 2, 76.5, 47, 9.5, "②  DECODE  +  SIGNAL QC   (NeuroKit2)",
        "Site-aware channel decode (SpO₂ scale auto-detect).\n"
        "QC:  ECG avgQRS + zhao2018 verdict  ·  RSP per-cycle  ·\n"
        "EEG/EOG model-free sanity (nan / flat / clip fraction).\n"
        "→ quality_standard.csv  (1090 × 35)",
        GOLD, body_size=8.2)

    box(ax, 51, 76.5, 47, 9.5, "◆  WHY no NK quality for EEG/EOG",
        "NeuroKit scores only signals with a repeating template\n"
        "it can model (ECG beats, RSP cycles). EEG is broadband,\n"
        "EOG is sparse saccades → no native score, so those get\n"
        "the flat / clip / nan sanity check instead.",
        MUTED, body_size=8.2)

    # ---------------------------------------------------------------- (3) FEATURES
    box(ax, 2, 55, 47, 17.5, "③  Hand-crafted features",
        "NK2 per-stage:  HRV (SDNN/RMSSD/LF-HF…) · EEG band-power ·\n"
        "                          respiration · spindles.\n"
        "Report-derived clinical features.\n"
        "Within-stage DISPERSION (SD/CV/pXX, non-averaged).\n"
        "Non-avg HRV over 120 s windows per stage:\n"
        "     hrv_windows 236 100 × 27  ·  hrv_dispersion 1090 × 183.\n"
        "Per-epoch LONG tables (EEG ~970k rows · spindles ~222k).\n"
        "\n"
        "→ WIDE feature matrix:  337 cols  (336 without age)",
        BLUE, body_size=8.0)

    box(ax, 51, 55, 47, 17.5, "③  SleepFM embeddings   (self-supervised)",
        "Contrastive foundation model over EEG/EOG/EMG, ECG,\n"
        "respiration.  NOT trained on CI labels → no label leakage\n"
        "from the embedder.  (preprocessed by collaborator)\n"
        "\n"
        "Per recording:  2560-d vector = 5 stages × 512-d.\n"
        "Missing-stage blocks zeroed.\n"
        "Aligned to cohort by subject id:  1080 / 1103.",
        VIOLET, body_size=8.0)

    # ---------------------------------------------------------------- leakage-safe band
    ax.add_patch(FancyBboxPatch(
        (2, 19.5), 96, 30.5, boxstyle="round,pad=0.01,rounding_size=1.2",
        linewidth=1.3, edgecolor=RED, facecolor="none", linestyle=(0, (5, 3)),
        zorder=1))
    ax.text(5.0, 49.6, "leakage-safe  —  all transforms below fit IN-FOLD (train rows only)",
            fontsize=8.6, style="italic", color=RED, va="top", ha="left", zorder=6)

    # ---------------------------------------------------------------- (4) FUSION + SELECT
    box(ax, 4, 39.0, 90, 9.3, "④  FUSION  +  SELECTION   (fit on training fold only)",
        "Drop the age column (strong CI proxy).   Concatenate hand-crafted  +  StandardScaler → PCA-32 on the embedding block.\n"
        "Univariate filter:  keep columns with train-set AUROC > 0.6.\n"
        "In-fold fitting is essential — at 84 positives, in-sample selection inflates test AUROC by +0.05–0.10.",
        GOLD, body_size=8.2, title_size=10.2)

    # ---------------------------------------------------------------- (5) MODEL
    box(ax, 4, 29.5, 90, 7.0, "⑤  MODEL   (production stack, feature_prep.py)",
        "HistGradientBoosting  site mixture-of-experts   +   Kaiser fine-tune   +   BMI imputer.\n"
        "Per-site expert routing with a Kaiser-specific fine-tuned head; missing BMI imputed per site.",
        AQUA, body_size=8.2, title_size=10.2)

    # ---------------------------------------------------------------- (6) EVAL
    box(ax, 4, 20.2, 90, 8.6, "⑥  EVALUATION",
        "Reward thresholds fit on validation (site×decade).   Primary metric = prevalence-weighted Reward.\n"
        "Also:  age-conditioned & age-weighted AUROC, AUPRC, F1, accuracy.\n"
        "Splits:  70/15/15 balanced on label×site×sex×age-bin (fast)   +   LOSO (honest site generalization).",
        INK2, body_size=8.2, title_size=10.2)

    # ---------------------------------------------------------------- results callout
    box(ax, 2, 3.5, 96, 15.0, "RESULTS   (LOSO pooled reward @ prevalence, bootstrap 95% CI)",
        "baseline, no age (336):        0.145  [−0.10, +0.45]\n"
        "  + AUROC>0.6 selection:       0.160  [−0.09, +0.45]        ← selection alone\n"
        "  + SleepFM PCA-32 + AUC>0.6:  0.253  [−0.01, +0.56]        ← best point estimate\n"
        "\n"
        "⚠  Confirmatory paired test (identical 70/15/15 splits, 20 seeds):  Δreward (emb+AUC) − (AUC-only) = −0.10  [−1.32, +0.66],  wins 9/20.\n"
        "     The embedding CI straddles 0 → the SleepFM gain is NOT yet distinguishable from the AUROC>0.6 regularization.  Do not claim +0.253 as a win.",
        RED, body_size=8.2, title_size=10.4)

    # ---------------------------------------------------------------- arrows
    arrow(ax, 50, 88.3, 50, 86.2, color=BLUE)            # data -> decode band
    arrow(ax, 25.5, 76.5, 25.5, 72.7, color=GOLD)        # decode -> hand-crafted
    arrow(ax, 25.5, 55, 42, 48.5, color=BLUE)            # hand-crafted -> fusion
    arrow(ax, 74.5, 55, 58, 48.5, color=VIOLET)          # embeddings -> fusion
    arrow(ax, 50, 39.0, 50, 36.7, color=GOLD)            # fusion -> model
    arrow(ax, 50, 29.5, 50, 28.7, color=AQUA)            # model -> eval
    arrow(ax, 50, 20.5, 50, 18.7, color=INK2)            # eval -> results

    _save(fig, "fig9_methodology")
    print("done.")


if __name__ == "__main__":
    main()
