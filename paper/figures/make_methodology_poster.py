#!/usr/bin/env python3
"""Poster-ready methodology figure for the Team SDG PhysioNet Challenge 2026 entry.

A horizontal pipeline (Data -> Features -> Site-aware model -> Reward-optimal
threshold -> LOSO validation/output) with two result callouts: the official
frozen submission (2634: Reward 0.168 / AC-AUROC 0.748) and the cross-site
transfer wall (within-site AUROC 0.87 vs LOSO 0.66). All numbers are measured;
no fabricated cells. Outputs fig_methodology_poster.{png,pdf}.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---- restrained, print-safe palette (blue pipeline + one warm accent) --------
INK      = "#1a2540"   # primary text
MUTE     = "#5b6577"   # secondary text
STAGE_FC = "#E9F0FB"; STAGE_EC = "#3F6FC4"   # pipeline stages (blue)
SUB_FC   = "#F4F6FA"; SUB_EC   = "#C2CBDA"   # detail chips (neutral)
GOOD_FC  = "#E4F3EC"; GOOD_EC  = "#2E8B63"   # official-result callout (green)
WARN_FC  = "#FBEEE0"; WARN_EC  = "#C77A2C"   # transfer-wall callout (amber)
ARROW    = "#3F6FC4"

fig, ax = plt.subplots(figsize=(16, 8.6), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, fc, ec, lw=1.8, rad=2.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0.02,rounding_size={rad}",
                 fc=fc, ec=ec, lw=lw, mutation_aspect=0.55, zorder=2))


def txt(x, y, s, size=11, color=INK, weight="normal", ha="center", va="center", style="normal"):
    ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va=va,
            style=style, zorder=4, linespacing=1.35)


def arrow(x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                 mutation_scale=20, lw=2.4, color=ARROW, zorder=3))


# ---- title -------------------------------------------------------------------
txt(50, 96.5, "Confounder-Resistant, Site-Aware Prediction of Cognitive Impairment from Overnight PSG",
    size=17, weight="bold")
txt(50, 92, "Team SDG  ·  George B. Moody PhysioNet Challenge 2026  ·  predict Cognitive_Impairment from one overnight sleep study + demographics",
    size=10.5, color=MUTE)

# ---- pipeline stages ---------------------------------------------------------
Y, H = 49, 35
xs   = [1.5, 21.3, 41.1, 60.9, 80.7]   # 5 columns
W    = 17.8
CY   = Y + H/2
titles = ["1 · DATA", "2 · FEATURES", "3 · MODEL", "4 · DECISION", "5 · VALIDATION"]
for x, t in zip(xs, titles):
    box(x, Y, W, H, STAGE_FC, STAGE_EC)
    txt(x + W/2, Y + H - 3.2, t, size=11.5, weight="bold", color=STAGE_EC)
for x0, x1 in zip(xs, xs[1:]):
    arrow(x0 + W + 0.1, x1 - 0.1, CY)

# 1 Data
txt(xs[0] + W/2, CY - 2.0,
    "3 clinical sites\nBIDMC · Kaiser · Emory\n\nLarge cohort n = 6530\n(497 CI+, prevalence↑ with age)\n\nEDF PSG + CAISR\nannotations + demographics\n(streamed from S3, ~130 GB)",
    size=9.2, va="center")

# 2 Features  (interpretable, annotation-level; no raw EEG)
txt(xs[1] + W/2, Y + H - 7.0, "interpretable, no raw EEG", size=8.6, color=MUTE, style="italic")
feat = [
    "Sleep architecture (CAISR):\nstages, efficiency, WASO,\nfragmentation, bouts, transitions",
    "Event indices:\nAHI, arousal, PLM",
    "Autonomic:\nECG-HRV (SDNN/RMSSD),\nSpO₂ desaturation burden",
    "Demographics  —  no AGE",
]
fy = Y + H - 9.2
for f in feat:
    nlines = f.count("\n") + 1
    bh = 1.8 * nlines + 1.2
    box(xs[1] + 0.9, fy - bh, W - 1.8, bh, SUB_FC, SUB_EC, lw=1.0, rad=1.2)
    txt(xs[1] + W/2, fy - bh/2, f, size=8.0)
    fy -= bh + 0.9

# 3 Model
txt(xs[2] + W/2, CY - 1.0,
    "Histogram gradient\nboosting (HGB)\n\nper-site mixture-of-experts\n+ global fallback\n\nisotonic calibration\nsite-median BMI impute\nNaN-tolerant",
    size=9.2, va="center")

# 4 Decision
txt(xs[3] + W/2, CY + 7.5,
    "Reward-optimal rule\n(derived from the\nChallenge reward):",
    size=9.4, va="center")
box(xs[3] + 2.4, CY - 1.0, W - 4.8, 5.6, "#FFFFFF", STAGE_EC, lw=1.4, rad=1.4)
txt(xs[3] + W/2, CY + 1.8, "predict CI  ⇔  q > p", size=12.5, weight="bold", color=INK)
txt(xs[3] + W/2, CY - 5.2, "q = calibrated prob.,\np = age-specific prevalence",
    size=8.4, color=MUTE, va="center")

# 5 Validation / output
txt(xs[4] + W/2, CY + 6.5,
    "Leave-One-Site-Out\n(LOSO) cross-validation\nwith official scorer\n— mirrors hidden test",
    size=9.2, va="center")
box(xs[4] + 2.2, Y + 3.0, W - 4.4, 6.6, "#FFFFFF", STAGE_EC, lw=1.2, rad=1.4)
txt(xs[4] + W/2, Y + 6.7, "Output per patient", size=8.6, color=MUTE)
txt(xs[4] + W/2, Y + 4.6, "CI label + probability", size=9.4, weight="bold")

# ---- metric strip (defines the primary reward) ------------------------------
box(1.5, 41, 97, 6.2, "#FFFFFF", SUB_EC, lw=1.0, rad=1.2)
txt(3.2, 44.1, "Primary metric — prevalence-weighted Reward:", size=9.4, weight="bold", ha="left")
txt(38, 44.1, "TP = 1/p − 1   ·   TN = 1/(1−p) − 1   ·   FP = FN = −1   (averaged over patients)",
    size=9.2, ha="left", color=INK)
txt(97, 44.1, "random / all-pos / all-neg ≈ 0", size=8.8, ha="right", color=MUTE, style="italic")

# ---- result callouts ---------------------------------------------------------
# Official frozen submission
box(1.5, 8, 47, 29, GOOD_FC, GOOD_EC, lw=2.0)
txt(25, 33.5, "OFFICIAL ENTRY  (frozen at deadline, submission 2634)", size=11, weight="bold", color=GOOD_EC)
txt(13, 24.5, "Reward\n0.168", size=20, weight="bold", color=INK)
txt(37, 24.5, "AC-AUROC\n0.748", size=20, weight="bold", color=INK)
txt(25, 13.5, "Large training cohort · positive reward = genuine skill\nbeyond base-rate guessing (random ≈ 0)",
    size=9.0, color=MUTE)

# Transfer wall
box(51.5, 8, 47, 29, WARN_FC, WARN_EC, lw=2.0)
txt(75, 33.5, "KEY FINDING — the cross-site transfer wall", size=11, weight="bold", color=WARN_EC)
txt(63, 24.5, "within-site\nAUROC 0.87", size=15.5, weight="bold", color=INK)
txt(75, 24.5, "→", size=20, weight="bold", color=WARN_EC)
txt(87, 24.5, "cross-site (LOSO)\nAUROC 0.66", size=15.5, weight="bold", color=INK)
txt(75, 13.2, "More patients close the within-site gap; only 3 sites\ncap new-site generalization. The wall is transfer, not features.",
    size=9.0, color=MUTE)

# footer
txt(50, 3.2, "LOSO scored with the official evaluate_model.py · age excluded from the model to resist the age–risk confounder the metric discounts",
    size=8.4, color=MUTE, style="italic")

plt.tight_layout(pad=0.4)
for ext in ("png", "pdf"):
    fig.savefig(f"fig_methodology_poster.{ext}", bbox_inches="tight", facecolor="white")
print("wrote fig_methodology_poster.png / .pdf")
