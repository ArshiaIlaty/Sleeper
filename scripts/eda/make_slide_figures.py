"""Chart figures for the 'Our approach & results' slide section.

Three figures, all in the validated figstyle palette, sized to drop into the
Scoring Metrics deck's content area:
  fig_slide_features   — the engineered-feature stack (3 families → 337 columns)
  fig_slide_benchmark  — internal LOSO reward: baseline vs +engineered features
  fig_slide_forest     — SleepFM fusion forest plot (point + bootstrap 95% CI)

Numbers are the ones verified this cycle (see PROJECT_LOG.md 2026-08-12):
  feature families: baseline 55 (CAISR + demographics), NeuroKit2 160, report 122.
  internal bench harness LOSO reward:  baseline 55-feat 0.098 → +NK2+report 337-feat 0.176.
  verified submission Reward = 0.168 (official evaluate_model, git anchor).
  LOSO pooled reward@π, bootstrap 95% CI (fuse_confirm.py, 2000× resample):
    no-age (336)              0.145 [−0.102, +0.445]
    + AUROC>0.6 selection     0.160 [−0.086, +0.447]
    + SleepFM PCA-32          0.128 [−0.131, +0.420]
    + SleepFM PCA-32 + AUC    0.253 [−0.009, +0.561]

Pure numpy + matplotlib; depends on figstyle for palette + rc.
"""
import os
import numpy as np
import figstyle as fs
from figstyle import RED, BLUE, GOLD, AQUA, VIOLET, INK, INK2, MUTED, GRID

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FIGDIR = os.path.join(REPO, "paper", "figures")


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), bbox_inches="tight", dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


# ---------------------------------------------------------------- feature stack
def fig_features():
    """Horizontal stacked bar: the three feature families summing to 337."""
    import matplotlib.pyplot as plt
    fams = [
        ("Baseline — CAISR sleep architecture + demographics", 55, BLUE),
        ("NeuroKit2 per-stage physiology (HRV · EEG · respiration)", 160, GOLD),
        ("Clinical report (spectral · spindles · oxygenation · resp events)", 122, AQUA),
    ]
    fig, ax = plt.subplots(figsize=(9.2, 2.5))
    left = 0
    for label, n, c in fams:
        ax.barh(0, n, left=left, height=0.5, color=c, edgecolor="white", linewidth=2,
                label=f"{label}  ({n})")
        ax.text(left + n / 2, 0, str(n), ha="center", va="center",
                color="white", fontsize=13, fontweight="bold")
        left += n
    ax.text(left + 6, 0, f"= {left}", ha="left", va="center", color=INK,
            fontsize=14, fontweight="bold")
    ax.set_xlim(0, left + 55)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.set_xlabel("number of engineered features")
    ax.grid(False)
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.45), frameon=False,
              fontsize=10.5, ncol=1, handlelength=1.1, labelspacing=0.5)
    _save(fig, "fig_slide_features")


# ---------------------------------------------------------------- benchmark
def fig_benchmark():
    """Two-bar internal LOSO reward: baseline features vs +engineered, with the
    verified submission Reward drawn as a reference line."""
    import matplotlib.pyplot as plt
    labels = ["Baseline\n(55 CAISR feat.)", "+ NeuroKit2 + report\n(337 feat.)"]
    vals = [0.098, 0.176]
    colors = [MUTED, BLUE]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = np.arange(len(vals))
    bars = ax.bar(x, vals, width=0.55, color=colors, edgecolor="white", linewidth=1.5)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", va="bottom",
                color=INK, fontsize=13, fontweight="bold")
    # verified submission anchor
    ax.axhline(0.168, color=RED, linewidth=1.6, linestyle=(0, (5, 3)), zorder=1)
    ax.text(-0.55, 0.168, "verified submission  0.168", ha="left", va="bottom",
            color=RED, fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel("LOSO pooled Reward")
    ax.set_ylim(0, 0.235)
    ax.set_xlim(-0.6, 1.6)
    ax.grid(axis="y")
    ax.tick_params(length=0)
    _save(fig, "fig_slide_benchmark")


# ---------------------------------------------------------------- forest plot
def fig_forest():
    """Forest plot of LOSO pooled reward: point estimate + bootstrap 95% CI for
    each fusion config. A vertical rule at 0 shows every CI still includes 0."""
    import matplotlib.pyplot as plt
    # (label, point, lo, hi, color, is_best)
    rows = [
        ("no age  ·  336 features",            0.145, -0.102, 0.445, BLUE,   False),
        ("+ AUROC > 0.6 selection",            0.160, -0.086, 0.447, GOLD,   False),
        ("+ SleepFM PCA-32",                   0.128, -0.131, 0.420, VIOLET, False),
        ("+ SleepFM PCA-32 + AUROC > 0.6",     0.253, -0.009, 0.561, RED,    True),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ys = np.arange(len(rows))[::-1]
    for y, (label, pt, lo, hi, c, best) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color=c, linewidth=3, solid_capstyle="round",
                zorder=3, alpha=0.9)
        ax.plot([lo, lo], [y - 0.12, y + 0.12], color=c, linewidth=2, zorder=3)
        ax.plot([hi, hi], [y - 0.12, y + 0.12], color=c, linewidth=2, zorder=3)
        ax.scatter([pt], [y], s=95, color=c, zorder=4, edgecolor="white", linewidth=1.3)
        ax.text(hi + 0.02, y, f"{pt:+.3f}  [{lo:+.2f}, {hi:+.2f}]", va="center",
                ha="left", fontsize=10, color=INK,
                fontweight="bold" if best else "normal")
    ax.axvline(0, color=INK2, linewidth=1.2, zorder=2)
    ax.text(0, len(rows) - 0.35, "no skill", ha="center", va="bottom",
            fontsize=8.5, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
    ax.set_xlabel("LOSO pooled Reward  (bootstrap 95% CI, 2000× resample)")
    ax.set_xlim(-0.35, 0.85)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.grid(axis="x")
    ax.tick_params(length=0)
    for sp in ("left", "right", "top"):
        ax.spines[sp].set_visible(False)
    _save(fig, "fig_slide_forest")


def main():
    fs.apply_style()
    print(f"slide figures -> {FIGDIR}")
    fig_features()
    fig_benchmark()
    fig_forest()
    print("done.")


if __name__ == "__main__":
    main()
