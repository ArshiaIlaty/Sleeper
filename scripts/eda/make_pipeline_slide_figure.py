"""A landscape, slide-optimized pipeline diagram (companion to the dense portrait
fig9_methodology used in the paper).

Six stages flow left→right in two rows so the whole thing fits a 16:9 content
area at a readable size: Data → Decode/QC → Features (two streams) → Fusion →
Model → Evaluation. Minimal text per box (the speaker notes carry detail).
Validated figstyle palette.
"""
import os
import figstyle as fs
from figstyle import RED, BLUE, GOLD, AQUA, VIOLET, INK, INK2, MUTED

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


def _lighten(hex_color, f=0.90):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * f); g = int(g + (255 - g) * f); b = int(b + (255 - b) * f)
    return f"#{r:02x}{g:02x}{b:02x}"


def box(ax, x, y, w, h, title, sub, accent):
    from matplotlib.patches import FancyBboxPatch, Rectangle
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.2, edgecolor=accent, facecolor=_lighten(accent, 0.90), zorder=3))
    ax.add_patch(Rectangle((x, y), 0.06, h, facecolor=accent, edgecolor="none", zorder=4))
    ax.text(x + w / 2, y + h * 0.63, title, fontsize=12.5, fontweight="bold",
            color=INK, va="center", ha="center", zorder=5)
    if sub:
        ax.text(x + w / 2, y + h * 0.27, sub, fontsize=9.2, color=INK2,
                va="center", ha="center", zorder=5)


def arrow(ax, x0, y0, x1, y1, color=MUTED):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                 mutation_scale=16, linewidth=2.0, color=color, zorder=2,
                 shrinkA=1, shrinkB=1))


def main():
    import matplotlib.pyplot as plt
    fs.apply_style()
    fig, ax = plt.subplots(figsize=(11.2, 4.7))
    ax.set_xlim(0, 100); ax.set_ylim(0, 42); ax.axis("off")

    # ---- top row: Data → Decode → two feature streams ----
    box(ax, 1, 30, 20, 9, "① Raw PSG", "1103 nights · 3 sites", BLUE)
    box(ax, 26, 30, 20, 9, "② Decode + QC", "NeuroKit2 · site-aware", GOLD)
    box(ax, 51, 33.5, 22, 7.5, "③ Hand-crafted features", "337 · stage-resolved", BLUE)
    box(ax, 51, 24.5, 22, 7.5, "③ SleepFM embeddings", "2560-d · no leakage", VIOLET)

    # ---- bottom row (right→left flow): Fusion → Model → Eval ----
    box(ax, 78, 24.5, 21, 16, "④ Fusion + selection", "PCA + AUC>0.6 · in-fold", GOLD)
    box(ax, 51, 6, 22, 9, "⑤ Model", "site experts + Kaiser", AQUA)
    box(ax, 26, 6, 20, 9, "⑥ Evaluate (LOSO)", "Reward @ prevalence", INK2)
    box(ax, 1, 6, 20, 9, "Risk score", "calibrated p(CI)", RED)

    # ---- arrows ----
    arrow(ax, 21, 34.5, 26, 34.5, BLUE)                 # data -> decode
    arrow(ax, 46, 34.5, 51, 37.2, GOLD)                 # decode -> hand-crafted
    arrow(ax, 46, 34.5, 51, 28.2, GOLD)                 # decode -> embeddings
    arrow(ax, 73, 37.2, 78, 34, BLUE)                   # hand-crafted -> fusion
    arrow(ax, 73, 28.2, 78, 30, VIOLET)                 # embeddings -> fusion
    arrow(ax, 88.5, 24.5, 88.5, 19, GOLD)               # fusion down
    arrow(ax, 88.5, 19, 62, 15.2, GOLD)                 # -> model (elbow-ish)
    arrow(ax, 51, 10.5, 46, 10.5, AQUA)                 # model -> eval
    arrow(ax, 26, 10.5, 21, 10.5, INK2)                 # eval -> risk score

    # leakage-safe note under the fusion column
    ax.text(84, 21.5, "leakage-safe:\nfit in-fold", fontsize=8.6, style="italic",
            color=RED, ha="right", va="top", linespacing=1.2)

    _save(fig, "fig_slide_pipeline")
    print("done.")


if __name__ == "__main__":
    main()
