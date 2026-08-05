"""Shared matplotlib style + validated palette for the paper figures.

Palette is Edwards-brand-anchored and validated with the dataviz skill's
validate_palette.js (light surface):
  - 2-class (CI vs non-CI): red #C8102E / blue #2166a0  -> ALL CHECKS PASS
  - categorical (<=5):      #C8102E,#2166a0,#a6791f,#1baf7a,#4a3aa7 -> PASS
Charts follow the skill's rules: one measure per axis, legend for >=2 series,
direct labels (no dual axes, no rainbow), recessive grid/axes, thin marks.

Depends only on numpy + matplotlib (no pandas — local pandas has an ABI break).
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- validated palette ----
RED = "#C8102E"       # Edwards red — used for the positive/CI class & highlights
BLUE = "#2166a0"      # saturated blue — non-CI / primary single-measure bars
GOLD = "#a6791f"      # deep gold (brand gold darkened to clear the chroma floor)
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"
CATEGORICAL = [RED, BLUE, GOLD, AQUA, VIOLET]

# Sequential blue ramp (light->dark) from the dataviz reference palette.
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

# Sleep-stage fixed colors (ordered W -> N1 -> N2 -> N3 -> REM). Sequential-ish
# depth of sleep in blue, REM in the brand red so it stands apart.
STAGE_COLORS = {
    "Wake": "#9ec5f4", "N1": "#5598e7", "N2": "#2a78d6",
    "N3": "#104281", "REM": RED,
}

# ---- ink / chrome (from reference palette, light mode) ----
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.labelcolor": INK2,
        "text.color": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "figure.dpi": 130,
    })


def load_rows(path):
    """Read the per-recording CSV into a list of dicts (no pandas)."""
    with open(path) as fh:
        return list(csv.DictReader(fh))


def col(rows, key, cast=float, where=None):
    """Extract a numeric column, dropping blanks and rows failing `where`."""
    out = []
    for r in rows:
        if where is not None and not where(r):
            continue
        v = r.get(key, "")
        if v == "" or v is None:
            continue
        try:
            out.append(cast(v))
        except (ValueError, TypeError):
            continue
    return np.array(out, dtype=float if cast is float else object)


def style_axes_labels(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, color=INK, loc="left", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.tick_params(length=0)


def savefig(fig, path):
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path
