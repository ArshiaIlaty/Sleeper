#!/usr/bin/env python3
"""Dimensionality-reduction visualizations: PCA vs UMAP vs t-SNE on the feature matrix.

Projects the standardized per-recording feature matrix to 2-D with three reducers
and renders a 3x3 panel (rows = PCA / UMAP / t-SNE; columns = colored by the three
axes that matter for this challenge: CI label, site, age). Purpose is EDA/insight
into structure and cross-site separation, NOT a model — no leakage rules apply
(this is unsupervised layout of the whole cohort for looking at).

Engine note: t-SNE uses sklearn Barnes-Hut on CPU. tsne-cuda (GPU) cannot run on
this box (its wheel is built for CUDA 10.2; the server has CUDA 12.5 -- hard ABI
gap). At this scale (~1k-6.5k points) CPU t-SNE finishes in seconds-to-minutes and
the GPU would save nothing (GPU t-SNE only pays off at 1e5-1e6 points).

Color (colorblind-safe by construction):
  * label -> Okabe-Ito orange (CI+) vs blue (CI-), the highest-separation pair
  * site  -> three distinct Okabe-Ito hues (categorical identity, fixed order)
  * age   -> viridis sequential ramp (perceptually uniform)

Usage (on pdmle, as arshia_ilaty_physio26):
  python3 embed_viz.py --cache /data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz \
      --out /data-temp/physio-viewer/exports/embed_viz_standard.png --seed 42
  # or from a combined CSV:
  python3 embed_viz.py --csv /data-temp/physio-viewer/exports/all_features_standard.csv \
      --out /data-temp/physio-viewer/exports/embed_viz_all.png
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Okabe-Ito colorblind-safe categorical palette
OI = {
    "orange": "#E69F00", "blue": "#0072B2", "green": "#009E73",
    "vermillion": "#D55E00", "skyblue": "#56B4E9", "yellow": "#F0E442",
    "purple": "#CC79A7", "black": "#000000",
}
LABEL_COLORS = {0: OI["blue"], 1: OI["orange"]}          # CI- vs CI+
SITE_ORDER = ["S0001", "I0002", "I0006"]                 # BIDMC, Emory, Kaiser
SITE_NAMES = {"S0001": "BIDMC", "I0002": "Emory", "I0006": "Kaiser"}
SITE_COLORS = {"S0001": OI["green"], "I0002": OI["vermillion"], "I0006": OI["purple"]}

REPO = "/data-temp/physio-viewer/bench/repo"


def load_from_cache(path):
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float64)
    y = d["y"].astype(int)
    ages = d["ages"].astype(float)
    sites = np.asarray([str(s) for s in d["sites"]])
    names = [str(n) for n in d["feature_names"]]
    return X, y, ages, sites, names


def load_from_embnpz(path):
    """Self-contained embedding npz (e.g. GNN last-layer): keys embedding, label,
    site, age, patient_id. Metadata already bundled -- no external join."""
    d = np.load(path, allow_pickle=True)
    X = np.asarray(d["embedding"]).astype(np.float64)
    y = np.asarray(d["label"]).astype(int)
    ages = np.asarray(d["age"]).astype(float)
    sites = np.asarray([str(s) for s in d["site"]])
    names = [f"emb_{i:02d}" for i in range(X.shape[1])]
    return X, y, ages, sites, names


def load_from_embdir(embdir, meta_npz):
    """Directory of per-recording .npy vectors (e.g. Kingson pure per-stage), joined
    to a metadata npz (patient_id/label/site/age) by the file stem sub-<PID>_ses-<n>.
    Only recordings present in BOTH the dir and the metadata are kept."""
    import glob
    d = np.load(meta_npz, allow_pickle=True)
    lab = {str(p): int(l) for p, l in zip(d["patient_id"], d["label"])}
    site_of = {str(p): str(s) for p, s in zip(d["patient_id"], d["site"])}
    age_of = {str(p): float(a) for p, a in zip(d["patient_id"], d["age"])}
    files = sorted(glob.glob(os.path.join(embdir, "*.npy")))
    rows, y, ages, sites, missing = [], [], [], [], 0
    for f in files:
        pid = os.path.basename(f)[:-4]
        if pid not in lab:
            missing += 1
            continue
        rows.append(np.load(f, allow_pickle=True).astype(np.float64).ravel())
        y.append(lab[pid]); ages.append(age_of[pid]); sites.append(site_of[pid])
    if missing:
        print(f"  ! {missing} .npy files had no metadata match (dropped)", file=sys.stderr)
    X = np.vstack(rows)
    names = [f"e{i:04d}" for i in range(X.shape[1])]
    return X, np.asarray(y, int), np.asarray(ages, float), np.asarray(sites), names


def load_from_csv(path):
    import pandas as pd
    df = pd.read_csv(path, low_memory=False)
    y = df["label"].map({True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}).astype(float)
    keep = y.isin([0, 1]).values
    df = df[keep].reset_index(drop=True)
    y = y[keep].astype(int).values
    ages = pd.to_numeric(df["age"], errors="coerce").astype(float).values
    sites = df["site"].astype(str).values
    drop = {"dataset", "bids_folder", "session", "site", "site_name", "label",
            "age", "sex", "race", "ethnicity", "time_to_event", "time_to_last_visit"}
    feat_cols = [c for c in df.columns if c not in drop
                 and not df[c].dtype == object]
    X = df[feat_cols].apply(__import__("pandas").to_numeric, errors="coerce").astype(np.float64).values
    return X, y, ages, sites, feat_cols


def prep(X):
    """Median-impute then standardize (t-SNE/UMAP need finite, scaled input)."""
    Xi = SimpleImputer(strategy="median").fit_transform(X)
    return StandardScaler().fit_transform(Xi)


def embed_all(Xs, seed, tsne_perplexity):
    reducers = {}
    reducers["PCA"] = PCA(n_components=2, random_state=seed).fit_transform(Xs)
    try:
        import umap
        reducers["UMAP"] = umap.UMAP(n_components=2, random_state=seed,
                                     n_neighbors=15, min_dist=0.1).fit_transform(Xs)
    except Exception as e:
        print(f"  ! UMAP failed: {type(e).__name__}: {e}", file=sys.stderr)
        reducers["UMAP"] = None
    reducers["t-SNE"] = TSNE(n_components=2, random_state=seed,
                             perplexity=tsne_perplexity, init="pca",
                             learning_rate="auto").fit_transform(Xs)
    return reducers


def _scatter(ax, XY, colors, s=9, alpha=0.75, order=None):
    if order is not None:
        ax.scatter(XY[order, 0], XY[order, 1], c=[colors[i] for i in order],
                   s=s, alpha=alpha, linewidths=0, rasterized=True)
    else:
        ax.scatter(XY[:, 0], XY[:, 1], c=colors, s=s, alpha=alpha,
                   linewidths=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#cccccc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--emb-npz", default=None,
                    help="self-contained embedding npz (keys: embedding,label,site,age,patient_id)")
    ap.add_argument("--emb-dir", default=None,
                    help="dir of per-recording .npy vectors (needs --meta-npz for label/site/age)")
    ap.add_argument("--meta-npz", default=None,
                    help="metadata npz (patient_id/label/site/age) to join --emb-dir against")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--perplexity", type=float, default=None,
                    help="t-SNE perplexity; default = min(30, n/40)")
    args = ap.parse_args()
    if not (args.cache or args.csv or args.emb_npz or args.emb_dir):
        sys.exit("need one of --cache / --csv / --emb-npz / --emb-dir")
    if args.emb_dir and not args.meta_npz:
        sys.exit("--emb-dir requires --meta-npz for label/site/age")

    if args.cache:
        X, y, ages, sites, names = load_from_cache(args.cache)
    elif args.emb_npz:
        X, y, ages, sites, names = load_from_embnpz(args.emb_npz)
    elif args.emb_dir:
        X, y, ages, sites, names = load_from_embdir(args.emb_dir, args.meta_npz)
    else:
        X, y, ages, sites, names = load_from_csv(args.csv)
    n = len(y)
    print(f"matrix: {X.shape}  positives={int(y.sum())}  sites={sorted(set(sites))}")
    perp = args.perplexity or min(30.0, max(5.0, n / 40.0))
    print(f"standardizing + embedding (t-SNE perplexity={perp:.1f}) ...")
    Xs = prep(X)
    emb = embed_all(Xs, args.seed, perp)

    # color vectors
    label_c = [LABEL_COLORS[int(v)] for v in y]
    site_c = [SITE_COLORS.get(s, "#888888") for s in sites]
    finite_age = ages[np.isfinite(ages)]
    amin, amax = (np.nanmin(finite_age), np.nanmax(finite_age)) if finite_age.size else (0, 1)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(amin, amax)
    age_c = [cmap(norm(a)) if np.isfinite(a) else "#dddddd" for a in ages]

    # draw CI+ points last (on top) so the rare positives are visible
    pos_last = np.argsort(y)

    rows = ["PCA", "UMAP", "t-SNE"]
    cols = [("CI label", label_c, pos_last),
            ("Site", site_c, None),
            ("Age", age_c, None)]
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 13.5))
    for r, method in enumerate(rows):
        XY = emb[method]
        for c, (title, colvec, order) in enumerate(cols):
            ax = axes[r][c]
            if XY is None:
                ax.text(0.5, 0.5, f"{method}\nunavailable", ha="center", va="center")
                ax.set_xticks([]); ax.set_yticks([])
            else:
                _scatter(ax, XY, colvec, order=order)
            if c == 0:
                ax.set_ylabel(method, fontsize=15, fontweight="bold")
            if r == 0:
                ax.set_title(title, fontsize=13)

    # legends: one row beneath, per color scheme
    label_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=LABEL_COLORS[1],
                            markersize=9, label=f"CI+ (n={int(y.sum())})"),
                     Line2D([0], [0], marker="o", color="w", markerfacecolor=LABEL_COLORS[0],
                            markersize=9, label=f"CI− (n={int((y==0).sum())})")]
    site_handles = [Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=SITE_COLORS[s], markersize=9,
                           label=f"{SITE_NAMES[s]} (n={int((sites==s).sum())})")
                    for s in SITE_ORDER if (sites == s).any()]
    axes[2][0].legend(handles=label_handles, loc="upper center",
                      bbox_to_anchor=(0.5, -0.04), ncol=2, frameon=False, fontsize=10)
    axes[2][1].legend(handles=site_handles, loc="upper center",
                      bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False, fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[2][2], orientation="horizontal",
                        fraction=0.05, pad=0.08)
    cbar.set_label("Age (years)", fontsize=10)

    src = args.cache or args.csv or args.emb_npz or args.emb_dir
    fig.suptitle(f"PCA / UMAP / t-SNE on {n} recordings ({int(y.sum())} CI+, "
                 f"{X.shape[1]} features)\n{src.split('/')[-1]}",
                 fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")

    # also dump the 2-D coordinates for downstream / interactive use
    npz = args.out.rsplit(".", 1)[0] + "_coords.npz"
    save = {"y": y, "ages": ages, "sites": sites}
    for m in rows:
        if emb[m] is not None:
            save[m.replace("-", "")] = emb[m]
    np.savez_compressed(npz, **save)
    print(f"wrote {npz}")
    print("DONE_EMBED_VIZ")


if __name__ == "__main__":
    main()
