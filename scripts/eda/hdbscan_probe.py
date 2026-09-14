#!/usr/bin/env python3
"""Unsupervised HDBSCAN pattern probe on a Challenge-2026 feature matrix.

Standardize -> UMAP -> HDBSCAN, then ask what the density clusters recover:
SITE/montage batch structure, CI phenotype, or something else. Reports cluster
alignment with site vs CI (ARI/NMI + per-cluster prevalence), a site-predictability
batch-effect check, and a 2D UMAP figure colored by site / CI / cluster.

Descriptive only -- nothing is folded into the champion unless a cluster-derived
feature clears the LOSO fusion gate. Aggregate stats only (DUA).

Parameterized by --cache so it can be re-pointed at PSG-feature or embedding
matrices later (any .npz with X + y + sites + ages [+ pids/feature_names]).

Run on pdmle as arshia_ilaty_physio26:
  python3 hdbscan_probe.py \
    --cache /data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz \
    --png /tmp/sleeper_umap.png
"""
import argparse
import warnings
import numpy as np

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help=".npz with X,y,sites,ages")
    ap.add_argument("--png", default="/tmp/umap_probe.png")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-neighbors", type=int, default=30)
    ap.add_argument("--min-cluster-sizes", type=int, nargs="+", default=[20, 40])
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    X = d["X"].astype(np.float32)
    y = d["y"].astype(int)
    ages = d["ages"].astype(float) if "ages" in d else np.full(len(y), np.nan)
    sites = np.asarray([str(s) for s in d["sites"]])
    print(f"X={X.shape}  CI+={int(y.sum())}/{len(y)} ({y.mean():.1%})")
    site_names, site_counts = np.unique(sites, return_counts=True)
    print("sites:", dict(zip(site_names, site_counts)))

    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X))

    # does SITE leak into the features at all? (batch-effect magnitude)
    from sklearn.model_selection import cross_val_predict
    from sklearn.ensemble import RandomForestClassifier
    site_int = np.searchsorted(site_names, sites)
    sp = cross_val_predict(RandomForestClassifier(200, n_jobs=-1, random_state=args.seed),
                           Xs, site_int, cv=5)
    print(f"\nSite predictability (5-fold RF acc): {(sp == site_int).mean():.3f}  "
          f"(majority baseline={site_counts.max() / len(y):.3f})")

    import umap
    emb10 = umap.UMAP(n_components=10, n_neighbors=args.n_neighbors, min_dist=0.0,
                      random_state=args.seed).fit_transform(Xs)
    emb2 = umap.UMAP(n_components=2, n_neighbors=args.n_neighbors, min_dist=0.1,
                     random_state=args.seed).fit_transform(Xs)

    from sklearn.cluster import HDBSCAN
    from sklearn.metrics import (adjusted_rand_score as ARI,
                                 normalized_mutual_info_score as NMI, silhouette_score)
    import pandas as pd
    print(f"\nSilhouette of SITE in UMAP-10: {silhouette_score(emb10, site_int):.3f} "
          f"(>0 => sites form separated blobs)")

    labels_final = None
    for mcs in args.min_cluster_sizes:
        lab = HDBSCAN(min_cluster_size=mcs, min_samples=10).fit_predict(emb10)
        uniq = [c for c in np.unique(lab) if c != -1]
        nn = lab != -1
        print(f"\n=== HDBSCAN(min_cluster_size={mcs}): {len(uniq)} clusters, "
              f"noise={(lab == -1).mean():.1%} ===")
        if len(uniq) < 2:
            print("  <2 clusters, skipping"); continue
        print(f"  cluster~SITE ARI={ARI(lab[nn], site_int[nn]):+.3f} NMI={NMI(lab[nn], site_int[nn]):.3f}")
        print(f"  cluster~CI   ARI={ARI(lab[nn], y[nn]):+.3f} NMI={NMI(lab[nn], y[nn]):.3f}")
        print(pd.crosstab(lab, sites).to_string())
        for c in sorted(np.unique(lab)):
            m = lab == c
            top = pd.Series(sites[m]).value_counts()
            nm = "noise" if c == -1 else f"c{c}"
            print(f"    {nm:8s} n={m.sum():4d} CI={y[m].mean():.3f} "
                  f"age={np.nanmean(ages[m]):5.1f} site={top.index[0]}({top.iloc[0] / m.sum():.0%})")
        labels_final = lab

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    oi = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442"]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    for i, s in enumerate(site_names):
        m = sites == s
        ax[0].scatter(emb2[m, 0], emb2[m, 1], s=6, c=oi[i % len(oi)], alpha=.6, label=f"{s} (n={m.sum()})")
    ax[0].set_title("UMAP by SITE"); ax[0].legend(markerscale=2, fontsize=8)
    neg, pos = y == 0, y == 1
    ax[1].scatter(emb2[neg, 0], emb2[neg, 1], s=6, c="#BBBBBB", alpha=.5, label=f"non-CI (n={neg.sum()})")
    ax[1].scatter(emb2[pos, 0], emb2[pos, 1], s=18, c="#D55E00", alpha=.9, edgecolors="k",
                  linewidths=.3, label=f"CI (n={pos.sum()})")
    ax[1].set_title("UMAP by CI label"); ax[1].legend(markerscale=1.5, fontsize=8)
    if labels_final is not None:
        for j, c in enumerate(sorted(np.unique(labels_final))):
            m = labels_final == c
            col = "#DDDDDD" if c == -1 else oi[j % len(oi)]
            ax[2].scatter(emb2[m, 0], emb2[m, 1], s=6, c=col, alpha=.6,
                          label=("noise" if c == -1 else f"c{c}"))
        ax[2].set_title("UMAP by HDBSCAN cluster"); ax[2].legend(markerscale=2, fontsize=7, ncol=2)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.savefig(args.png, dpi=110)
    print(f"\nsaved figure -> {args.png}")


if __name__ == "__main__":
    main()
