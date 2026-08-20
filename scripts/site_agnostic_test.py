#!/usr/bin/env python3
"""Site-agnostic vs site-MoE under LOSO (new-site-test-set strategy).

Decision: assume the WORST -- the hidden challenge test set is NEW sites -> LOSO is
the real metric, 70/15/15 is a vanity number. Under that assumption the current
production architecture is pointed the wrong way: the site mixture-of-experts has no
expert for an unseen site (falls back to global), the Kaiser fine-tuned head is dead
weight off-Kaiser, and features that only help within one site won't transfer.

This tests, on the large cohort (497 positives) under LOSO (train 2 sites, predict
the held-out 3rd), whether a SITE-AGNOSTIC pipeline generalizes better. Layers added
one at a time so each contribution is visible:

  A moe_baseline   current production: per-site experts + Kaiser fine-tune + global-pi
                   threshold  (== run_local_cv.loso)
  B pooled         single global model, no site experts, no Kaiser head
  C pooled+sitez   + per-site z-score (held-out site standardized by its OWN feature
                   distribution -- label-free, test-legal)
  D pooled+sitez+inv  + drop the top-K most SITE-DISCRIMINATIVE features (highest
                   between-site / within-site variance ratio, measured on TRAIN sites
                   only) -> keep only cross-site-invariant features

All LOSO-honest: every stat (model, z-score, invariance ranking, threshold) is fit
on TRAINING sites only; the held-out site contributes no labels. Threshold is the
train-prevalence global pi for every variant (a new site has no labels to tune one).

Usage (on pdmle, as arshia_ilaty_physio26):
  PYTHONPATH=/data-temp/physio-viewer/bench/repo python3 site_agnostic_test.py \
    --exports /data-temp/physio-viewer/exports \
    --repo /data-temp/physio-viewer/bench/repo --cvdir /data-temp/physio-viewer/bench \
    --cohort large --drop-frac 0.25
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import pandas as pd

JOIN = ["bids_folder", "session"]
NON_FEATURE = {
    "dataset", "bids_folder", "session", "site", "site_name", "label", "age",
    "sex", "race", "ethnicity", "time_to_event", "time_to_last_visit",
    "ecg_channel", "eeg_channel", "rsp_channel", "spo2_channel", "eog_channel",
    "emg_channel", "has_resp_caisr", "report_seconds", "nk_seconds",
    "micro_seconds", "n_beats_total", "n_beats_clean", "arousal_fs",
    "n_epochs_scored", "duration_hours", "n_epochs",
}


def read_keyed(path):
    df = pd.read_csv(path, dtype={"bids_folder": str, "session": str}, low_memory=False)
    for k in JOIN:
        df[k] = df[k].fillna("").astype(str).str.strip()
    return df.drop_duplicates(JOIN)


def build_matrix(exports, cohort):
    base = read_keyed(os.path.join(exports, f"features_{cohort}.csv"))
    arch = read_keyed(os.path.join(exports, f"arch_features_{cohort}.csv"))
    rep = read_keyed(os.path.join(exports, f"report_features_{cohort}.csv"))

    def prefix(df, pfx):
        cols = {c: f"{pfx}{c}" for c in df.columns
                if c not in JOIN and c not in NON_FEATURE}
        return df[JOIN + list(cols)].rename(columns=cols)

    m = base.merge(prefix(arch, "arch__"), on=JOIN, how="left", validate="one_to_one") \
            .merge(prefix(rep, "rep__"), on=JOIN, how="left", validate="one_to_one")
    lab = m["label"].map({True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0})
    keep = lab.isin([0, 1]).values
    m = m[keep].reset_index(drop=True)
    y = lab[keep].astype(int).values
    ages = pd.to_numeric(m["age"], errors="coerce").astype(float).values
    sites = m["site"].astype(str).values
    base_feats = [c for c in base.columns if c not in JOIN and c not in NON_FEATURE]
    feat_cols = base_feats + [c for c in m.columns if c.startswith(("arch__", "rep__"))]
    X = m[feat_cols].apply(pd.to_numeric, errors="coerce").astype(np.float64).values
    bmi_idx = feat_cols.index("bmi") if "bmi" in feat_cols else None
    return X, y, ages, sites, feat_cols, bmi_idx


def site_z(X, sites, train_mask, protect):
    Xo = X.copy()
    for s in np.unique(sites):
        m = sites == s
        fit = m & train_mask
        rows = fit if fit.sum() >= 10 else m
        mu = np.nanmean(X[rows], axis=0)
        sd = np.nanstd(X[rows], axis=0); sd = np.where(sd < 1e-8, 1.0, sd)
        Xo[m] = (X[m] - mu) / sd
    for c in protect:
        Xo[:, c] = X[:, c]
    return Xo


def site_discriminability(X, sites, train_mask):
    """Per-feature between-site / within-site variance ratio (eta-like), TRAIN only.
    High => the feature separates sites => likely won't transfer. Label-free."""
    ts = [s for s in np.unique(sites) if (train_mask & (sites == s)).sum() >= 10]
    grand = np.nanmean(X[train_mask], axis=0)
    nfeat = X.shape[1]
    between = np.zeros(nfeat); within = np.zeros(nfeat); ntot = 0
    for s in ts:
        m = (sites == s) & train_mask
        n_i = m.sum(); ntot += n_i
        mu_i = np.nanmean(X[m], axis=0)
        between += n_i * (mu_i - grand) ** 2
        within += np.nansum((X[m] - mu_i) ** 2, axis=0)
    within = np.where(within < 1e-8, 1e-8, within)
    return (between / max(len(ts) - 1, 1)) / (within / max(ntot - len(ts), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cvdir", required=True)
    ap.add_argument("--cohort", default="large")
    ap.add_argument("--drop-frac", type=float, default=0.25,
                    help="fraction of most site-discriminative features to drop in variant D")
    args = ap.parse_args()
    for p in (args.repo, args.cvdir):
        if p not in sys.path:
            sys.path.insert(0, p)
    import feature_prep as fp
    import evaluate_model as ev
    import run_local_cv as rlc

    X, y, ages, sites, names, bmi_idx = build_matrix(args.exports, args.cohort)
    protect = [bmi_idx] if bmi_idx is not None else []
    print(f"cohort={args.cohort}  n={len(y)}  pos={int(y.sum())}  feats={X.shape[1]}  drop_frac={args.drop_frac}")

    def loso(variant):
        yt_all, yp_all, ya_all = [], [], []
        folds = []
        for site in np.unique(sites):
            te = sites == site; tr = ~te
            if te.sum() == 0 or len(np.unique(y[tr])) < 2:
                continue
            Xf = X
            if variant in ("C", "D"):
                Xf = site_z(Xf, sites, tr, protect)
            colmask = np.ones(Xf.shape[1], bool)
            if variant == "D" and args.drop_frac > 0:
                disc = site_discriminability(Xf, sites, tr)
                k = int(round(args.drop_frac * Xf.shape[1]))
                if k > 0:
                    drop = np.argsort(disc)[::-1][:k]
                    colmask[drop] = False
                    if bmi_idx is not None:
                        colmask[bmi_idx] = True     # never drop bmi (imputer needs it)
            Xf = Xf[:, colmask].astype(np.float32)
            nm = [n for n, keep in zip(names, colmask) if keep]

            imp = fp.fit_bmi_imputer(Xf[tr], sites[tr], nm)
            Xtr = fp.apply_bmi_imputer(Xf[tr], sites[tr], imp)
            Xte = fp.apply_bmi_imputer(Xf[te], sites[te], imp)
            if variant == "A":
                models = fp.fit_site_models(Xtr, y[tr], sites[tr])
                models = fp.fit_kaiser_finetuned(models, Xtr, y[tr], sites[tr])
                prob = fp.predict_with_kaiser_override(models, Xte, sites[te], use_kaiser_finetuned=True)
            else:
                clf = fp.fit_clf(Xtr, y[tr])                 # single pooled model
                prob = clf.predict_proba(Xte)[:, 1]
            binary = (prob > float(y[tr].mean())).astype(int)
            a2p = ev.compute_prevalence(ages[te], y, ages, gap=2)
            folds.append((str(site), int(te.sum()),
                          float(ev.compute_reward(y[te], binary, ages[te], a2p)),
                          float(ev.compute_auroc_age(y[te], prob, ages[te], 2)),
                          float(ev.compute_auroc(y[te], prob))))
            yt_all.extend(y[te]); yp_all.extend(prob); ya_all.extend(ages[te])
        yt = np.asarray(yt_all); yp = np.asarray(yp_all); ya = np.asarray(ya_all)
        a2p_all = ev.compute_prevalence(ya, y, ages, gap=2)
        binpi = (yp > float(y.mean())).astype(int)
        pooled = {
            "reward_at_pi": float(ev.compute_reward(yt, binpi, ya, a2p_all)),
            "auroc": float(ev.compute_auroc(yt, yp)),
            "age_auroc": float(ev.compute_auroc_age(yt, yp, ya, 2)),
            "auprc": float(ev.compute_auprc(yt, yp)),
        }
        return pooled, folds

    labels = {"A": "moe_baseline (site experts + Kaiser)",
              "B": "pooled (single global model)",
              "C": "pooled + site_z",
              "D": f"pooled + site_z + drop {int(args.drop_frac*100)}% site-discriminative"}
    for v in ("A", "B", "C", "D"):
        pooled, folds = loso(v)
        print(f"\n=== {v}: {labels[v]} ===")
        print(f"  POOLED reward@pi={pooled['reward_at_pi']:+.4f}  AUROC={pooled['auroc']:.4f}"
              f"  age-AUROC={pooled['age_auroc']:.4f}  AUPRC={pooled['auprc']:.4f}")
        for s, n, r, aa, au in folds:
            print(f"    hold-out {rlc.SITE_NAMES.get(s, s):>6} n={n:5d}  reward={r:+.3f}"
                  f"  age-AUROC={aa:.3f}  AUROC={au:.3f}")

    print("\nDONE_SITEAGNOSTIC")


if __name__ == "__main__":
    main()
