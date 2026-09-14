#!/usr/bin/env python3
"""Per-site RANK-normalization + reward-threshold prototype on the champion cache.

Stolen levers (competitor review, teams 480/476/553):
  * Per-site rank normalization (SIREN, team 480): replace each feature by its
    within-site percentile rank on a fixed 256-level grid. Nonparametric batch-effect
    remover. LOSO-legal because ranking uses NO labels -- a new site arrives as a
    batch and is ranked within itself (transductive). Applied per-site globally ==
    what each LOSO fold would compute for the held-out site.
  * Reward-aware decision threshold (476/553): production uses the train-prevalence
    threshold pi. We compare pi vs a DEPLOYABLE cross-site-transfer threshold (for the
    held-out site, pick the reward-max threshold on the OTHER sites' OOF preds) vs an
    ORACLE ceiling (reward-max on the held-out labels themselves -- upper bound, NOT
    deployable, shown only to bound the headroom).

All arms run through the EXACT production stack (feature_prep.fit_site_models /
fit_kaiser_finetuned / predict_with_kaiser_override / fit_bmi_imputer, HGB MoE) so the
'moe_none' arm reproduces the champion LOSO anchor byte-for-byte. Only the feature
matrix is pre-transformed and (for pooled arms) the model is swapped to a single global
HGB. BMI is protected (kept raw) so the name-based imputer semantics never change.

Usage (on pdmle, as arshia_ilaty_physio26):
  PYTHONPATH=/data-temp/physio-viewer/bench/repo python3 rank_norm_test.py \
    --cache /data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz \
    --repo  /data-temp/physio-viewer/bench/repo
"""
from __future__ import annotations
import argparse, sys
import numpy as np

SITE_NAMES = {"S0001": "BIDMC", "I0002": "Emory", "I0006": "Kaiser"}
GRID = np.linspace(0.01, 0.50, 80)   # matches feature_prep._best_threshold grid


# ------------------------------------------------------------------ transforms
def site_z(X, sites, protect, levels=None):
    """Per-site z-score (each site standardized by its OWN rows; label-free)."""
    Xo = X.copy()
    for s in np.unique(sites):
        m = sites == s
        mu = np.nanmean(X[m], axis=0)
        sd = np.nanstd(X[m], axis=0); sd = np.where(sd < 1e-8, 1.0, sd)
        Xo[m] = (X[m] - mu) / sd
    for c in protect:
        Xo[:, c] = X[:, c]
    return Xo


def site_rank(X, sites, protect, levels=256):
    """Per-site percentile rank on a fixed `levels`-step grid (SIREN). Each feature
    ranked within its own site's rows; NaNs stay NaN (HGB handles them). Label-free."""
    from scipy.stats import rankdata
    Xo = np.full_like(X, np.nan)
    for s in np.unique(sites):
        idx = np.where(sites == s)[0]
        sub = X[idx]
        out = np.full_like(sub, np.nan)
        for j in range(sub.shape[1]):
            col = sub[:, j]
            ok = np.isfinite(col)
            n_ok = int(ok.sum())
            if n_ok >= 2:
                rr = rankdata(col[ok], method="average")      # 1..n_ok, ties averaged
                pr = (rr - 1.0) / (n_ok - 1.0)                 # -> [0,1]
                pr = np.round(pr * (levels - 1)) / (levels - 1)
                out[np.where(ok)[0], j] = pr
            elif n_ok == 1:
                out[np.where(ok)[0], j] = 0.5
        Xo[idx] = out
    for c in protect:
        Xo[:, c] = X[:, c]
    return Xo


# ------------------------------------------------------------------ LOSO driver
def run_loso(X, y, ages, sites, fp, ev, names, model="moe"):
    """One LOSO pass. Returns pooled metrics, per-fold rows, and OOF arrays."""
    oof_site, tt, pp, aa = [], [], [], []
    folds = []
    for site in np.unique(sites):
        te = sites == site; tr = ~te
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        imp = fp.fit_bmi_imputer(X[tr], sites[tr], names)
        Xtr = fp.apply_bmi_imputer(X[tr], sites[tr], imp)
        Xte = fp.apply_bmi_imputer(X[te], sites[te], imp)
        if model == "moe":
            models = fp.fit_site_models(Xtr, y[tr], sites[tr])
            models = fp.fit_kaiser_finetuned(models, Xtr, y[tr], sites[tr])
            prob = fp.predict_with_kaiser_override(models, Xte, sites[te], use_kaiser_finetuned=True)
        else:  # single pooled global model
            clf = fp.fit_clf(Xtr, y[tr])
            prob = clf.predict_proba(Xte)[:, 1]
        binary = (prob > float(y[tr].mean())).astype(int)
        a2p = ev.compute_prevalence(ages[te], y, ages, gap=2)
        folds.append((str(site), int(te.sum()),
                      float(ev.compute_reward(y[te], binary, ages[te], a2p)),
                      float(ev.compute_auroc_age(y[te], prob, ages[te], 2)),
                      float(ev.compute_auroc(y[te], prob))))
        oof_site += [str(site)] * int(te.sum())
        tt.extend(y[te]); pp.extend(prob); aa.extend(ages[te])
    yt = np.asarray(tt); yp = np.asarray(pp); ya = np.asarray(aa)
    osite = np.asarray(oof_site)
    a2p_all = ev.compute_prevalence(ya, y, ages, gap=2)
    binpi = (yp > float(y.mean())).astype(int)
    pooled = {
        "reward_at_pi": float(ev.compute_reward(yt, binpi, ya, a2p_all)),
        "auroc": float(ev.compute_auroc(yt, yp)),
        "age_auroc": float(ev.compute_auroc_age(yt, yp, ya, 2)),
        "auprc": float(ev.compute_auprc(yt, yp)),
    }
    return pooled, folds, (osite, yt, yp, ya)


# --------------------------------------------------------- reward-threshold study
def threshold_study(oof, y_full, ages_full, ev):
    """pi vs cross-site-transfer (deployable) vs oracle (ceiling) reward, on stored OOF."""
    osite, yt, yp, ya = oof
    a2p_all = ev.compute_prevalence(ya, y_full, ages_full, gap=2)

    def reward(binary):
        return float(ev.compute_reward(yt, binary, ya, a2p_all))

    # pi (train prevalence of the whole set == the production rule proxy)
    r_pi = reward((yp > float(y_full.mean())).astype(int))

    # oracle: single global threshold maximizing pooled OOF reward (uses test labels)
    r_oracle = max(reward((yp > t).astype(int)) for t in GRID)

    # deployable cross-site transfer: for each held-out site, pick the reward-max
    # threshold on the OTHER sites' OOF predictions, apply to the held-out site.
    binary = np.zeros(len(yt), int)
    for s in np.unique(osite):
        te = osite == s; tr = ~te
        a2p_tr = ev.compute_prevalence(ya[tr], y_full, ages_full, gap=2)
        best_t, best_r = float(y_full.mean()), -1e9
        for t in GRID:
            r = float(ev.compute_reward(yt[tr], (yp[tr] > t).astype(int), ya[tr], a2p_tr))
            if r > best_r:
                best_r, best_t = r, t
        binary[te] = (yp[te] > best_t).astype(int)
    r_transfer = reward(binary)
    return r_pi, r_transfer, r_oracle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--repo", required=True)
    args = ap.parse_args()
    if args.repo not in sys.path:
        sys.path.insert(0, args.repo)
    import feature_prep as fp
    import evaluate_model as ev

    d = np.load(args.cache, allow_pickle=True)
    X = d["X"].astype(np.float64); y = d["y"].astype(int)
    ages = d["ages"].astype(float); sites = np.asarray([str(s) for s in d["sites"]])
    names = [str(n) for n in d["feature_names"]]
    bmi_idx = names.index("bmi") if "bmi" in names else None
    protect = [bmi_idx] if bmi_idx is not None else []
    print(f"cache: {X.shape[0]}x{X.shape[1]}  pos={int(y.sum())}  "
          f"sites={ {s:int((sites==s).sum()) for s in np.unique(sites)} }  bmi_protected={protect!=[]}")

    arms = [
        ("moe_none  (CHAMPION ANCHOR)", "moe",    None),
        ("moe_rank  (MoE + per-site rank)", "moe", site_rank),
        ("pooled_none", "pooled",              None),
        ("pooled_sitez (per-site z-score)", "pooled", site_z),
        ("pooled_rank  (per-site rank-norm)", "pooled", site_rank),
    ]
    print("\n" + "=" * 92)
    print("LOSO arms (production stack; only the feature matrix / model head vary)")
    print("=" * 92)
    results = {}
    for tag, model, tf in arms:
        Xin = X if tf is None else tf(X, sites, protect)
        pooled, folds, oof = run_loso(Xin, y, ages, sites, fp, ev, names, model=model)
        r_pi, r_tr, r_or = threshold_study(oof, y, ages, ev)
        results[tag] = (pooled, r_pi, r_tr, r_or)
        print(f"\n--- {tag} ---")
        for s, n, r, ag, au in folds:
            print(f"    hold-out {SITE_NAMES.get(s, s):>6} n={n:4d}  reward={r:+.3f}  "
                  f"AC-AUROC={ag:.3f}  AUROC={au:.3f}")
        print(f"  POOLED  reward@pi={pooled['reward_at_pi']:+.4f}  AC-AUROC={pooled['age_auroc']:.4f}  "
              f"AUROC={pooled['auroc']:.4f}  AUPRC={pooled['auprc']:.4f}")
        print(f"  THRESHOLD  reward: pi={r_pi:+.4f}  transfer(deployable)={r_tr:+.4f}  "
              f"oracle(ceiling)={r_or:+.4f}")

    print("\n" + "=" * 92)
    print(f"{'arm':<36}{'AC-AUROC':>10}{'AUROC':>9}{'rwd@pi':>9}{'rwd@tr':>9}{'rwd@orac':>10}")
    print("-" * 92)
    for tag, (p, r_pi, r_tr, r_or) in results.items():
        print(f"{tag:<36}{p['age_auroc']:>10.4f}{p['auroc']:>9.4f}"
              f"{r_pi:>+9.4f}{r_tr:>+9.4f}{r_or:>+10.4f}")
    print("\nANCHOR (memory): AC-AUROC 0.6352 / AUROC 0.712 / reward@pi +0.274")
    print("DONE_RANKNORM")


if __name__ == "__main__":
    main()
