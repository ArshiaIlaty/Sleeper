#!/usr/bin/env python3
"""Calibration / reliability probe (LOSO, large cohort).

Answers three questions the challenge metrics never scored (the official scorer has NO
calibration metric — only reward/AUROC/AC-AUROC/AUPRC):

  1. Does calibration DRIFT by site?  -> per held-out-site Brier, ECE, and reliability bins,
     for BOTH the uncalibrated base HGB and the production isotonic-calibrated model.
  2. Does the isotonic step even HELP cross-site reward, or is it neutralized by our
     threshold tuning?  -> reward under the four decision rules (pi/transfer/oracle/q_gt_pa),
     isotonic-ON vs isotonic-OFF, on the IDENTICAL LOSO OOF.
  3. Calibration slope / intercept per held-out site (logistic recalibration of logit(p) vs y;
     slope=1, intercept=0 = perfect; slope<1 = over-confident, drift-in-the-large via intercept).

Two feature representations: raw features (what the shipped submission uses) and per-site
rank-norm (the champion analysis stack where the +0.171 transfer-threshold lever lives).
Both go through the production BMI imputer + the EXACT feature_prep HGB params. Everything is
LOSO-honest; only aggregate stats (per-site metrics + reliability BINS, not rows) are emitted.

Usage:
  PYTHONPATH=<USP>:<repo>:<this-dir> python3 calib_probe.py \
      --exports <exports dir with {fam}_large.csv> --repo <repo> --levers-dir <dir> \
      --cohort large --out /tmp/calib/calib_metrics.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np


# ----------------------------------------------------------------- calibration metrics
def brier(y, p):
    return float(np.mean((p - y) ** 2))


def ece(y, p, n_bins=10):
    """Expected Calibration Error: sum_b (n_b/N) |obs_freq_b - mean_pred_b|, 10 equal-width bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    N = len(y); e = 0.0
    for b in range(n_bins):
        m = idx == b
        nb = int(m.sum())
        if nb == 0:
            continue
        e += (nb / N) * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(e)


def mce(y, p, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    worst = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        worst = max(worst, abs(float(y[m].mean()) - float(p[m].mean())))
    return float(worst)


def reliability_bins(y, p, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        nb = int(m.sum())
        out.append({
            "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": nb,
            "mean_pred": (float(p[m].mean()) if nb else None),
            "obs_freq": (float(y[m].mean()) if nb else None),
        })
    return out


def calib_slope_intercept(y, p):
    """Logistic recalibration: fit y ~ a + b*logit(p_hat). b=slope (ideal 1), a=intercept (ideal 0).
    b<1 => predictions too extreme (over-confident). Near-unpenalized logistic."""
    from sklearn.linear_model import LogisticRegression
    if len(np.unique(y)) < 2:
        return None, None
    eps = 1e-6
    pc = np.clip(p, eps, 1 - eps)
    lo = np.log(pc / (1 - pc)).reshape(-1, 1)
    lr = LogisticRegression(C=1e9, solver="lbfgs", max_iter=2000)
    lr.fit(lo, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


# ----------------------------------------------------------------- production base HGB (raw arm)
def make_base_hgb():
    """Identical params to feature_prep.fit_clf's base (the uncalibrated 'isotonic-OFF' arm)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, random_state=42,
    )


def loso_raw_and_cal(X, y, ages, sites, fp, ev, names):
    """LOSO producing BOTH the uncalibrated base-HGB OOF (raw) and the production
    isotonic-calibrated OOF (cal). fp.fit_clf gives the calibrated arm exactly as shipped."""
    osite, yt, ya, praw, pcal = [], [], [], [], []
    per_site_auroc = {}
    for site in np.unique(sites):
        te = sites == site; tr = ~te
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        imp = fp.fit_bmi_imputer(X[tr], sites[tr], names)
        Xtr = fp.apply_bmi_imputer(X[tr], sites[tr], imp)
        Xte = fp.apply_bmi_imputer(X[te], sites[te], imp)
        base = make_base_hgb(); base.fit(Xtr, y[tr])
        pr = base.predict_proba(Xte)[:, 1]
        pc = fp.fit_clf(Xtr, y[tr]).predict_proba(Xte)[:, 1]  # production isotonic-wrapped
        osite += [str(site)] * int(te.sum())
        yt.extend(y[te]); ya.extend(ages[te]); praw.extend(pr); pcal.extend(pc)
        per_site_auroc[str(site)] = {
            "raw": float(ev.compute_auroc(y[te], pr)), "cal": float(ev.compute_auroc(y[te], pc))}
    return (np.asarray(osite), np.asarray(yt), np.asarray(ya),
            np.asarray(praw), np.asarray(pcal), per_site_auroc)


def calib_block(y, p):
    slope, intercept = calib_slope_intercept(y, p)
    return {"n": int(len(y)), "pos_rate": float(np.mean(y)), "brier": brier(y, p),
            "ece": ece(y, p), "mce": mce(y, p), "slope": slope, "intercept": intercept,
            "bins": reliability_bins(y, p)}


def run_rep(tag, Xin, y, ages, sites, fp, ev, lv, names, site_names):
    osite, yt, ya, praw, pcal, psa = loso_raw_and_cal(Xin, y, ages, sites, fp, ev, names)

    # reward A/B: four decision rules on identical OOF, isotonic OFF (raw) vs ON (cal)
    dr_raw = lv.decision_rules((osite, yt, praw, ya), y, ages, ev)
    dr_cal = lv.decision_rules((osite, yt, pcal, ya), y, ages, ev)
    pooled = {
        "auroc_raw": float(ev.compute_auroc(yt, praw)), "auroc_cal": float(ev.compute_auroc(yt, pcal)),
        "raw": calib_block(yt, praw), "cal": calib_block(yt, pcal)}

    per_site = {}
    for s in np.unique(osite):
        m = osite == s
        per_site[s] = {"site_name": site_names.get(s, s),
                       "raw": calib_block(yt[m], praw[m]), "cal": calib_block(yt[m], pcal[m]),
                       "auroc": psa[s]}
    return {"reward": {"raw": dr_raw, "cal": dr_cal}, "pooled": pooled, "per_site": per_site}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--levers-dir", required=True)
    ap.add_argument("--cohort", default="large")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for p in [a.repo, a.levers_dir]:
        if p and p not in sys.path:
            sys.path.insert(0, p)
    import feature_prep as fp
    import evaluate_model as ev
    import levers_test as lv

    X, y, ages, sites, names, bmi_idx = lv.load_csv(a.exports, a.cohort)
    protect = [bmi_idx] if bmi_idx is not None else []
    print(f"n={len(y)} pos={int(y.sum())} feats={X.shape[1]} "
          f"sites={ {s: int((sites == s).sum()) for s in np.unique(sites)} }", flush=True)

    Xr = lv.site_rank(X, sites, protect)
    reps = {}
    for tag, Xin in [("raw_feats", X), ("rank_norm", Xr)]:
        print(f"[calib] running rep={tag} ...", flush=True)
        reps[tag] = run_rep(tag, Xin, y, ages, sites, fp, ev, lv, names, lv.SITE_NAMES)

    result = {"cohort": a.cohort, "n": int(len(y)), "pos": int(y.sum()),
              "sites": {s: int((sites == s).sum()) for s in np.unique(sites)}, "reps": reps}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(result, fh, indent=2)

    # ---- readable tables to the log ----
    for tag in ("raw_feats", "rank_norm"):
        R = reps[tag]
        print("\n" + "=" * 90)
        print(f"REP = {tag}")
        print("-" * 90)
        rw, rc = R["reward"]["raw"], R["reward"]["cal"]
        print("reward (isotonic OFF -> ON):")
        for k in ("pi", "transfer", "oracle", "q_gt_pa"):
            print(f"    {k:<9} {rw[k]:+.4f} -> {rc[k]:+.4f}   (Δ {rc[k]-rw[k]:+.4f})")
        pr, pc = R["pooled"]["raw"], R["pooled"]["cal"]
        print(f"pooled AUROC raw={R['pooled']['auroc_raw']:.4f} cal={R['pooled']['auroc_cal']:.4f}"
              f"  (isotonic is monotonic -> ranking ~unchanged)")
        print(f"pooled  Brier {pr['brier']:.4f}->{pc['brier']:.4f}  ECE {pr['ece']:.4f}->{pc['ece']:.4f}"
              f"  slope {pr['slope']:.3f}->{pc['slope']:.3f}  intercept {pr['intercept']:+.3f}->{pc['intercept']:+.3f}")
        print(f"{'site':<8}{'n':>6}{'pos%':>7}  | OFF: Brier  ECE   slope  int   | ON: Brier  ECE   slope  int")
        for s, blk in R["per_site"].items():
            o, c = blk["raw"], blk["cal"]
            print(f"{blk['site_name']:<8}{o['n']:>6}{100*o['pos_rate']:>6.1f}%  |"
                  f" {o['brier']:.4f} {o['ece']:.4f} {o['slope']:>5.2f} {o['intercept']:>+5.2f}  |"
                  f" {c['brier']:.4f} {c['ece']:.4f} {c['slope']:>5.2f} {c['intercept']:>+5.2f}")
    print(f"\nwrote {a.out}")
    print("DONE_CALIB")


if __name__ == "__main__":
    main()
