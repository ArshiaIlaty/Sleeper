#!/usr/bin/env python3
"""Age-conditioned Bayes decision threshold A/B (challenge improvement #1).

The prevalence-weighted reward's Bayes-optimal rule is: predict positive iff the
model's risk q exceeds the AGE-SPECIFIC prevalence p(age), NOT the global
prevalence. Derivation from evaluate_model.compute_reward payoffs
(TP=1/p-1, FP=-1, FN=-1, TN=1/(1-p)-1): E[score|predict 1] > E[score|predict 0]
reduces to q(1-p) > p(1-q)  <=>  q > p. The production LOSO path currently
thresholds at prob > y.mean() (a single global 0.076), so this is pure
metric-alignment: same model, better decision rule.

The model excludes age, so q is "risk from sleep alone"; q > p(age) fires when a
subject looks worse than typical FOR THEIR AGE -- exactly what the age-discounted
metric rewards.

Applies several decision rules to the SAME pooled LOSO OOF probabilities (model
identical across rules -> isolates the threshold), scored with evaluate_model's
exact reward + the SAME full-cohort age-prevalence table the pipeline uses:

  global_pi       q > y.mean()                    (current production rule)
  global_grid     best single global thr on OOF   (oracle single-thr ceiling)
  ageBayes_oracle q > p_full(age)                 (decision uses full-cohort p =
                                                   theoretical ceiling of the idea)
  ageBayes_train  q > p_train(age)                (decision uses TRAIN-fold-only p,
                                                   LOSO-safe = honest deployable)

Oracle vs train decomposes the idea: if oracle doesn't help, the rule is wrong;
if oracle helps but train doesn't, prevalence ESTIMATION is the bottleneck.

Paired subject bootstrap on the reward delta (ageBayes - global_pi); reward is a
per-subject additive score so the resample is an exact resampled mean.

Usage (on pdmle, as arshia_ilaty_physio26):
  PYTHONPATH=/data-temp/physio-viewer/bench/repo python3 age_threshold_ab.py \
    --new /data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz \
    --old /data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.PREBATCH1.npz \
    --repo /data-temp/physio-viewer/bench/repo --cvdir /data-temp/physio-viewer/bench \
    --nboot 10000
"""
from __future__ import annotations
import argparse, sys
import numpy as np


def load_cache(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def loso_oof_with_train_prev(X, y, ages, sites, fp, ev, names, gap=2):
    """Run the production LOSO stack; return pooled OOF (y, prob, age) PLUS, for
    each held-out subject, the age-specific prevalence estimated from the TRAIN
    fold only (LOSO-safe) -- the deployable p(age)."""
    X = X.astype(np.float32)
    tt, pp, aa, ptrain = [], [], [], []
    for site in np.unique(sites):
        te = sites == site
        tr = ~te
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        imp = fp.fit_bmi_imputer(X[tr], sites[tr], names)
        Xtr = fp.apply_bmi_imputer(X[tr], sites[tr], imp)
        Xte = fp.apply_bmi_imputer(X[te], sites[te], imp)
        models = fp.fit_site_models(Xtr, y[tr], sites[tr])
        models = fp.fit_kaiser_finetuned(models, Xtr, y[tr], sites[tr])
        prob = fp.predict_with_kaiser_override(models, Xte, sites[te], use_kaiser_finetuned=True)
        # p(age) for held-out subjects, estimated from TRAIN labels/ages only
        p_tr = ev.compute_prevalence(ages[te], y[tr], ages[tr], gap=gap)
        tt.extend(y[te]); pp.extend(prob); aa.extend(ages[te])
        ptrain.extend([p_tr.get(a, np.nan) for a in ages[te]])
    return (np.asarray(tt, int), np.asarray(pp, float),
            np.asarray(aa, float), np.asarray(ptrain, float))


def reward_scores(y, binary, ages, a2p, m):
    """Per-subject reward contribution (evaluate_model.compute_reward, exact)."""
    lo, hi = 0.5 / m, 1.0 - 0.5 / m
    fin = np.isfinite(ages)
    p = np.array([min(max(a2p.get(a, 0.5), lo), hi) if f else 1.0
                  for a, f in zip(ages, fin)], float)
    s = np.zeros(len(y), float)
    tp = fin & (y == 1) & (binary == 1); s[tp] = 1.0 / p[tp] - 1.0
    fp_ = fin & (y == 0) & (binary == 1); s[fp_] = -1.0
    fn = fin & (y == 1) & (binary == 0); s[fn] = -1.0
    tn = fin & (y == 0) & (binary == 0); s[tn] = 1.0 / (1.0 - p[tn]) - 1.0
    return s, fin


def reward_from_binary(y, binary, ages, a2p, m):
    s, fin = reward_scores(y, binary, ages, a2p, m)
    return s[fin].sum() / fin.sum(), s, fin


def run_cache(tag, cache, fp, ev, args):
    y = cache["y"].astype(int); ages = cache["ages"].astype(float)
    sites = np.asarray(cache["sites"]); X = cache["X"]
    names = [str(n) for n in cache["feature_names"]]
    yt, yp, ya, p_tr = loso_oof_with_train_prev(X, y, ages, sites, fp, ev, names, gap=2)
    n = len(yt)

    # SCORING prevalence table: identical to the pipeline (full cohort, gap 2)
    a2p = ev.compute_prevalence(ya, y, ages, gap=2)
    p_full = np.array([a2p.get(a, np.nan) for a in ya], float)  # decision-side oracle p

    pi_global = float(y.mean())

    # ---- decision rules (all applied to the SAME probabilities yp) ----
    rules = {}
    rules["global_pi"] = (yp > pi_global).astype(int)

    # best single global threshold on the OOF (oracle single-threshold ceiling)
    best_t, best_r = pi_global, -1e9
    for t in np.linspace(0.01, 0.60, 120):
        r, _, _ = reward_from_binary(yt, (yp > t).astype(int), ya, a2p, n)
        if r > best_r:
            best_r, best_t = r, t
    rules["global_grid"] = (yp > best_t).astype(int)

    # age-conditioned Bayes: q > p(age). oracle uses full-cohort p; train uses LOSO-safe p.
    pf = np.where(np.isfinite(p_full), p_full, pi_global)
    pt = np.where(np.isfinite(p_tr),   p_tr,   pi_global)
    rules["ageBayes_oracle"] = (yp > pf).astype(int)
    rules["ageBayes_train"]  = (yp > pt).astype(int)

    # ---- metrics per rule ----
    def auroc(yy, pp_): return ev.compute_auroc(yy, pp_)
    def auprc(yy, pp_): return ev.compute_auprc(yy, pp_)
    ageauc = ev.compute_auroc_age(yt, yp, ya, 2)   # threshold-independent

    print(f"\n{'='*84}\n{tag}   [n={n}, pos={int(yt.sum())}, pi_global={pi_global:.4f}, best_grid_thr={best_t:.3f}]")
    print(f"  (age-AUROC={ageauc:.4f}, AUROC={auroc(yt,yp):.4f}, AUPRC={auprc(yt,yp):.4f} -- threshold-independent)")
    print(f"  {'rule':<18} {'reward':>9}  {'n_pos_pred':>10}  {'sens':>6} {'spec':>6}")
    out = {}
    for name, binary in rules.items():
        r, s, fin = reward_from_binary(yt, binary, ya, a2p, n)
        tp = int(((yt == 1) & (binary == 1)).sum()); pos = int((yt == 1).sum())
        tn = int(((yt == 0) & (binary == 0)).sum()); neg = int((yt == 0).sum())
        sens = tp / pos if pos else float("nan")
        spec = tn / neg if neg else float("nan")
        print(f"  {name:<18} {r:+9.4f}  {int(binary.sum()):>10}  {sens:6.3f} {spec:6.3f}")
        out[name] = {"reward": r, "scores": s, "fin": fin, "binary": binary}
    return {"n": n, "yt": yt, "ya": ya, "a2p": a2p, "rules": out}


def bootstrap_delta(res, rule_a, rule_b, nboot, seed=1234):
    """Paired subject bootstrap of reward(rule_a) - reward(rule_b)."""
    n = res["n"]
    sa = res["rules"][rule_a]["scores"]; fa = res["rules"][rule_a]["fin"].astype(float)
    sb = res["rules"][rule_b]["scores"]; fb = res["rules"][rule_b]["fin"].astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(nboot, n))
    ra = sa[idx].sum(1) / fa[idx].sum(1)
    rb = sb[idx].sum(1) / fb[idx].sum(1)
    d = ra - rb
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(np.mean(d > 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", default=None)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cvdir", required=True)
    ap.add_argument("--nboot", type=int, default=10000)
    args = ap.parse_args()
    for p in (args.repo, args.cvdir):
        if p not in sys.path:
            sys.path.insert(0, p)
    import feature_prep as fp
    import evaluate_model as ev

    caches = [("NEW (436)", args.new)]
    if args.old:
        caches.append(("OLD (337)", args.old))

    for tag, path in caches:
        res = run_cache(tag, load_cache(path), fp, ev, args)
        for rb in ("ageBayes_oracle", "ageBayes_train"):
            lo, hi, pgt = bootstrap_delta(res, rb, "global_pi", args.nboot)
            d = res["rules"][rb]["reward"] - res["rules"]["global_pi"]["reward"]
            print(f"  boot  {rb:<16} - global_pi  reward Δ={d:+.4f}"
                  f"  95% CI [{lo:+.4f}, {hi:+.4f}]  P(Δ>0)={pgt:.3f}")

    print("\nDONE_AGE_THR")


if __name__ == "__main__":
    main()
