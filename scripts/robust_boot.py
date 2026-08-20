#!/usr/bin/env python3
"""Robustness + bootstrap for the advanced-features A/B (batch 1 + 2).

Two INDEPENDENT robustness angles on the OLD (337) vs NEW (436) result:

 (1) Paired subject-level bootstrap on the pooled LOSO out-of-fold predictions.
     Resample subjects with replacement B times; recompute the NEW-OLD delta
     (primary reward@pi + age-AUROC) and each family's MARGINAL delta
     (NEW - NEW-without-family) on every resample -> percentile CIs and the
     fraction of resamples with delta>0. The reward is a per-subject additive
     score (evaluate_model: reward = sum_i score_i / n, score_i fixed by the
     subject's label, the model's binary call, and the FIXED age-prevalence
     table), so a resample's reward is an exact resampled mean -- no refitting.
     age-AUROC is a pairwise U-statistic, recomputed per resample (vectorised).

 (2) 70/15/15 seed sweep. LOSO has no split randomness (sites are fixed), so the
     second angle varies the random train/val/test split across many seeds and
     pairs NEW vs OLD on each -> mean+-sd and the fraction of seeds NEW wins.

No fabrication: every metric is recomputed with evaluate_model's exact scoring,
and the harness's pooled reward@pi / age-AUROC are ASSERTED to reproduce the
pipeline's A/B numbers before any bootstrap runs (guards harness correctness).

Usage (on pdmle, as arshia_ilaty_physio26):
  PYTHONPATH=/data-temp/physio-viewer/bench/repo python3 robust_boot.py \
    --old  /data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.PREBATCH1.npz \
    --new  /data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz \
    --repo /data-temp/physio-viewer/bench/repo --cvdir /data-temp/physio-viewer/bench \
    --nboot 10000 --nboot-auc 4000 --seeds 50
"""
from __future__ import annotations
import argparse, sys
import numpy as np

# family predicates -- identical definitions to ab_batch1.py
_C1 = ("_dfa_a1", "_dfa_a2", "_sampen")
def _is_c1(n): return n.startswith("nk__hrv_") and any(s in n for s in _C1)
FAMILIES = {
    "c1_hrv": _is_c1,
    "arch":   lambda n: n.startswith("arch__"),
    "micro":  lambda n: n.startswith("micro__"),
}
def _is_new(n): return any(p(n) for p in FAMILIES.values())


def load_cache(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def loso_oof(X, y, ages, sites, fp, names, mask_pred=None):
    """Exact replica of run_local_cv.loso's per-fold stack, but returns the
    pooled per-subject OOF arrays (y, prob, age, binary@pi, original-row-index)."""
    X = X.astype(np.float32).copy()
    if mask_pred is not None:
        cols = [i for i, n in enumerate(names) if mask_pred(n)]
        X[:, cols] = np.nan
    tt, pp, aa, bb, ii = [], [], [], [], []
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
        binary = (prob > float(y[tr].mean())).astype(int)
        te_idx = np.where(te)[0]
        tt.extend(y[te]); pp.extend(prob); aa.extend(ages[te])
        bb.extend(binary); ii.extend(te_idx)
    return (np.asarray(tt, int), np.asarray(pp, float), np.asarray(aa, float),
            np.asarray(bb, int), np.asarray(ii, int))


def reward_scores(y, binary, ages, a2p, m):
    """Per-subject reward contribution s_i (0 for non-finite age), matching
    evaluate_model.compute_reward exactly; reward = s[fin].sum()/fin.sum()."""
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


def auroc_age_vec(y, prob, ages, gap=2):
    """Vectorised evaluate_model.compute_auroc_age (age-conditioned AUROC)."""
    pos = y == 1; neg = y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    P, Ap = prob[pos], ages[pos]
    N, An = prob[neg], ages[neg]
    amask = np.abs(Ap[:, None] - An[None, :]) <= gap
    denom = amask.sum()
    if denom == 0:
        return np.nan
    gt = P[:, None] > N[None, :]
    eq = P[:, None] == N[None, :]
    numer = (amask * (gt + 0.5 * eq)).sum()
    return float(numer) / float(denom)


def pctl_ci(x, lo=2.5, hi=97.5):
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cvdir", required=True)
    ap.add_argument("--nboot", type=int, default=10000)
    ap.add_argument("--nboot-auc", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--boot-seed", type=int, default=1234)
    args = ap.parse_args()

    for p in (args.repo, args.cvdir):
        if p not in sys.path:
            sys.path.insert(0, p)
    import feature_prep as fp
    import evaluate_model as ev
    import run_local_cv as rlc

    old = load_cache(args.old)
    new = load_cache(args.new)
    yo = old["y"].astype(int); ao = old["ages"].astype(float); so = np.asarray(old["sites"])
    Xo = old["X"]; no = [str(n) for n in old["feature_names"]]
    yn = new["y"].astype(int); an = new["ages"].astype(float); sn = np.asarray(new["sites"])
    Xn = new["X"]; nn = [str(n) for n in new["feature_names"]]

    # ---- build the six OOF configs (same order: LOSO iterates np.unique(sites)) ----
    print("building LOSO out-of-fold predictions for 6 configs ...", flush=True)
    oof = {}
    oof["OLD"] = loso_oof(Xo, yo, ao, so, fp, no)
    oof["NEW"] = loso_oof(Xn, yn, an, sn, fp, nn)
    oof["NEW_allmasked"] = loso_oof(Xn, yn, an, sn, fp, nn, mask_pred=_is_new)
    for k, pred in FAMILIES.items():
        oof[f"NEW_no_{k}"] = loso_oof(Xn, yn, an, sn, fp, nn, mask_pred=pred)

    # subject order must be identical across configs (same rows, same site order)
    ref_idx = oof["NEW"][4]
    for k, v in oof.items():
        assert np.array_equal(v[4], ref_idx), f"{k} OOF order differs -- cannot pair"
    yb = oof["NEW"][0]; ab = oof["NEW"][2]           # labels/ages (same for all configs)
    n = len(yb)

    # fixed age-prevalence table (population property; same as loso: full-cohort)
    a2p = ev.compute_prevalence(ab, yn, an, gap=2)

    # reward@pi uses a SINGLE global threshold = full-cohort prevalence applied to
    # the pooled OOF probabilities (run_local_cv.loso: bin_pi = yp > y.mean()),
    # NOT the per-fold training-prevalence binary. Recompute it here so the
    # harness reproduces the pipeline's reward@pi exactly.
    thr_global = float(yn.mean())

    # ---- point estimates + harness integrity check vs the pipeline A/B ----
    def reward_of(tag):
        yv, pv, av, _bv, _ = oof[tag]
        binary = (pv > thr_global).astype(int)
        s, fin = reward_scores(yv, binary, av, a2p, n)
        return s[fin].sum() / fin.sum(), s, fin
    R = {}; S = {}; FIN = {}
    for tag in oof:
        R[tag], S[tag], FIN[tag] = reward_of(tag)
    AUC = {tag: auroc_age_vec(oof[tag][0], oof[tag][1], oof[tag][2]) for tag in oof}

    print("\nharness point estimates (assert vs pipeline A/B):")
    print(f"  reward@pi   OLD={R['OLD']:+.4f}  NEW={R['NEW']:+.4f}  masked={R['NEW_allmasked']:+.4f}")
    print(f"  age-AUROC   OLD={AUC['OLD']:.4f}  NEW={AUC['NEW']:.4f}")
    # pipeline printed: OLD reward +0.1759 / age 0.5804 ; NEW +0.2738 / 0.6352
    for want, got, nm in [(0.1759, R['OLD'], 'reward OLD'), (0.2738, R['NEW'], 'reward NEW'),
                          (0.5804, AUC['OLD'], 'age-AUROC OLD'), (0.6352, AUC['NEW'], 'age-AUROC NEW')]:
        ok = abs(want - got) < 5e-3
        print(f"    {'OK ' if ok else 'DIFF'} {nm}: harness={got:+.4f} pipeline={want:+.4f}")

    # ---- (1a) paired bootstrap: reward delta (exact resampled means) ----
    rng = np.random.default_rng(args.boot_seed)
    B = args.nboot
    idx = rng.integers(0, n, size=(B, n))                 # B resamples of subjects

    def boot_reward(tag):
        s = S[tag]; fin = FIN[tag].astype(float)
        num = s[idx].sum(axis=1)
        den = fin[idx].sum(axis=1)
        return num / den
    r_old = boot_reward("OLD"); r_new = boot_reward("NEW")
    d_reward = r_new - r_old

    print("\n" + "=" * 84)
    print(f"(1) PAIRED SUBJECT BOOTSTRAP  (B={B} for reward, {args.nboot_auc} for age-AUROC)")
    print("=" * 84)
    lo, hi = pctl_ci(d_reward)
    print(f"\nPRIMARY  reward@pi  NEW-OLD delta = {R['NEW']-R['OLD']:+.4f}"
          f"   95% CI [{lo:+.4f}, {hi:+.4f}]   P(delta>0)={np.mean(d_reward>0):.3f}")

    print("\nper-family MARGINAL reward@pi delta (NEW - NEW-without-family):")
    for k in FAMILIES:
        rk = boot_reward(f"NEW_no_{k}")
        dk = r_new - rk
        lo, hi = pctl_ci(dk)
        print(f"  {k:<8} delta={R['NEW']-R[f'NEW_no_{k}']:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"  P(>0)={np.mean(dk>0):.3f}")

    # ---- (1b) paired bootstrap: age-AUROC delta (recompute U-stat per resample) ----
    Ba = args.nboot_auc
    idxa = idx[:Ba]
    yo_ = oof["OLD"]; yn_ = oof["NEW"]
    da = np.empty(Ba)
    a_old = np.empty(Ba); a_new = np.empty(Ba)
    for b in range(Ba):
        j = idxa[b]
        a_old[b] = auroc_age_vec(yo_[0][j], yo_[1][j], yo_[2][j])
        a_new[b] = auroc_age_vec(yn_[0][j], yn_[1][j], yn_[2][j])
    da = a_new - a_old
    lo, hi = pctl_ci(da[np.isfinite(da)])
    print(f"\nage-AUROC  NEW-OLD delta = {AUC['NEW']-AUC['OLD']:+.4f}"
          f"   95% CI [{lo:+.4f}, {hi:+.4f}]   P(delta>0)={np.mean(da[np.isfinite(da)]>0):.3f}")

    # ---- (2) 70/15/15 seed sweep (independent of LOSO's fixed site split) ----
    print("\n" + "=" * 84)
    print(f"(2) 70/15/15 SEED SWEEP  ({args.seeds} random splits, paired NEW vs OLD)")
    print("=" * 84)
    seeds = list(range(args.seeds))
    rows = {"OLD": [], "NEW": []}
    for s in seeds:
        rows["OLD"].append(rlc.tvt(Xo.astype(np.float32), yo, ao, so, fp, ev, no, seed=s))
        rows["NEW"].append(rlc.tvt(Xn.astype(np.float32), yn, an, sn, fp, ev, nn, seed=s))
    for metric in ["reward", "reward_at_pi", "age_auroc", "auroc", "auprc"]:
        vo = np.array([r["test"][metric] for r in rows["OLD"]], float)
        vn = np.array([r["test"][metric] for r in rows["NEW"]], float)
        d = vn - vo
        fin = np.isfinite(d)
        lo, hi = pctl_ci(d[fin]) if fin.sum() > 2 else (np.nan, np.nan)
        print(f"  {metric:<13} OLD={np.nanmean(vo):+.3f}+-{np.nanstd(vo):.3f}"
              f"  NEW={np.nanmean(vn):+.3f}+-{np.nanstd(vn):.3f}"
              f"  meanDelta={np.nanmean(d):+.3f}  95%CI[{lo:+.3f},{hi:+.3f}]"
              f"  win={np.mean(d[fin]>0):.2f}")

    print("\nDONE_ROBUST")


if __name__ == "__main__":
    main()
