#!/usr/bin/env python3
"""Random 80/20 within-distribution eval — the headline the user asked for as a
LOSO counter-check.

WHY: LOSO (leave-one-site-out) is the honest NEW-SITE proxy and it's been flat
(~0.66). The user is (reasonably) worried LOSO is pessimistic for a test set that
may be drawn from the SAME distribution as training. A random 80/20 answers the
complementary question: if the holdout is in-distribution, how good are we? This
is the within-distribution ceiling; LOSO is the cross-site floor. Report BOTH.

SPLIT SEMANTICS: the production stack fits reward thresholds on a validation
slice, so a literal 80/20 with no val makes REWARD in-sample (inflated). We lock
a 20% test set and carve the threshold-fitting val out of the 80% dev portion:
    train 0.68 / val 0.12 / TEST 0.20   (val = 15% of the 80% dev set)
The threshold-FREE metrics (AUROC, AC-AUROC=age-conditioned AUROC gap=2, AUPRC)
depend only on predicted probabilities, so they are an honest 80/20 headline no
matter how val is carved -- those are the primary Challenge metrics. Reward is
reported too but is threshold-dependent (val-tuned, applied to the locked test).

Everything else = the exact production stack, reused from run_local_cv:
site-MoE + Kaiser fine-tune + BMI imputer + site_decade reward thresholds,
stratified by (label, site). Both cohorts built identically via
large_cohort_eval.build_matrix so this is apples-to-apples with the LOSO run.
Aggregate stats only leave the box.

Usage (on pdmle, as arshia_ilaty_physio26):
  PYTHONPATH=/data-temp/physio-viewer/bench/repo:/data-temp/physio-viewer/exports \
  python3 random_split_eval.py \
    --exports /data-temp/physio-viewer/exports \
    --repo /data-temp/physio-viewer/bench/repo --cvdir /data-temp/physio-viewer/bench \
    --out /data-temp/physio-viewer/exports/random_split_eval --seeds 50
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np


def run_split(X, y, ages, sites, fp, ev, rlc, names, train_frac, val_frac, seed):
    """One random stratified split -> production fit -> locked-test metrics.

    Mirrors run_local_cv.tvt exactly, but with configurable train/val fractions
    (tvt hardcodes 0.70/0.15). test_frac = 1 - train_frac - val_frac.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    train, val, test = rlc.make_splits(y, sites, train_frac=train_frac,
                                        val_frac=val_frac, seed=seed)
    imp = fp.fit_bmi_imputer(X[train], sites[train], names)
    Xtr = fp.apply_bmi_imputer(X[train], sites[train], imp)
    Xva = fp.apply_bmi_imputer(X[val], sites[val], imp)
    Xte = fp.apply_bmi_imputer(X[test], sites[test], imp)
    models = fp.fit_site_models(Xtr, y[train], sites[train])
    models = fp.fit_kaiser_finetuned(models, Xtr, y[train], sites[train])

    def predict(Xs, site_arr):
        return fp.predict_with_kaiser_override(models, Xs, site_arr,
                                               use_kaiser_finetuned=True)

    probs_va = predict(Xva, sites[val])
    reward_thr = fp.fit_reward_thresholds(y[val], probs_va, ages[val],
                                          sites[val], mode="site_decade")
    probs_te = predict(Xte, sites[test])
    bin_te = fp.apply_reward_thresholds(probs_te, ages[test], sites[test], reward_thr)
    prev_y, prev_ages = y[train], ages[train]
    a2p = ev.compute_prevalence(ages[test], prev_y, prev_ages, gap=2)
    return {
        "n_test": int(test.sum()), "n_train": int(train.sum()), "n_val": int(val.sum()),
        "prevalence": float(y[test].mean()),
        "reward": rlc.safe(ev.compute_reward, y[test], bin_te, ages[test], a2p),
        "reward_at_pi": rlc.safe(ev.compute_reward, y[test],
                                 (probs_te > float(prev_y.mean())).astype(int),
                                 ages[test], a2p),
        "auroc": rlc.safe(roc_auc_score, y[test], probs_te),
        "auprc": rlc.safe(average_precision_score, y[test], probs_te),
        "age_auroc": rlc.safe(ev.compute_auroc_age, y[test], probs_te, ages[test], 2),
        "age_weighted": rlc.safe(ev.compute_auroc_weighted, y[test], probs_te, ages[test], 2),
    }


def summarize(rows, keys):
    out = {}
    for k in keys:
        v = np.array([r[k] for r in rows], float)
        out[k] = dict(mean=float(np.nanmean(v)), sd=float(np.nanstd(v)),
                      lo=float(np.nanpercentile(v, 2.5)), hi=float(np.nanpercentile(v, 97.5)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cvdir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--train-frac", type=float, default=0.68)
    ap.add_argument("--val-frac", type=float, default=0.12)
    args = ap.parse_args()
    for p in (args.repo, args.cvdir, args.exports):
        if p not in sys.path:
            sys.path.insert(0, p)
    import feature_prep as fp        # noqa: F401  (used via run_split)
    import evaluate_model as ev
    import run_local_cv as rlc
    from large_cohort_eval import build_matrix, align, subset

    test_frac = 1.0 - args.train_frac - args.val_frac
    print(f"SPLIT: train={args.train_frac:.2f} val={args.val_frac:.2f} "
          f"TEST={test_frac:.2f}  (dev={args.train_frac+args.val_frac:.2f} / test={test_frac:.2f} "
          f"= '{round((args.train_frac+args.val_frac)*100)}/{round(test_frac*100)}')", flush=True)

    Xs, ys, ages_s, sites_s, names_s = build_matrix(args.exports, "standard")
    Xl, yl, ages_l, sites_l, names_l = build_matrix(args.exports, "large")
    common = align(names_s, names_l)
    Xs = subset(Xs, names_s, common)
    Xl = subset(Xl, names_l, common)
    print(f"standard: n={len(ys)} pos={int(ys.sum())} feats={Xs.shape[1]}", flush=True)
    print(f"large:    n={len(yl)} pos={int(yl.sum())} feats={Xl.shape[1]}", flush=True)
    print(f"common feature columns: {len(common)}", flush=True)

    keys = ["reward", "reward_at_pi", "auroc", "age_auroc", "age_weighted", "auprc"]
    results = {}
    for tag, (X, y, ages, sites) in [("standard", (Xs, ys, ages_s, sites_s)),
                                     ("large", (Xl, yl, ages_l, sites_l))]:
        print("\n" + "=" * 90, flush=True)
        print(f"COHORT {tag.upper()}  n={len(y)}  positives={int(y.sum())} "
              f"({100*y.mean():.1f}%)  features={X.shape[1]}", flush=True)
        print("=" * 90, flush=True)
        rows = []
        for sd in range(args.seeds):
            try:
                rows.append(run_split(X, y, ages, sites, fp, ev, rlc, common,
                                      args.train_frac, args.val_frac, sd))
            except Exception as e:
                print(f"      seed {sd} failed: {type(e).__name__}: {e}", flush=True)
        summ = summarize(rows, keys)
        print(f"  random {round((args.train_frac+args.val_frac)*100)}/{round(test_frac*100)} "
              f"over {len(rows)} seeds (mean ± SD  [2.5–97.5%]):", flush=True)
        for k in keys:
            s = summ[k]
            print(f"      {k:<14} {s['mean']:+.3f} ± {s['sd']:.3f}   "
                  f"[{s['lo']:+.3f}, {s['hi']:+.3f}]", flush=True)
        results[tag] = dict(n=int(len(y)), positives=int(y.sum()),
                            features=int(X.shape[1]), seeds=len(rows), metrics=summ,
                            median_test_n=int(np.median([r["n_test"] for r in rows])) if rows else 0)

    print("\n" + "=" * 90, flush=True)
    print(f"STANDARD vs LARGE — random {round((args.train_frac+args.val_frac)*100)}/"
          f"{round(test_frac*100)} (primary metrics, mean ± SD)", flush=True)
    print("=" * 90, flush=True)
    for m in ["age_auroc", "auroc", "auprc", "reward", "reward_at_pi"]:
        s = results["standard"]["metrics"][m]; l = results["large"]["metrics"][m]
        print(f"  {m:<14} standard={s['mean']:+.3f}±{s['sd']:.3f}   "
              f"large={l['mean']:+.3f}±{l['sd']:.3f}   Δmean={l['mean']-s['mean']:+.3f}"
              f"   ΔSD={l['sd']-s['sd']:+.3f}", flush=True)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "random_split_eval_summary.json"), "w") as fh:
            json.dump({"split": {"train": args.train_frac, "val": args.val_frac,
                                 "test": test_frac}, "results": results,
                       "common_features": common, "seeds": args.seeds}, fh, indent=2)
        print(f"\nwrote random_split_eval_summary.json to {args.out}", flush=True)
    print("DONE_RANDOM_SPLIT_EVAL", flush=True)


if __name__ == "__main__":
    main()
