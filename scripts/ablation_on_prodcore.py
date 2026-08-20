#!/usr/bin/env python3
"""THE gating test: do advanced blocks add on top of the PRODUCTION core
(which already includes the whole-night autonomic HRV/SpO2 summary)?

The earlier CSV ablation lacked autonomic, so arch/micro there may only have
been *recovering* signal autonomic already carries. This uses the plus cache
(feature_matrix_local_plus.npz, built from the raw-EDF waveform pass) whose
core block DOES contain autonomic -- so a positive delta here means the block
adds signal PRODUCTION DOES NOT ALREADY HAVE. That, and only that, justifies
paying the deadline-day container-port cost.

prod_core = the 55 core/other features MINUS age (age -> scoring array only,
matching preset `submit` which excludes age as a feature). Variants add the
arch__ / micro__ / nk__ / rep__ blocks on top. Metric = LOSO AC-AUROC (cross-
site, the number that actually needs to move) + random 80/20 AC-AUROC.

Production stack (site-MoE + Kaiser + BMI-impute + site_decade thresholds),
reused from run_local_cv + random_split_eval. Aggregate only.

Usage (on pdmle):
  PYTHONPATH=/data-temp/physio-viewer/bench/repo:/data-temp/physio-viewer/exports \
  python3 ablation_on_prodcore.py \
    --cache /data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz \
    --repo /data-temp/physio-viewer/bench/repo --cvdir /data-temp/physio-viewer/bench \
    --out /data-temp/physio-viewer/exports/ablation_prodcore --seeds 30
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np


def block_of(name):
    for p, tag in (("arch__", "arch"), ("nk__", "nk"), ("rep__", "rep"), ("micro__", "micro")):
        if name.startswith(p):
            return tag
    return "core"   # includes demographics + autonomic + CAISR macro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cvdir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args()
    for p in (args.repo, args.cvdir, os.path.dirname(os.path.abspath(__file__))):
        if p not in sys.path:
            sys.path.insert(0, p)
    import feature_prep as fp
    import evaluate_model as ev
    import run_local_cv as rlc
    from random_split_eval import run_split

    c = np.load(args.cache, allow_pickle=True)
    X = c["X"].astype(np.float32); y = c["y"].astype(int)
    ages = c["ages"].astype(float); sites = np.asarray([str(s) for s in c["sites"]])
    names = [str(n) for n in c["feature_names"]]
    # age is a feature column in the cache; production `submit` excludes it (used
    # only for scoring/thresholds). Drop it from X so the core matches production.
    age_i = names.index("age") if "age" in names else None
    keep_cols = [i for i in range(len(names)) if i != age_i]
    X = X[:, keep_cols]; names = [names[i] for i in keep_cols]
    print(f"cache: n={len(y)} pos={int(y.sum())} feats={X.shape[1]} (age dropped from features)", flush=True)
    from collections import Counter
    print(f"blocks: {dict(Counter(block_of(n) for n in names))}", flush=True)

    def idx_for(keep):
        return [i for i, n in enumerate(names) if block_of(n) in keep]

    variants = [
        ("prod_core", {"core"}),
        ("+arch", {"core", "arch"}),
        ("+micro", {"core", "micro"}),
        ("+arch+micro", {"core", "arch", "micro"}),
        ("+nk", {"core", "nk"}),
        ("+rep", {"core", "rep"}),
        ("ALL", {"core", "arch", "micro", "nk", "rep"}),
    ]

    results = {}
    for tag, keep in variants:
        idx = idx_for(keep)
        Xv = X[:, idx]; nv = [names[i] for i in idx]
        pooled, _ = rlc.loso(Xv, y, ages, sites, fp, ev, nv)
        ac, au = [], []
        for sd in range(args.seeds):
            try:
                r = run_split(Xv, y, ages, sites, fp, ev, rlc, nv, 0.68, 0.12, sd)
                ac.append(r["age_auroc"]); au.append(r["auroc"])
            except Exception as e:
                print(f"    {tag} seed {sd}: {type(e).__name__}: {e}", flush=True)
        ac, au = np.array(ac, float), np.array(au, float)
        results[tag] = dict(
            n_feats=len(idx),
            loso=dict(ac_auroc=pooled["age_auroc"], auroc=pooled["auroc"],
                      reward_at_pi=pooled["reward_at_pi"], auprc=pooled["auprc"]),
            tvt=dict(ac_auroc_mean=float(np.nanmean(ac)), ac_auroc_sd=float(np.nanstd(ac)),
                     auroc_mean=float(np.nanmean(au)), auroc_sd=float(np.nanstd(au))))
        r = results[tag]
        print(f"\n[{tag:<12}] feats={len(idx):3d}", flush=True)
        print(f"    LOSO   AC-AUROC={r['loso']['ac_auroc']:.3f}  AUROC={r['loso']['auroc']:.3f}  "
              f"reward@pi={r['loso']['reward_at_pi']:+.3f}", flush=True)
        print(f"    80/20  AC-AUROC={r['tvt']['ac_auroc_mean']:.3f}±{r['tvt']['ac_auroc_sd']:.3f}", flush=True)

    print("\n" + "=" * 78, flush=True)
    print("SUMMARY — delta vs prod_core (positive = block adds signal OVER autonomic)", flush=True)
    print("=" * 78, flush=True)
    base = results["prod_core"]
    for tag, _ in variants:
        r = results[tag]
        d_loso = r["loso"]["ac_auroc"] - base["loso"]["ac_auroc"]
        d_tvt = r["tvt"]["ac_auroc_mean"] - base["tvt"]["ac_auroc_mean"]
        print(f"  {tag:<12} feats={r['n_feats']:3d}  "
              f"LOSO Δ={d_loso:+.3f}   80/20 Δ={d_tvt:+.3f}", flush=True)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "ablation_prodcore_summary.json"), "w") as fh:
            json.dump({"n": int(len(y)), "pos": int(y.sum()), "seeds": args.seeds,
                       "results": results}, fh, indent=2)
        print(f"\nwrote ablation_prodcore_summary.json to {args.out}", flush=True)
    print("DONE_ABLATION_PRODCORE", flush=True)


if __name__ == "__main__":
    main()
