#!/usr/bin/env python3
"""Feature-block ablation at FIXED cohort — does the advanced handcrafted work
actually add signal, or was the large-cohort win just more positives?

Deadline-day gate for the "port 438 features into team_code" decision. We hold
the STANDARD cohort fixed (same n, same 84 positives) and vary ONLY the feature
set, so the positives effect is removed and any delta is the FEATURES' doing:

  core+demo              base stage/arch-macro/event + BMI + demographic one-hot
  + arch                 transition-Markov architecture block
  + nk                   per-stage HRV block
  + micro                CAP micro-event block
  + rep                  report block
  ALL (438)              everything

For each: LOSO (honest cross-site proxy) AC-AUROC / reward@pi / AUROC, and a
random 80/20 (within-distribution) AC-AUROC over N seeds. If the advanced
blocks show robust lift on the PRIMARY metric (AC-AUROC) over core+demo, the
container port is justified. If flat, it isn't -- ship the working submission.

Production stack throughout (site-MoE + Kaiser + BMI-impute + site_decade
thresholds), reused from run_local_cv + random_split_eval. Aggregate only.

Usage (on pdmle):
  PYTHONPATH=/data-temp/physio-viewer/bench/repo:/data-temp/physio-viewer/exports \
  python3 ablation_features.py --exports /data-temp/physio-viewer/exports \
    --repo /data-temp/physio-viewer/bench/repo --cvdir /data-temp/physio-viewer/bench \
    --out /data-temp/physio-viewer/exports/ablation_features --seeds 30
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np


def block_of(name):
    if name.startswith("arch__"): return "arch"
    if name.startswith("nk__"): return "nk"
    if name.startswith("rep__"): return "rep"
    if name.startswith("micro__"): return "micro"
    if name.startswith(("sex_", "race_", "eth_")): return "demo"
    return "core"


def variant_idx(names, keep):
    return [i for i, n in enumerate(names) if block_of(n) in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cvdir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, default=30)
    args = ap.parse_args()
    for p in (args.repo, args.cvdir, args.exports):
        if p not in sys.path:
            sys.path.insert(0, p)
    import feature_prep as fp
    import evaluate_model as ev
    import run_local_cv as rlc
    from large_cohort_eval import build_matrix
    from random_split_eval import run_split

    X, y, ages, sites, names = build_matrix(args.exports, "standard")
    counts = {}
    for n in names:
        counts[block_of(n)] = counts.get(block_of(n), 0) + 1
    print(f"standard: n={len(y)} pos={int(y.sum())} feats={X.shape[1]}", flush=True)
    print(f"blocks: {counts}", flush=True)

    variants = [
        ("core+demo", {"core", "demo"}),
        ("+arch", {"core", "demo", "arch"}),
        ("+nk", {"core", "demo", "nk"}),
        ("+micro", {"core", "demo", "micro"}),
        ("+rep", {"core", "demo", "rep"}),
        ("ALL(438)", {"core", "demo", "arch", "nk", "rep", "micro"}),
    ]

    results = {}
    for tag, keep in variants:
        idx = variant_idx(names, keep)
        Xv = X[:, idx]
        nv = [names[i] for i in idx]
        # LOSO
        pooled, _ = rlc.loso(Xv, y, ages, sites, fp, ev, nv)
        # random 80/20 (train .68 / val .12 / test .20)
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
        print(f"\n[{tag:<10}] feats={len(idx):3d}", flush=True)
        print(f"    LOSO      AC-AUROC={r['loso']['ac_auroc']:.3f}  "
              f"AUROC={r['loso']['auroc']:.3f}  reward@pi={r['loso']['reward_at_pi']:+.3f}", flush=True)
        print(f"    80/20     AC-AUROC={r['tvt']['ac_auroc_mean']:.3f}±{r['tvt']['ac_auroc_sd']:.3f}  "
              f"AUROC={r['tvt']['auroc_mean']:.3f}±{r['tvt']['auroc_sd']:.3f}", flush=True)

    print("\n" + "=" * 78, flush=True)
    print("SUMMARY — delta vs core+demo (positive = advanced block HELPS)", flush=True)
    print("=" * 78, flush=True)
    base = results["core+demo"]
    for tag, _ in variants:
        r = results[tag]
        d_loso = r["loso"]["ac_auroc"] - base["loso"]["ac_auroc"]
        d_tvt = r["tvt"]["ac_auroc_mean"] - base["tvt"]["ac_auroc_mean"]
        print(f"  {tag:<10} feats={r['n_feats']:3d}  "
              f"LOSO AC-AUROC Δ={d_loso:+.3f}   80/20 AC-AUROC Δ={d_tvt:+.3f}", flush=True)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "ablation_features_summary.json"), "w") as fh:
            json.dump({"cohort": "standard", "n": int(len(y)), "pos": int(y.sum()),
                       "block_counts": counts, "seeds": args.seeds, "results": results}, fh, indent=2)
        print(f"\nwrote ablation_features_summary.json to {args.out}", flush=True)
    print("DONE_ABLATION_FEATURES", flush=True)


if __name__ == "__main__":
    main()
