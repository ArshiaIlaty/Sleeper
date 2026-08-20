#!/usr/bin/env python3
"""A/B the advanced features (batch 1 + batch 2) on LOSO, per family.

Compares local feature caches through the EXACT production stack (site-MoE +
Kaiser fine-tune + BMI imputer, per-fold training-prevalence reward threshold) by
calling run_local_cv.loso — no scoring logic is duplicated here. Prints pooled
LOSO reward@pi / best reward / AUROC / age-AUROC / AUPRC for each configuration
side by side, plus per-hold-out-site rewards.

At n=84 positives with correlated columns, the aggregate "all new features"
number hides which family earned the delta. So beyond OLD vs NEW we run a
**leave-one-family-out** ablation on the NEW cache: for each family we zero ONLY
that family's columns (columns dropped -> NaN, which HGB handles; rows identical),
so NEW − (NEW minus family) is that family's marginal contribution on the same
rows. We also report NEW with ALL new columns masked, which isolates the batch as
a whole and — compared to OLD — exposes any base-nk re-extraction drift.

Families (matched by column name in the plus cache):
  c1_hrv   C1 nonlinear HRV      nk__hrv_*_{dfa_a1,dfa_a2,sampen}[_ratio]
  arch     F1 transition Markov + A1 microarousals   arch__*
  micro    B3 SO-spindle + E2 RSWA + A2 CAP          micro__*

Usage (on pdmle, as arshia_ilaty_physio26):
  PYTHONPATH=/data-temp/physio-viewer/bench/repo python3 ab_batch1.py \
    --old  /data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.PREBATCH1.npz \
    --new  /data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz \
    --repo /data-temp/physio-viewer/bench/repo \
    --cvdir /data-temp/physio-viewer/bench
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

# C1 nonlinear-HRV columns only. These live under the nk__hrv_ prefix; the bare
# "_sampen" substring would otherwise also catch the PRE-EXISTING
# nk__eegc_<stage>_sampen_{mean,sd} EEG-complexity columns (present in the
# pre-batch snapshot too), which would over-attribute the C1 delta.
_C1_HRV_SUBSTR = ("_dfa_a1", "_dfa_a2", "_sampen")


def _is_c1_hrv(name):
    return name.startswith("nk__hrv_") and any(s in name for s in _C1_HRV_SUBSTR)


# family key -> (label, predicate on column name)
FAMILIES = {
    "c1_hrv": ("C1 nonlinear HRV (DFA/SampEn)", _is_c1_hrv),
    "arch":   ("F1 transitions + A1 arousals", lambda n: n.startswith("arch__")),
    "micro":  ("B3/E2/A2 microstructure",       lambda n: n.startswith("micro__")),
}


def is_new_col(name):
    """Any advanced-feature column (union of all families)."""
    return any(pred(name) for _, pred in FAMILIES.values())


def load_cache(path):
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def run_one(tag, cache, rlc, fp, ev, mask_pred=None, mask_note=""):
    """Run production LOSO on `cache`, optionally zeroing columns whose NAME
    satisfies `mask_pred` (dropped info; rows unchanged). Returns pooled dict."""
    X = cache["X"].astype(np.float32).copy()
    y = cache["y"].astype(int)
    ages = cache["ages"].astype(float)
    sites = np.asarray(cache["sites"])
    names = [str(n) for n in cache["feature_names"]]
    n_masked = 0
    if mask_pred is not None:
        cols = [i for i, n in enumerate(names) if mask_pred(n)]
        X[:, cols] = np.nan
        n_masked = len(cols)
    pooled, folds = rlc.loso(X, y, ages, sites, fp, ev, names)
    print(f"\n{tag}  [{X.shape[0]}x{X.shape[1]}]"
          + (f"  ({mask_note}: {n_masked} cols masked)" if mask_pred is not None else ""))
    for s, n, r, aa in folds:
        print(f"    hold-out {rlc.SITE_NAMES.get(s, s):>6} n={n:4d}  reward={r:+.3f}  age-AUROC={aa:.3f}")
    print(f"  POOLED reward@pi={pooled['reward_at_pi']:+.4f}  best={pooled['best_reward']:+.4f}"
          f"  AUROC={pooled['auroc']:.4f}  age-AUROC={pooled['age_auroc']:.4f}  AUPRC={pooled['auprc']:.4f}")
    return pooled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="pre-batch plus cache (.npz)")
    ap.add_argument("--new", required=True, help="new plus cache (.npz)")
    ap.add_argument("--repo", required=True, help="dir with feature_prep.py + evaluate_model.py")
    ap.add_argument("--cvdir", required=True, help="dir with run_local_cv.py")
    args = ap.parse_args()

    for p in (args.repo, args.cvdir):
        if p not in sys.path:
            sys.path.insert(0, p)
    import feature_prep as fp
    import evaluate_model as ev
    import run_local_cv as rlc

    old = load_cache(args.old)
    new = load_cache(args.new)
    new_names = [str(n) for n in new["feature_names"]]

    # how many columns each family actually contributes in the NEW cache
    fam_counts = {k: sum(1 for n in new_names if pred(n)) for k, (_, pred) in FAMILIES.items()}

    print("=" * 88)
    print("ADVANCED-FEATURES A/B — production LOSO (site-MoE + Kaiser + BMI-impute), reward@pi primary")
    print("=" * 88)
    for k, (label, _) in FAMILIES.items():
        print(f"  family {k:<8} {label:<34} {fam_counts[k]:>3d} cols")

    p_old = run_one("OLD  (pre-batch)", old, rlc, fp, ev)
    p_new = run_one("NEW  (+all families)", new, rlc, fp, ev)

    # batch as a whole (all new columns masked) — vs OLD reveals base-nk drift
    p_none = run_one("NEW  (all new masked)", new, rlc, fp, ev,
                     mask_pred=is_new_col, mask_note="all new")

    # leave-one-family-out: NEW minus exactly one family
    p_loo = {}
    for k, (label, pred) in FAMILIES.items():
        if fam_counts[k] == 0:
            continue
        p_loo[k] = run_one(f"NEW  (-{k})", new, rlc, fp, ev,
                           mask_pred=pred, mask_note=f"drop {k}")

    keys = ["reward_at_pi", "best_reward", "auroc", "age_auroc", "auprc"]
    print("\n" + "=" * 88)
    print("SUMMARY — primary = pooled LOSO reward@pi")
    print("=" * 88)
    for kk in keys:
        print(f"  {kk:<14} OLD={p_old[kk]:+.4f}  NEW={p_new[kk]:+.4f}"
              f"  Δ(NEW-OLD)={p_new[kk]-p_old[kk]:+.4f}   NEW-allmasked={p_none[kk]:+.4f}")

    print("\nPER-FAMILY marginal contribution  (NEW − NEW-without-family; +ve = family helps):")
    print(f"  {'family':<8} {'cols':>4}  " + "  ".join(f"{kk:>12}" for kk in keys))
    for k in FAMILIES:
        if k not in p_loo:
            continue
        deltas = [p_new[kk] - p_loo[k][kk] for kk in keys]
        print(f"  {k:<8} {fam_counts[k]:>4}  " + "  ".join(f"{d:+12.4f}" for d in deltas))

    print("\nInterpretation:")
    print("  * NEW vs OLD: total effect, but includes any base-nk re-extraction drift.")
    print("  * NEW vs NEW-allmasked: the batch as a whole, on IDENTICAL rows (drift-free).")
    print("  * per-family row: that family's MARGINAL value with all other new features present.")
    print("    At n=84 positives, trust reward@pi/age-AUROC direction over small AUROC wobble.")


if __name__ == "__main__":
    main()
