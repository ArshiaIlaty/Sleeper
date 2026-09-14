#!/usr/bin/env python3
"""Within-night (physiology-referenced) normalization — orthogonal lever on top of per-site rank.

Motivation. Per-site rank-norm (scripts/rank_norm/, the confirmed winning lever) removes the
BETWEEN-site batch effect by ranking each feature within its site. It cannot remove WITHIN-site,
per-NIGHT gain: if a recording has high electrode impedance / amplifier gain, every absolute
band-power on that night is inflated together, so it gets high ranks on all of them within its
site. The fix has to happen at the source, referenced to the subject's own night.

Our feature matrix is almost entirely STAGE-RESOLVED (wake/n1/n2/n3/rem[/nrem/sleep]); the design
hypothesis (FEATURE_LIST.md) is that cognitive impairment shows up as *blunted cross-stage
modulation*, and the team already hand-built a few cross-stage contrasts. This script makes that
systematic: for every stage-resolved metric it derives a within-night self-referenced feature, in
two flavors, and tests whether adding them ON TOP OF per-site rank lifts LOSO.

  ratio   f_stage / f_ref               (ref = a chosen reference stage, default n2)
          -> for positive metrics; multiplicative per-night gain cancels in the ratio.
  zwithin (f_stage - mean_stages)/std_stages   within the night, per metric across its stages
          -> sign-safe; captures modulation SHAPE; multiplicative per-night gain is divided out
             (mean and std both scale with the gain).

Arms (LOSO, HGB, all normalized with per-site rank via levers_test.site_rank):
  rank baseline                 full feature set, per-site rank              (anchor)
  rank + wn_ratio(<ref>)        + within-night ratio-over-reference block
  rank + wn_zwithin             + within-night z-across-stages block (all stage-resolved metrics)
  rank + wn_zwithin (abs-only)  + within-night z, restricted to uV-scale absolutes (the gain carriers)

Reuses levers_test.py (load_csv, site_rank, run_loso, decision_rules, GRID, ABS_AMP_RE) so the
plumbing is identical to the rank-norm / four-levers sweeps. Run on pdmle; aggregate metrics only.

Usage:
  PYTHONPATH=<repo> python3 within_night_test.py --exports <exports> --repo <repo> \
      --cvdir <bench> --levers-dir <dir-with-levers_test.py> --cohort large --ref n2
"""
from __future__ import annotations
import argparse, os, re, sys
import numpy as np

# family prefix -> the stage vocabulary that can appear immediately after it
STAGE_FAMILIES = [
    ("nk__hrv_", ["wake", "n1", "n2", "n3", "rem", "nrem", "sleep"]),
    ("nk__eegc_", ["wake", "n1", "n2", "n3", "rem"]),
    ("nk__rsp_", ["wake", "n1", "n2", "n3", "rem"]),
    ("rep__eeg_", ["wake", "n1", "n2", "n3", "rem"]),
    ("micro__couple_", ["n2", "n3", "nrem"]),
]
TINY = 1e-9


def parse_stage_groups(names):
    """Group column indices by (family_prefix, metric) -> {stage: col_idx}.

    Cross-stage *contrasts* already in the matrix (e.g. nk__hrv_rem_nrem_*_ratio, *_n3_wake_*)
    parse to a metric string that no other stage shares, so they land in singleton groups and are
    ignored downstream (need >=2 stages, or the reference stage, to emit anything)."""
    groups: dict[tuple[str, str], dict[str, int]] = {}
    for prefix, stages in STAGE_FAMILIES:
        # longer stage tokens first so 'sleep'/'nrem' win over 'n?'/'rem' at the anchor
        alt = "|".join(sorted(stages, key=len, reverse=True))
        rx = re.compile(rf"^{re.escape(prefix)}({alt})_(.+)$")
        for i, nm in enumerate(names):
            m = rx.match(nm)
            if m:
                groups.setdefault((prefix, m.group(2)), {})[m.group(1)] = i
    return groups


def build_within_night(X, names, mode, ref="n2", restrict=None):
    """Return (WN_matrix, wn_names) for the requested within-night transform.

    mode='ratio'   : f_stage / f_ref for every non-ref stage of each metric that has the ref stage.
    mode='zwithin' : (f_stage - nanmean_stages)/nanstd_stages across a metric's stages (>=2 finite).
    restrict       : optional regex; only metrics whose columns ALL match are transformed
                     (used for the abs-only, gain-carrier variant)."""
    groups = parse_stage_groups(names)
    cols, cnames = [], []
    for (prefix, metric), sd in groups.items():
        if restrict is not None and not all(restrict.search(names[idx]) for idx in sd.values()):
            continue
        if mode == "ratio":
            if ref not in sd:
                continue
            den = X[:, sd[ref]]
            bad = ~np.isfinite(den) | (np.abs(den) < TINY)
            for st, idx in sd.items():
                if st == ref:
                    continue
                r = X[:, idx] / np.where(bad, np.nan, den)
                r[bad] = np.nan
                cols.append(r)
                cnames.append(f"wn__{prefix}{st}_{metric}__over_{ref}")
        elif mode == "zwithin":
            if len(sd) < 2:
                continue
            stages = list(sd)
            M = np.column_stack([X[:, sd[st]] for st in stages])
            nfin = np.sum(np.isfinite(M), axis=1, keepdims=True)
            mu = np.nanmean(np.where(np.isfinite(M), M, np.nan), axis=1, keepdims=True)
            sdv = np.nanstd(np.where(np.isfinite(M), M, np.nan), axis=1, keepdims=True)
            Z = (M - mu) / np.where(sdv < TINY, np.nan, sdv)
            Z[np.repeat(nfin < 2, M.shape[1], axis=1)] = np.nan
            for k, st in enumerate(stages):
                cols.append(Z[:, k])
                cnames.append(f"wn__{prefix}{st}_{metric}__zwn")
    if not cols:
        return np.zeros((X.shape[0], 0)), []
    return np.column_stack(cols), cnames


def run_arm(tag, X, names, y, ages, sites, protect, lv, fp, ev):
    Xr = lv.site_rank(X, sites, protect)
    pooled, _, oof = lv.run_loso(Xr, y, ages, sites, fp, ev, names, model="hgb")
    dr = lv.decision_rules(oof, y, ages, ev)
    print(f"  {tag:<34} nfeat={X.shape[1]:>4}  AC-AUROC={pooled['age_auroc']:.4f} "
          f"AUROC={pooled['auroc']:.4f}  rwd pi={dr['pi']:+.4f} transfer={dr['transfer']:+.4f} "
          f"oracle={dr['oracle']:+.4f}")
    return pooled, dr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cvdir")
    ap.add_argument("--levers-dir", required=True, help="directory containing levers_test.py")
    ap.add_argument("--cohort", default="large")
    ap.add_argument("--ref", default="n2", help="reference stage for the ratio arm")
    ap.add_argument("--promote", action="store_true",
                    help="fuller per-site LOSO of baseline vs +wn_ratio(ref): folds + all 4 rules")
    args = ap.parse_args()

    for p in [args.repo, args.cvdir, args.levers_dir]:
        if p and p not in sys.path:
            sys.path.insert(0, p)
    import feature_prep as fp
    import evaluate_model as ev
    import levers_test as lv

    X, y, ages, sites, names, bmi_idx = lv.load_csv(args.exports, args.cohort)
    protect = [bmi_idx] if bmi_idx is not None else []
    print(f"cohort={args.cohort} n={len(y)} pos={int(y.sum())} feats={X.shape[1]} "
          f"sites={ {s: int((sites == s).sum()) for s in np.unique(sites)} }")

    WNr, nr = build_within_night(X, names, mode="ratio", ref=args.ref)

    if args.promote:
        print("\n" + "=" * 100)
        print(f"PROMOTE — fuller per-site LOSO: champion(rank) vs champion+wn_ratio({args.ref})(rank)")
        print("=" * 100)
        for tag, Xin, nm in [("champion (427)", X, names),
                             (f"champion + wn_ratio({args.ref}) ({X.shape[1] + WNr.shape[1]})",
                              np.hstack([X, WNr]), names + nr)]:
            Xr = lv.site_rank(Xin, sites, protect)
            pooled, folds, oof = lv.run_loso(Xr, y, ages, sites, fp, ev, nm, model="hgb")
            dr = lv.decision_rules(oof, y, ages, ev)
            print(f"\n  {tag}")
            print(f"    pooled: AC-AUROC={pooled['age_auroc']:.4f}  AUROC={pooled['auroc']:.4f}  "
                  f"AUPRC={pooled['auprc']:.4f}")
            print(f"    reward: pi={dr['pi']:+.4f}  q>pa={dr['q_gt_pa']:+.4f}  "
                  f"transfer={dr['transfer']:+.4f}  oracle={dr['oracle']:+.4f}")
            print(f"    {'held-out site':<16}{'n':>6}{'AC-AUROC':>10}{'AUROC':>9}{'reward@pi':>11}")
            for site, n, rwd, aca, au in folds:
                print(f"    {lv.SITE_NAMES.get(site, site) + ' (' + site + ')':<16}{n:>6}"
                      f"{aca:>10.4f}{au:>9.4f}{rwd:>+11.4f}")
        print("\nDONE_WITHIN_NIGHT")
        return

    WNw, nw = build_within_night(X, names, mode="ratio", ref="wake")
    WNz, nz = build_within_night(X, names, mode="zwithin")
    WNza, nza = build_within_night(X, names, mode="zwithin", restrict=lv.ABS_AMP_RE)
    print(f"within-night blocks: ratio-over-{args.ref}={WNr.shape[1]}  ratio-over-wake={WNw.shape[1]}  "
          f"zwithin(all)={WNz.shape[1]}  zwithin(abs-only)={WNza.shape[1]}")

    print("\n" + "=" * 100)
    print("WITHIN-NIGHT NORMALIZATION — HGB on per-site-rank feats (all arms ranked identically)")
    print("=" * 100)
    run_arm("rank baseline", X, names, y, ages, sites, protect, lv, fp, ev)
    run_arm(f"rank + wn_ratio({args.ref})",
            np.hstack([X, WNr]), names + nr, y, ages, sites, protect, lv, fp, ev)
    run_arm("rank + wn_ratio(wake)",
            np.hstack([X, WNw]), names + nw, y, ages, sites, protect, lv, fp, ev)
    run_arm("rank + wn_zwithin (all)",
            np.hstack([X, WNz]), names + nz, y, ages, sites, protect, lv, fp, ev)
    run_arm("rank + wn_zwithin (abs-only)",
            np.hstack([X, WNza]), names + nza, y, ages, sites, protect, lv, fp, ev)
    run_arm(f"rank + wn_ratio({args.ref}) + wn_zwithin",
            np.hstack([X, WNr, WNz]), names + nr + nz, y, ages, sites, protect, lv, fp, ev)

    print("\nDONE_WITHIN_NIGHT")


if __name__ == "__main__":
    main()
