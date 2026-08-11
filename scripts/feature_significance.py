#!/usr/bin/env python3
"""Univariate discrimination of every feature vs the Cognitive_Impairment label.

For each numeric feature we compute how well it *alone* separates CI+ from CI- on
the standard cohort — the "does this feature carry signal at all" screen that
precedes any modelling. Per feature:

  * AUROC              — P(random CI+ scores higher than random CI-); 0.5 = no signal.
  * effect = |AUROC-0.5|  — symmetric effect size (a feature lower in CI+ is as useful
                            as one higher in CI+; 0.63 and 0.37 are equally strong).
  * direction          — "CI+ higher" / "CI- higher".
  * Mann-Whitney U + z + two-sided p-value (tie-corrected normal approximation;
    AUROC and U are the same statistic, so both come from one ranking — no scipy).
  * q-value            — Benjamini-Hochberg FDR across ALL features tested (the honest
                         multiple-comparisons correction over the whole screen).

Features are grouped by their source so the team can see each family on its own:

  baseline  — CAISR sleep macro-architecture + autonomic (features_standard.csv)
  nk        — per-stage NeuroKit HRV / EEG-complexity / resp (nk_features_standard.csv)
  report    — EEG spectral/spindles, SpO2/hypoxic burden, resp events, REM density
              (report_features_standard.csv)
  demographic (reference) — age, bmi, kept aside so signal features can be read
                            against the obvious age baseline (not model inputs).

Writes TWO artifacts (both aggregate statistics, no patient data):
  <out>.csv  — every feature, one row, sorted by group then effect.
  <out>.md   — a presentable summary: cohort line, per-group top features, and the
               count significant at q<0.05.

Usage (on pdmle, as arshia_ilaty_physio26):
    python3 feature_significance.py --exports /data-temp/physio-viewer/exports \
        --out /data-temp/physio-viewer/exports/feature_significance_standard
"""
from __future__ import annotations

import argparse
import csv
import math
import os

import numpy as np

# Columns that are NEVER features (identifiers, target, leakage, provenance/timing,
# per-recording channel names, compute bookkeeping) — mirrors the modelling scripts.
NON_FEATURE = {
    "dataset", "bids_folder", "session", "site", "site_name", "label",
    "sex", "race", "ethnicity",              # categorical strings — not AUROC-able here
    "time_to_event", "time_to_last_visit",   # outcome-timing leakage
    "ecg_channel", "eeg_channel", "rsp_channel", "spo2_channel", "eog_channel",
    "has_resp_caisr", "report_seconds", "nk_seconds",
    "n_beats_total", "n_beats_clean",
}
# numeric demographics reported for reference only (never model inputs; the reward
# already discounts age) — assigned their own group so they don't crowd the signal.
DEMOGRAPHIC = {"age", "bmi"}


def _f(s):
    try:
        v = float(s)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def load_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def key(r):
    return (r.get("bids_folder", ""), r.get("session", ""))


def _phi(z):
    """Standard-normal CDF via erf (stdlib) — no scipy dependency."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mannwhitney(x, y):
    """AUROC + tie-corrected Mann-Whitney U two-sided p for feature x, binary y.

    Returns (auroc, u1, z, p, n_used, n_pos, n_neg) or None if a class is too small.
    AUROC = U1/(n1*n0); U and AUROC are one statistic, computed from a single ranking.
    """
    m = ~np.isnan(x)
    xx, yy = x[m], y[m]
    n = len(xx)
    n1 = int(yy.sum())          # CI+
    n0 = n - n1                 # CI-
    if n1 < 5 or n0 < 5:
        return None

    # average ranks (1..n), ties share the mean rank
    order = np.argsort(xx, kind="mergesort")
    sx = xx[order]
    ranks_sorted = np.arange(1, n + 1, dtype=float)
    tie_term = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks_sorted[i:j + 1] = avg
            t = j - i + 1
            tie_term += t ** 3 - t
        i = j + 1
    ranks = np.empty(n)
    ranks[order] = ranks_sorted

    r1 = ranks[yy == 1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2.0
    auroc = u1 / (n1 * n0)

    mu = n1 * n0 / 2.0
    var = (n1 * n0 / 12.0) * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
    if var <= 0:
        z = 0.0
    else:
        # continuity-corrected z
        z = (abs(u1 - mu) - 0.5) / math.sqrt(var)
        z = max(z, 0.0)
    p = 2.0 * (1.0 - _phi(z))
    p = min(1.0, max(p, 0.0))
    return auroc, u1, z, p, n, n1, n0


def bh_fdr(pvals):
    """Benjamini-Hochberg q-values for a list of p-values (returns same order)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    prev = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        val = pvals[idx] * m / rank
        prev = min(prev, val)
        q[idx] = min(1.0, prev)
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", default="/data-temp/physio-viewer/exports")
    ap.add_argument("--out", default=None,
                    help="output path stem (writes <stem>.csv and <stem>.md)")
    args = ap.parse_args()
    ex = args.exports
    out = args.out or os.path.join(ex, "feature_significance_standard")

    base = load_csv(os.path.join(ex, "features_standard.csv"))
    nk = {key(r): r for r in load_csv(os.path.join(ex, "nk_features_standard.csv"))}
    rep = {key(r): r for r in load_csv(os.path.join(ex, "report_features_standard.csv"))}
    # snapshot the ORIGINAL baseline header now: the join below mutates these same
    # row dicts (adds nk__/rep__ keys), so reading base[0].keys() afterwards would
    # fold NK/report columns into the baseline group and double-count them.
    base_header = list(base[0].keys())

    # keep recordings with a valid binary label
    rows, ys = [], []
    for r in base:
        lab = r.get("label", "")
        if lab not in ("True", "False", "0", "1"):
            continue
        rows.append(r)
        ys.append(1 if lab in ("True", "1") else 0)
    y = np.asarray(ys, int)

    # attach nk + report columns by (bids_folder, session)
    n_nk_missing = n_rep_missing = 0
    for r in rows:
        k = key(r)
        n = nk.get(k)
        if n:
            for c, v in n.items():
                if c not in NON_FEATURE:
                    r["nk__" + c] = v
        else:
            n_nk_missing += 1
        p = rep.get(k)
        if p:
            for c, v in p.items():
                if c not in NON_FEATURE:
                    r["rep__" + c] = v
        else:
            n_rep_missing += 1

    base_cols = [c for c in base_header if c not in NON_FEATURE and c not in DEMOGRAPHIC]
    demo_cols = [c for c in base_header if c in DEMOGRAPHIC]
    nk_cols = [c for c in nk[next(iter(nk))].keys() if c not in NON_FEATURE]
    rep_cols = [c for c in rep[next(iter(rep))].keys() if c not in NON_FEATURE]

    groups = [
        ("demographic (reference)", [(c, c) for c in demo_cols]),
        ("baseline",                [(c, c) for c in base_cols]),
        ("nk",                      [("nk__" + c, c) for c in nk_cols]),
        ("report",                  [("rep__" + c, c) for c in rep_cols]),
    ]

    # score every feature
    records = []   # (group, feature, auroc, effect, direction, n, n_pos, n_neg, u, z, p)
    for gname, cols in groups:
        for colkey, disp in cols:
            x = np.array([_f(r.get(colkey)) for r in rows])
            res = mannwhitney(x, y)
            if res is None:
                continue
            auroc, u1, z, p, n, n1, n0 = res
            records.append({
                "group": gname, "feature": disp,
                "auroc": auroc, "effect": abs(auroc - 0.5),
                "direction": "CI+ higher" if auroc >= 0.5 else "CI- higher",
                "n": n, "n_pos": n1, "n_neg": n0, "u_stat": u1, "z": z, "p_value": p,
            })

    # BH-FDR across ALL features tested
    qs = bh_fdr([r["p_value"] for r in records])
    for r, q in zip(records, qs):
        r["q_value"] = q
        r["significant_q05"] = q < 0.05

    # ---- write CSV (sorted by group, then effect desc) ----
    group_order = {g: i for i, (g, _) in enumerate(groups)}
    records.sort(key=lambda r: (group_order[r["group"]], -r["effect"]))
    fields = ["group", "feature", "auroc", "effect", "direction", "n", "n_pos",
              "n_neg", "u_stat", "z", "p_value", "q_value", "significant_q05"]
    with open(out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in records:
            row = dict(r)
            for k2 in ("auroc", "effect", "z"):
                row[k2] = f"{r[k2]:.4f}"
            row["u_stat"] = f"{r['u_stat']:.1f}"
            row["p_value"] = f"{r['p_value']:.3e}"
            row["q_value"] = f"{r['q_value']:.3e}"
            w.writerow(row)

    # ---- write markdown summary ----
    n_tested = len(records)
    lines = []
    lines.append("# Univariate feature significance — Cognitive_Impairment (standard cohort)\n")
    lines.append(f"- **Cohort:** {len(y)} recordings | CI+ = {int(y.sum())} | "
                 f"CI- = {int((1 - y).sum())} | prevalence = {y.mean():.3f}")
    lines.append(f"- **Features tested:** {n_tested} numeric features "
                 f"(demographic reference + baseline + NK + report)")
    lines.append(f"- **Test:** Mann-Whitney U (two-sided), AUROC as effect, "
                 f"Benjamini-Hochberg FDR across all {n_tested} features")
    lines.append(f"- **Merge coverage:** {n_nk_missing} recordings missing NK row, "
                 f"{n_rep_missing} missing report row (scored on available values)")
    lines.append("")
    lines.append("AUROC 0.5 = no signal. |AUROC-0.5| ≥ ~0.10 (i.e. AUROC ≥0.60 or ≤0.40) "
                 "is a notable single feature. `q` is FDR-corrected; `q<0.05` = survives "
                 "multiple comparisons.\n")

    # per-group summary counts
    lines.append("## Significant features per group (q < 0.05)\n")
    lines.append("| group | features tested | significant (q<0.05) | strongest AUROC |")
    lines.append("|---|--:|--:|---|")
    for gname, _ in groups:
        grp = [r for r in records if r["group"] == gname]
        if not grp:
            continue
        sig = sum(1 for r in grp if r["significant_q05"])
        best = max(grp, key=lambda r: r["effect"])
        lines.append(f"| {gname} | {len(grp)} | {sig} | "
                     f"{best['feature']} ({best['auroc']:.3f}) |")
    lines.append("")

    # per-group top tables
    for gname, _ in groups:
        grp = [r for r in records if r["group"] == gname]
        if not grp:
            continue
        top = grp[:15]
        lines.append(f"## {gname} — top {len(top)} by effect\n")
        lines.append("| feature | AUROC | direction | n | p | q | sig |")
        lines.append("|---|--:|---|--:|--:|--:|:-:|")
        for r in top:
            sig = "**✓**" if r["significant_q05"] else ""
            lines.append(f"| {r['feature']} | {r['auroc']:.3f} | {r['direction']} | "
                         f"{r['n']} | {r['p_value']:.1e} | {r['q_value']:.1e} | {sig} |")
        lines.append("")

    with open(out + ".md", "w") as fh:
        fh.write("\n".join(lines))

    # console recap
    print(f"cohort: {len(y)} recordings | CI+={int(y.sum())} CI-={int((1 - y).sum())} "
          f"prevalence={y.mean():.3f}")
    print(f"features tested: {n_tested} | nk missing={n_nk_missing} report missing={n_rep_missing}")
    for gname, _ in groups:
        grp = [r for r in records if r["group"] == gname]
        if not grp:
            continue
        sig = sum(1 for r in grp if r["significant_q05"])
        best = max(grp, key=lambda r: r["effect"])
        print(f"  {gname:<26} tested={len(grp):3d}  sig(q<.05)={sig:3d}  "
              f"best={best['feature']} AUROC={best['auroc']:.3f}")
    print(f"\nwrote {out}.csv and {out}.md")


if __name__ == "__main__":
    main()
