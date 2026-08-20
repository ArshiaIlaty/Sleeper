#!/usr/bin/env python3
"""Subgroup + error analysis of the model's LOSO out-of-fold predictions.

Two deliverables, both from the SAME leakage-safe LOSO out-of-fold pass through
the production stack (site-MoE + Kaiser fine-tune + BMI imputer; per-fold
training-prevalence decision threshold — matches run_local_cv.loso):

  1. SUBGROUP PERFORMANCE  (--out-metrics <stem>.csv / printed table)
     For every subgroup value (overall, by sex, age-bin, site, BMI category,
     AHI severity) report n, prevalence, confusion counts, and
     recall / specificity / precision / AUROC / reward. Shows where the model
     is weak (e.g. a sex or an age band with low recall).

  2. CASE CARDS  (--out-cards <stem>.csv)
     One row per MISCLASSIFIED subject (FN and FP). Columns: pid, subgroup
     attributes, predicted prob, and the top-k features where the subject most
     deviates from the CI- (non-converter) baseline, as signed z-scores
     (z=(x-mean_neg)/std_neg). Lets you open a specific failed subject and see
     WHICH physiology drove the miss. Also prints the GROUP-LEVEL story: mean
     feature z of FN-vs-TP (why converters get missed) and FP-vs-TN.

Time_to_Event is joined for context (converters only) but is NOT a model input.
Runs on the pdmle box; the case-cards CSV holds per-subject rows (de-identified
BidsFolder IDs) -> keep it on the box, world-readable; only aggregate metrics +
figures leave. age is index 0 of X but is excluded from model training; it is
shown as a subgroup axis and flagged out of the z-score ranking by default.

  PYTHONPATH=/data-temp/physio-viewer/bench/repo python3 subgroup_analysis.py \
    --cache /data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz \
    --demo  /data-temp/shared-physionet26-dataset/extracted/demographics.csv \
    --out-metrics $HOME/subgroup_metrics --out-cards $HOME/error_case_cards
"""
from __future__ import annotations
import argparse, csv, warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)
import feature_prep as fp
import evaluate_model as ev
from sklearn.metrics import roc_auc_score

YEAR = 365.25


def safe(fn, *a):
    try:
        return float(fn(*a))
    except Exception:
        return float("nan")


def loso_oof(X, y, ages, sites, feat_names):
    """Per-recording OOF prob + per-fold training-prevalence binary (input order)."""
    prob = np.full(len(y), np.nan)
    binr = np.full(len(y), -1, dtype=int)
    for site in np.unique(sites):
        te = sites == site; tr = ~te
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        imp = fp.fit_bmi_imputer(X[tr], sites[tr], feat_names)
        Xtr = fp.apply_bmi_imputer(X[tr], sites[tr], imp)
        Xte = fp.apply_bmi_imputer(X[te], sites[te], imp)
        models = fp.fit_site_models(Xtr, y[tr], sites[tr])
        models = fp.fit_kaiser_finetuned(models, Xtr, y[tr], sites[tr])
        p = fp.predict_with_kaiser_override(models, Xte, sites[te], use_kaiser_finetuned=True)
        prob[te] = p
        binr[te] = (p > float(y[tr].mean())).astype(int)
    return prob, binr


# ---- subgroup definitions -------------------------------------------------
def age_bin(a):
    if not np.isfinite(a): return "age:unknown"
    if a < 50: return "age:<50"
    if a < 60: return "age:50-59"
    if a < 70: return "age:60-69"
    if a < 80: return "age:70-79"
    return "age:80+"


def bmi_cat(b):
    if not np.isfinite(b): return "bmi:unknown"
    if b < 18.5: return "bmi:under"
    if b < 25: return "bmi:normal"
    if b < 30: return "bmi:over"
    return "bmi:obese"


def ahi_sev(a):
    if not np.isfinite(a): return "ahi:unknown"
    if a < 5: return "ahi:none(<5)"
    if a < 15: return "ahi:mild(5-15)"
    if a < 30: return "ahi:moderate(15-30)"
    return "ahi:severe(30+)"


def build_subgroups(X, names, ages, sites, y):
    """Return list of (axis, value_array) subgroup axes."""
    SITE_NAMES = {"S0001": "BIDMC", "I0002": "Emory", "I0006": "Kaiser"}
    col = {n: i for i, n in enumerate(names)}
    sex = np.where(X[:, col["sex_f"]] == 1, "sex:F",
                   np.where(X[:, col["sex_m"]] == 1, "sex:M", "sex:other/unk"))
    agev = np.array([age_bin(a) for a in ages], dtype=object)
    sitev = np.array([f"site:{SITE_NAMES.get(str(s), str(s))}" for s in sites], dtype=object)
    bmiv = np.array([bmi_cat(b) for b in X[:, col["bmi"]]], dtype=object)
    ahiv = np.array([ahi_sev(a) for a in X[:, col["ahi"]]], dtype=object)
    return [("sex", sex), ("age", agev), ("site", sitev), ("bmi", bmiv), ("ahi", ahiv)], \
           {"sex": sex, "age": agev, "site": sitev, "bmi": bmiv, "ahi": ahiv}


def group_metrics(y, prob, binr, ages, mask, y_all, ages_all):
    """Confusion + rates + AUROC + reward on a subgroup slice."""
    yy, pp, bb, aa = y[mask], prob[mask], binr[mask], ages[mask]
    n = int(mask.sum())
    tp = int(np.sum((yy == 1) & (bb == 1))); fn = int(np.sum((yy == 1) & (bb == 0)))
    fp_ = int(np.sum((yy == 0) & (bb == 1))); tn = int(np.sum((yy == 0) & (bb == 0)))
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp_) if (tn + fp_) else float("nan")
    prec = tp / (tp + fp_) if (tp + fp_) else float("nan")
    auc = safe(roc_auc_score, yy, pp) if len(np.unique(yy)) == 2 else float("nan")
    a2p = ev.compute_prevalence(aa, y_all, ages_all, gap=2)
    reward = safe(ev.compute_reward, yy, bb, aa, a2p)
    return {"n": n, "n_pos": tp + fn, "prev": (tp + fn) / n if n else float("nan"),
            "TP": tp, "FN": fn, "FP": fp_, "TN": tn,
            "recall": recall, "specificity": spec, "precision": prec,
            "auroc": auc, "reward": reward}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--demo", required=True)
    ap.add_argument("--out-metrics", default=None)
    ap.add_argument("--out-cards", default=None)
    ap.add_argument("--topk", type=int, default=8, help="features per case card")
    ap.add_argument("--min-group", type=int, default=15, help="min n to report a subgroup")
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    X = d["X"].astype(np.float32); y = d["y"].astype(int)
    ages = d["ages"].astype(float); sites = np.asarray([str(s) for s in d["sites"]])
    pids = np.asarray([str(p) for p in d["pids"]])
    names = [str(n) for n in d["feature_names"]]

    prob, binr = loso_oof(X, y, ages, sites, names)
    valid = binr >= 0

    # join Time_to_Event (converters only) for context
    tte_by = {}
    with open(args.demo) as fh:
        for r in csv.DictReader(fh):
            v = (r.get("Time_to_Event") or "").strip()
            try:
                tte_by[r.get("BidsFolder", "")] = float(v) if v not in ("", "nan", "NaN") else np.nan
            except (ValueError, TypeError):
                tte_by[r.get("BidsFolder", "")] = np.nan
    tte = np.array([tte_by.get(p, np.nan) for p in pids])

    axes, axmap = build_subgroups(X, names, ages, sites, y)

    # ---------------- 1. subgroup performance ----------------
    rows = []
    overall = group_metrics(y, prob, binr, ages, valid, y, ages)
    overall.update({"axis": "overall", "group": "ALL"})
    rows.append(overall)
    for axis, vals in axes:
        for g in sorted(set(vals[valid])):
            mask = valid & (vals == g)
            if mask.sum() < args.min_group:
                continue
            m = group_metrics(y, prob, binr, ages, mask, y, ages)
            m.update({"axis": axis, "group": g})
            rows.append(m)

    hdr = ["axis", "group", "n", "n_pos", "prev", "TP", "FN", "FP", "TN",
           "recall", "specificity", "precision", "auroc", "reward"]
    print("=" * 104)
    print("SUBGROUP PERFORMANCE — LOSO out-of-fold, per-fold prevalence threshold")
    print("=" * 104)
    print(f"{'subgroup':<22}{'n':>5}{'pos':>4}{'prev':>6}{'recall':>8}{'spec':>7}"
          f"{'prec':>7}{'auroc':>7}{'reward':>8}")
    print("-" * 104)
    for r in rows:
        def f(k, p=2):
            v = r[k]
            return f"{v:.{p}f}" if isinstance(v, float) and np.isfinite(v) else "  -"
        lead = "  " if r["axis"] != "overall" else ""
        print(f"{lead}{r['group']:<22}"[:22].ljust(22) +
              f"{r['n']:>5}{r['n_pos']:>4}{f('prev'):>6}{f('recall'):>8}"
              f"{f('specificity'):>7}{f('precision'):>7}{f('auroc'):>7}{f('reward','3'):>8}")

    if args.out_metrics:
        p = args.out_metrics + ".csv"
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=hdr); w.writeheader()
            for r in rows:
                w.writerow({k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in hdr})
        print(f"\nwrote {p}")

    # ---------------- 2. case cards + group-level story ----------------
    # z-score every feature vs the CI- (non-converter) baseline; skip age (idx 0, not a model input)
    neg = valid & (y == 0)
    mu = np.array([np.nanmean(X[neg, j]) for j in range(X.shape[1])])
    sd = np.array([np.nanstd(X[neg, j]) for j in range(X.shape[1])])
    sd[sd == 0] = np.nan
    Z = (X - mu) / sd
    # exclude age (not a model input) and one-hot demographic dummies (z-scores of a
    # 0/1 column aren't meaningful "deviations") from the "why" ranking.
    CAT_PREFIX = ("sex_", "race_", "eth_")
    rankable = [j for j, n in enumerate(names)
                if n != "age" and not n.startswith(CAT_PREFIX)]

    def quad(i):
        if y[i] == 1: return "TP" if binr[i] == 1 else "FN"
        return "FP" if binr[i] == 1 else "TN"

    # group-level: mean signed z per feature, FN vs TP and FP vs TN
    def mean_z(mask):
        return np.array([np.nanmean(Z[mask, j]) for j in range(X.shape[1])])
    fn_mask = valid & (y == 1) & (binr == 0)
    tp_mask = valid & (y == 1) & (binr == 1)
    fp_mask = valid & (y == 0) & (binr == 1)
    zfn, ztp, zfp = mean_z(fn_mask), mean_z(tp_mask), mean_z(fp_mask)
    print("\n" + "=" * 104)
    print("WHY CONVERTERS ARE MISSED — features where FN (missed) differ most from TP (caught)")
    print("=" * 104)
    diff = zfn - ztp
    order = sorted(rankable, key=lambda j: -abs(diff[j]) if np.isfinite(diff[j]) else 0)
    for j in order[:15]:
        if not np.isfinite(diff[j]):
            continue
        arrow = "FN more like CI-" if abs(zfn[j]) < abs(ztp[j]) else "FN more extreme"
        print(f"  {names[j]:<34} z_FN={zfn[j]:+.2f}  z_TP={ztp[j]:+.2f}  Δ={diff[j]:+.2f}  ({arrow})")

    if args.out_cards:
        p = args.out_cards + ".csv"
        with open(p, "w", newline="") as fh:
            cols = (["pid", "outcome", "prob", "label", "age", "sex", "site", "bmi_cat",
                     "ahi_sev", "time_to_event_yr"] +
                    sum([[f"feat{k}", f"z{k}"] for k in range(1, args.topk + 1)], []))
            w = csv.writer(fh); w.writerow(cols)
            errs = np.where(valid & (binr != y))[0]
            for i in errs:
                zi = Z[i]
                cand = [j for j in rankable if np.isfinite(zi[j])]
                cand.sort(key=lambda j: -abs(zi[j]))
                top = cand[:args.topk]
                row = [pids[i], quad(i), round(float(prob[i]), 4), int(y[i]),
                       round(float(ages[i]), 1) if np.isfinite(ages[i]) else "",
                       axmap["sex"][i].split(":")[1], axmap["site"][i].split(":")[1],
                       axmap["bmi"][i].split(":")[1], axmap["ahi"][i].split(":")[1],
                       round(float(tte[i] / YEAR), 2) if np.isfinite(tte[i]) else ""]
                for j in top:
                    row += [names[j], round(float(zi[j]), 2)]
                row += [""] * (2 * (args.topk - len(top)))
                w.writerow(row)
        print(f"\nwrote {p}  ({int((valid & (binr != y)).sum())} misclassified subjects: "
              f"{int(fn_mask.sum())} FN + {int(fp_mask.sum())} FP)")


if __name__ == "__main__":
    main()
