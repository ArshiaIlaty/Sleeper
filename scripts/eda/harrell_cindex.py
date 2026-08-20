#!/usr/bin/env python3
"""Harrell's C-index (survival concordance) of the model's LOSO risk scores.

The binary C-statistic == AUROC (same quantity), which we already report. This
adds the SURVIVAL C-index (Harrell 1982): concordance between the model's risk
score and the observed time-to-event WITH right-censoring — i.e. does the model
rank sooner-to-convert subjects as higher risk, using the censored negatives
correctly rather than discarding them?

  event time  = Time_to_Event      (the 84 converters; event=1)
  censor time = Time_to_Last_Visit (all subjects; negatives are censored, event=0)

A pair (i, j) is COMPARABLE if the subject with the shorter observed time had the
event (so the ordering of their true risk is known). It is CONCORDANT if the model
gives that shorter-time subject the higher risk score. Ties in score score 0.5.
C = concordant / comparable. C=0.5 is chance; higher is better.

Risk scores are the production LOSO out-of-fold probabilities (site-MoE + Kaiser
fine-tune + BMI imputer) — leakage-safe, and the timing columns are used ONLY
here for evaluation, never as model inputs. Aggregate stats only leave the box.

  PYTHONPATH=/data-temp/physio-viewer/bench/repo python3 harrell_cindex.py \
      --cache /data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz \
      --demo  /data-temp/shared-physionet26-dataset/extracted/demographics.csv
"""
from __future__ import annotations
import argparse, csv
import numpy as np
import feature_prep as fp
import evaluate_model as ev


def loso_oof(X, y, ages, sites, feat_names):
    prob = np.full(len(y), np.nan)
    for site in np.unique(sites):
        te = sites == site; tr = ~te
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        imp = fp.fit_bmi_imputer(X[tr], sites[tr], feat_names)
        Xtr = fp.apply_bmi_imputer(X[tr], sites[tr], imp)
        Xte = fp.apply_bmi_imputer(X[te], sites[te], imp)
        models = fp.fit_site_models(Xtr, y[tr], sites[tr])
        models = fp.fit_kaiser_finetuned(models, Xtr, y[tr], sites[tr])
        prob[te] = fp.predict_with_kaiser_override(models, Xte, sites[te], use_kaiser_finetuned=True)
    return prob


def harrell_c(risk, time, event):
    """Harrell's C-index with right-censoring. risk: higher = sooner event expected.

    A comparable pair pits a converter ('case', event=1) against anyone observed
    to survive longer (t[b] > t[case]); the case should score higher risk.
    Concordant when risk[case] > risk[b] (tie 0.5). Two events at the same time
    are comparable and score 0.5 only if their risks tie. Loops over the (few)
    event subjects and vectorizes the inner comparison, so bootstrap is cheap.
    """
    event = np.asarray(event); time = np.asarray(time); risk = np.asarray(risk)
    conc = 0.0
    comp = 0
    ev_idx = np.where(event == 1)[0]
    for a in ev_idx:
        later = time > time[a]                       # anyone (censored or event) surviving longer
        comp += int(later.sum())
        conc += float((risk[a] > risk[later]).sum()) + 0.5 * float((risk[a] == risk[later]).sum())
        # simultaneous events (each unordered pair once): count vs later-indexed events only
        same = (time == time[a]) & (event == 1) & (np.arange(len(time)) > a)
        comp += int(same.sum())
        conc += 0.5 * float((risk[a] == risk[same]).sum())   # only ties are (half-)concordant
    return (conc / comp if comp else float("nan")), comp


def boot_ci(risk, time, event, nboot=1000, seed=0):
    rng = np.random.RandomState(seed); n = len(risk); vals = []
    for _ in range(nboot):
        idx = rng.randint(0, n, n)
        c, comp = harrell_c(risk[idx], time[idx], event[idx])
        if comp and np.isfinite(c):
            vals.append(c)
    vals = np.array(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--demo", required=True)
    ap.add_argument("--nboot", type=int, default=1000)
    args = ap.parse_args()

    d = np.load(args.cache, allow_pickle=True)
    X = d["X"].astype(np.float32); y = d["y"].astype(int)
    ages = d["ages"].astype(float); sites = np.asarray([str(s) for s in d["sites"]])
    pids = np.asarray([str(p) for p in d["pids"]])
    names = [str(n) for n in d["feature_names"]]

    risk = loso_oof(X, y, ages, sites, names)

    tte, ttlv = {}, {}
    with open(args.demo) as fh:
        for r in csv.DictReader(fh):
            pid = r.get("BidsFolder", "")
            for col, tgt in (("Time_to_Event", tte), ("Time_to_Last_Visit", ttlv)):
                v = (r.get(col) or "").strip()
                try:
                    tgt[pid] = float(v) if v not in ("", "nan", "NaN") else np.nan
                except (ValueError, TypeError):
                    tgt[pid] = np.nan

    # observed time = event time for converters, else censoring (last-visit) time
    event = y.astype(int)
    time = np.array([
        (tte.get(p, np.nan) if event[i] == 1 else ttlv.get(p, np.nan))
        for i, p in enumerate(pids)
    ], dtype=float)

    ok = np.isfinite(risk) & np.isfinite(time)
    r, t, e = risk[ok], time[ok], event[ok]
    c, comp = harrell_c(r, t, e)
    lo, hi = boot_ci(r, t, e, nboot=args.nboot, seed=0)
    auroc = ev.compute_auroc(y[ok], risk[ok])

    print(f"n usable = {int(ok.sum())}/{len(y)}   events(converters) = {int(e.sum())}")
    print(f"comparable pairs = {comp}")
    print(f"Harrell C-index (survival concordance) = {c:.3f}  [95% CI {lo:.3f}, {hi:.3f}]")
    print(f"binary C-statistic (== AUROC, for reference) = {auroc:.3f}")


if __name__ == "__main__":
    main()
