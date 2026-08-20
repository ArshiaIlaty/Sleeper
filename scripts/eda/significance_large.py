#!/usr/bin/env python3
"""Run the univariate feature-significance suite on the LARGE cohort (497 positives).

The original stats_significance.py ran on the standard cohort (84 positives) and was
power-limited. This joins the four large-cohort feature families
(features_ + arch_ + nk_ + report_) into one per-recording table, normalizes the
label to "1"/"0", and calls stats_significance.run() UNCHANGED -- so the tests
(Welch t + Mann-Whitney, chi-square, Cohen's d / Cramer's V, BH-FDR q<0.05) are
byte-identical to the standard-cohort run. Only the cohort (and thus the power)
differs.

Prefixes match combine_features.py (arch__ / nk__ / rep__) so a "significant"
feature is traceable to its source. Leakage/ID/meta columns are excluded exactly as
in the standard run (time_to_event, ids, channel names, *_seconds, etc.).

Writes feature_significance_large.csv next to --out. Aggregate stats only leave the box.

Usage (on pdmle, as arshia_ilaty_physio26):
  PYTHONPATH=<repo> python3 significance_large.py \
      --exports /data-temp/physio-viewer/exports \
      --sig-dir /data-temp/physio-viewer/bench   # dir containing stats_significance.py
"""
from __future__ import annotations
import argparse, csv, os, sys

JOIN = ["bids_folder", "session"]
# id / label / leakage / bookkeeping never tested as a feature (mirrors the
# standard run's _EXCLUDE + the cache NON_FEATURE set)
DROP = {
    "dataset", "site_name", "label", "time_to_event", "time_to_last_visit",
    "n_epochs", "n_epochs_scored", "duration_hours",
    "ecg_channel", "eeg_channel", "rsp_channel", "spo2_channel", "eog_channel",
    "emg_channel", "has_resp_caisr", "report_seconds", "nk_seconds",
    "micro_seconds", "n_beats_total", "n_beats_clean", "arousal_fs", "arousal_index_src",
}
# families joined onto features_ (stem, prefix). Keys+demographics live in features_.
FAMILIES = [("arch_features", "arch__"), ("nk_features", "nk__"),
            ("report_features", "rep__")]


def read_keyed(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in JOIN:
            r[k] = (r.get(k) or "").strip()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", required=True)
    ap.add_argument("--cohort", default="large")
    ap.add_argument("--sig-dir", required=True, help="dir containing stats_significance.py")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.sig_dir not in sys.path:
        sys.path.insert(0, args.sig_dir)
    import stats_significance as ss

    base = read_keyed(os.path.join(args.exports, f"features_{args.cohort}.csv"))
    by_key = {(r["bids_folder"], r["session"]): dict(r) for r in base}
    # normalize label True/False -> 1/0 (the tests key on "1"/"0")
    for r in by_key.values():
        lab = str(r.get("label")).strip().lower()
        r["label"] = "1" if lab in ("1", "true") else ("0" if lab in ("0", "false") else "")

    for stem, prefix in FAMILIES:
        path = os.path.join(args.exports, f"{stem}_{args.cohort}.csv")
        if not os.path.exists(path):
            print(f"  ! skip {stem}: not found", file=sys.stderr)
            continue
        n_match = 0
        for r in read_keyed(path):
            k = (r["bids_folder"], r["session"])
            tgt = by_key.get(k)
            if tgt is None:
                continue
            n_match += 1
            for c, v in r.items():
                if c in JOIN or c in DROP:
                    continue
                tgt[f"{prefix}{c}"] = v
        print(f"  joined {stem}: {n_match}/{len(base)} matched", file=sys.stderr)

    rows = list(by_key.values())
    # drop excluded id/leakage cols from the base too (arch/nk/rep already filtered).
    # KEEP "label" -- it is in DROP so it is never TESTED as a feature, but it is the
    # target the tests key on, so it must remain in the table.
    for r in rows:
        for c in list(r.keys()):
            if c in DROP and c not in JOIN and c != "label":
                r.pop(c, None)

    # write a temp joined table then reuse ss.run (it reads a CSV path)
    out_csv = args.out or os.path.join(args.exports, f"feature_significance_{args.cohort}.csv")
    tmp = os.path.join(args.exports, f"_per_recording_{args.cohort}.tmp.csv")
    fields = sorted({c for r in rows for c in r.keys()})
    # keep JOIN + label first for readability
    ordered = JOIN + ["label"] + [c for c in fields if c not in JOIN + ["label"]]
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ordered, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    res = ss.run(tmp, dynamics_csv=None, out_csv=out_csv,
                 progress=lambda m: print(m, file=sys.stderr))
    os.remove(tmp)

    npos = sum(1 for r in rows if r.get("label") == "1")
    print(f"\ncohort={args.cohort}  n={len(rows)}  positives={npos}")
    print(f"features tested: {res['n_features']}  significant (q<0.05): {res['n_significant']}")
    # top 25 by |effect|
    sig = [r for r in res["results"] if r["significant"]]
    print(f"\nTop {min(25, len(sig))} significant by |effect|:")
    print(f"  {'feature':<34}{'effect':>8}  {'q_fdr':>9}  {'mean_ci':>9} {'mean_noci':>9}")
    for r in sig[:25]:
        eff = r["effect"] if r["effect"] is not None else float("nan")
        q = r["q_fdr"] if r["q_fdr"] is not None else float("nan")
        mc = r["mean_ci"] if r["mean_ci"] is not None else float("nan")
        mn = r["mean_noci"] if r["mean_noci"] is not None else float("nan")
        print(f"  {r['feature']:<34}{eff:>8.3f}  {q:>9.2e}  {mc:>9} {mn:>9}")
    print(f"\nwrote {out_csv}")
    print("DONE_SIGLARGE")


if __name__ == "__main__":
    main()
