"""Orchestrator for the PhysioNet 2026 dataset EDA suite.

Runs the demographic, biosignal (header-only), and sleep-architecture modules,
writes machine-readable JSON + a few flat CSVs, and renders a comprehensive
Markdown report. All outputs go to EDA_OUT_DIR (default ./eda, gitignored).

Usage (on the pdmle machine, where the data lives):
    sudo python3 run_eda.py                 # full cohort
    sudo python3 run_eda.py --limit 30      # quick smoke test (30 files/site)
    sudo python3 run_eda.py --skip-biosignals
"""
import os
import sys
import json
import time
import argparse

import common
from common import ensure_out, SITE_NAMES
import stats_demographics
import stats_biosignals
import stats_sleep
import stats_quality
import report

try:
    import stats_transitions
    import dump_transitions
except Exception:  # optional; only needed when computing dynamics
    stats_transitions = None
    dump_transitions = None


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="max EDF files per site (smoke test)")
    ap.add_argument("--skip-biosignals", action="store_true")
    ap.add_argument("--skip-sleep", action="store_true")
    ap.add_argument("--skip-transitions", action="store_true")
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--expert-sample", type=int, default=60,
                    help="per-site CAISR-vs-expert agreement sample size")
    args = ap.parse_args()

    out_dir = ensure_out()
    _log(f"DATA_ROOT = {common.DATA_ROOT}")
    _log(f"OUT_DIR   = {out_dir}")
    results = {"_meta": {"data_root": common.DATA_ROOT,
                         "limit_per_site": args.limit,
                         "sites": SITE_NAMES}}

    _log("== demographics ==")
    t = time.time()
    results["demographics"] = stats_demographics.run()
    _log(f"   done in {time.time()-t:.1f}s")

    if not args.skip_biosignals:
        _log("== biosignals (header-only) ==")
        t = time.time()
        results["biosignals"] = stats_biosignals.run(limit_per_site=args.limit, progress=_log)
        _log(f"   done in {time.time()-t:.1f}s")

    if not args.skip_sleep:
        _log("== sleep architecture (CAISR) ==")
        t = time.time()
        results["sleep"] = stats_sleep.run(limit_per_site=args.limit,
                                           expert_agreement_sample=args.expert_sample,
                                           progress=_log)
        _log(f"   done in {time.time()-t:.1f}s")

    if not args.skip_transitions and stats_transitions is not None:
        _log("== stage-transition dynamics & fragmentation ==")
        t = time.time()
        labels = dump_transitions._label_lookup() if dump_transitions else None
        results["transitions"] = stats_transitions.run(limit_per_site=args.limit,
                                                        label_lookup=labels,
                                                        progress=_log)
        _log(f"   done in {time.time()-t:.1f}s")

    if not args.skip_quality:
        _log("== data quality & coverage ==")
        t = time.time()
        results["quality"] = stats_quality.run(progress=_log)
        _log(f"   done in {time.time()-t:.1f}s")

    # Feature significance runs off the per-recording CSVs (written separately by
    # dump_per_recording.py / dump_transitions.py), so it's computed at report
    # time; see the README for the full-pipeline order.

    # ---- write JSON ----
    json_path = os.path.join(out_dir, "dataset_stats.json")
    with open(json_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    _log(f"wrote {json_path}")

    # ---- write flat CSVs for quick eyeballing ----
    report.write_csvs(results, out_dir)

    # ---- render Markdown report ----
    md = report.render(results)
    md_path = os.path.join(out_dir, "DATASET_REPORT.md")
    with open(md_path, "w") as fh:
        fh.write(md)
    _log(f"wrote {md_path}")

    print(md_path)


if __name__ == "__main__":
    main()
