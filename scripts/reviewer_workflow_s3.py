#!/usr/bin/env python3
"""Reviewer workflow: train → run → evaluate (S3-backed).

Usage:
  python scripts/reviewer_workflow_s3.py --train-from-s3 -v
  python scripts/reviewer_workflow_s3.py --from-cache v3 -v   # fast (~30s train)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.environ.get("SLEEPER_PYTHON", "/home/ec2-user/miniconda3/envs/sleeper/bin/python")


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end reviewer-style workflow.")
    ap.add_argument("-m", "--model-folder", default=os.path.join(ROOT, "model"))
    ap.add_argument("-o", "--output-folder", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--from-cache", default=None, help="Skip EDF streaming (e.g. v3)")
    ap.add_argument("--train-from-s3", action="store_true", help="Stream all EDFs from S3")
    ap.add_argument("--run-limit", type=int, default=None, help="Limit inference patients")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-run", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    labels = os.path.join(ROOT, "data", "training", "demographics.csv")
    preds = os.path.join(args.output_folder, "demographics.csv")
    scores = os.path.join(ROOT, "eda", "reviewer_scores.txt")

    if not args.skip_train:
        cmd = [PYTHON, os.path.join(ROOT, "scripts", "train_full_s3.py"), "-m", args.model_folder]
        if args.from_cache:
            cmd += ["--from-cache", args.from_cache]
        elif not args.train_from_s3:
            cmd += ["--from-cache", "v3"]
        if args.verbose:
            cmd.append("-v")
        print("=== TRAIN ===")
        subprocess.check_call(cmd)

    if not args.skip_run:
        cmd = [
            PYTHON, os.path.join(ROOT, "scripts", "run_model_s3.py"),
            "-m", args.model_folder, "-o", args.output_folder,
        ]
        if args.run_limit:
            cmd += ["--limit", str(args.run_limit)]
        if args.verbose:
            cmd.append("-v")
        print("\n=== RUN MODEL ===")
        subprocess.check_call(cmd)

    if os.path.exists(labels) and os.path.exists(preds):
        os.makedirs(os.path.dirname(scores), exist_ok=True)
        cmd = [
            PYTHON, os.path.join(ROOT, "evaluate_model.py"),
            "-d", labels, "-o", preds,
            "-p", labels,
            "-s", scores,
        ]
        print("\n=== EVALUATE (official metrics) ===")
        subprocess.check_call(cmd)
        with open(scores) as f:
            print(f.read())
        print(f"Scores saved: {scores}")

    cmd = [PYTHON, os.path.join(ROOT, "scripts", "optimize_reward.py"), "--version", "v3"]
    if not args.verbose:
        cmd.append("--quiet")
    print("\n=== REWARD LOSO SWEEP ===")
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
