#!/usr/bin/env python3
"""Run trained model on S3 patients (reviewer-equivalent to run_model.py).

Reviewers:  python run_model.py -d holdout_data -m model -o outputs -v
This script: python scripts/run_model_s3.py -m model -o outputs -v
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, SCRIPTS, os.path.join(ROOT, "claude")):
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib.util

spec = importlib.util.spec_from_file_location("tc", os.path.join(ROOT, "claude", "team_code.py"))
tc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tc)

from extract_features_s3 import S3ChallengeDataset, extract_record_features  # noqa: E402
from helper_code import (  # noqa: E402
    DEMOGRAPHICS_FILE,
    HEADERS,
    find_patients,
    is_boolean,
    is_number,
    is_nan,
    update_demographics_table,
)


def run(
    model_folder: str,
    output_folder: str,
    *,
    limit: int | None = None,
    verbose: bool = True,
    allow_failures: bool = False,
) -> dict:
    model = tc.load_model(model_folder, verbose)
    ds = S3ChallengeDataset()
    data_folder = os.path.join(ROOT, "data", "training")
    os.makedirs(data_folder, exist_ok=True)
    demo_path = os.path.join(data_folder, DEMOGRAPHICS_FILE)
    if not os.path.exists(demo_path):
        with open(demo_path, "wb") as f:
            f.write(ds.demographics_bytes())

    records = find_patients(demo_path)
    if limit:
        records = records[:limit]
    num_records = len(records)
    os.makedirs(output_folder, exist_ok=True)

    results = {}
    infer_times = []
    t0 = time.perf_counter()

    for i, record in enumerate(tqdm(records, desc="run_model_s3", disable=not verbose)):
        pid = record[HEADERS["bids_folder"]]
        site = record[HEADERS["site_id"]]
        sess = record[HEADERS["session_id"]]
        if verbose:
            w = len(str(num_records))
            print(f"- {i+1:>{w}}/{num_records}: {pid} (site {site}, ses {sess})...")

        t_inf = time.perf_counter()
        try:
            use_alt = (
                model.get("kaiser_alt_models") is not None
                and str(site) == "I0006"
                and model.get("kaiser_alt_preset")
            )
            preset = model.get("kaiser_alt_preset") if use_alt else model.get("preset", tc.DEFAULT_PRESET)
            feats, _ = extract_record_features(ds, record, tc.DEFAULT_CSV_PATH, preset=preset)
            demo = ds.load_demographics_for_record(record)
            age = float(demo.get(HEADERS["age"], np.nan))
            binary, prob = tc.predict_from_features(model, feats, age, str(site))
            assert is_boolean(binary) or is_nan(binary)
            assert is_number(prob)
        except Exception:
            if allow_failures:
                binary, prob = float("nan"), float("nan")
            else:
                ds.close()
                raise
        infer_times.append(time.perf_counter() - t_inf)
        results[pid] = (binary, prob)

    ds.close()
    out_path = update_demographics_table(demo_path, output_folder, results)
    wall = time.perf_counter() - t0
    timing = {
        "n_records": num_records,
        "wall_clock_sec": round(wall, 2),
        "mean_infer_sec": round(float(np.mean(infer_times)), 4) if infer_times else 0.0,
        "total_infer_sec": round(float(np.sum(infer_times)), 2),
        "output_csv": out_path,
    }
    if verbose:
        print(f"Results: {out_path}")
        print(f"Timing: {timing}")
    return {"timing": timing, "output": out_path}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run model on S3 cohort.")
    ap.add_argument("-m", "--model-folder", default=os.path.join(ROOT, "model"))
    ap.add_argument("-o", "--output-folder", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-f", "--allow-failures", action="store_true")
    args = ap.parse_args()
    run(
        args.model_folder, args.output_folder,
        limit=args.limit, verbose=args.verbose, allow_failures=args.allow_failures,
    )


if __name__ == "__main__":
    main()
