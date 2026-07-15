#!/usr/bin/env python3
"""Full training on ~1.1k patients from S3 (same logic as train_model.py).

Reviewers run:  python train_model.py -d training_data -m model -v
This script:    python scripts/train_full_s3.py -m model -v

Use --from-cache for a fast fit-only pass on the cached feature matrix.
Default streams EDFs from S3 (reviewer-equivalent feature extraction).
"""

from __future__ import annotations

import argparse
import os
import shutil
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

from extract_features_s3 import (  # noqa: E402
    S3ChallengeDataset,
    extract_record_features,
    load_feature_cache,
)
from feature_presets import select_indices  # noqa: E402
from helper_code import DEMOGRAPHICS_FILE, HEADERS  # noqa: E402


def prepare_training_folder(ds: S3ChallengeDataset, dest: str) -> str:
    """Copy demographics.csv so folder layout matches Challenge training_data/."""
    os.makedirs(dest, exist_ok=True)
    demo_src = os.path.join(ds.cache_dir, "demographics.csv")
    if not os.path.exists(demo_src):
        ds.patient_records()
    demo_dst = os.path.join(dest, DEMOGRAPHICS_FILE)
    shutil.copy2(demo_src, demo_dst)
    return demo_dst


def extract_all_from_s3(ds: S3ChallengeDataset, csv_path: str, verbose: bool):
    """Extract submit + kaiser-alt feature matrices from S3."""
    records = ds.patient_records()
    X, Xa, y, sites, ages, names, alt_names = [], [], [], [], [], None, None
    t0 = time.perf_counter()
    for rec in tqdm(records, desc="S3 feature extraction", disable=not verbose, unit="pt"):
        pid = rec[HEADERS["bids_folder"]]
        label = ds.load_label(pid)
        if label not in (0, 1):
            continue
        try:
            feats, names = extract_record_features(ds, rec, csv_path, preset=tc.DEFAULT_PRESET)
            fa, alt_names = extract_record_features(
                ds, rec, csv_path, preset=tc.KAISER_ALT_PRESET,
            )
        except Exception as exc:
            if verbose:
                tqdm.write(f"skip {pid}: {exc}")
            continue
        demo = ds.load_demographics_for_record(rec)
        try:
            age = float(demo.get(HEADERS["age"], np.nan))
        except Exception:
            age = float("nan")
        X.append(feats)
        Xa.append(fa)
        y.append(int(label))
        sites.append(rec[HEADERS["site_id"]])
        ages.append(age)

    elapsed = time.perf_counter() - t0
    return (
        np.vstack(X).astype(np.float32),
        np.vstack(Xa).astype(np.float32),
        np.asarray(y, dtype=int),
        np.asarray(sites),
        np.asarray(ages, dtype=np.float64),
        names,
        alt_names,
        elapsed,
    )


def load_from_cache(version: str, preset: str):
    cache = load_feature_cache(version)
    names = [str(n) for n in cache["feature_names"]]
    X = cache["X"].astype(np.float32)
    keep = select_indices(names, preset)
    names_out = [names[i] for i in keep]
    return X[:, keep], cache["y"].astype(int), np.asarray(cache["sites"]), cache["ages"].astype(float), names_out


def run(
    model_folder: str,
    *,
    from_cache: str | None = None,
    verbose: bool = True,
) -> dict:
    csv_path = os.path.join(ROOT, "channel_table.csv")
    data_folder = os.path.join(ROOT, "data", "training")
    ds = S3ChallengeDataset()
    prepare_training_folder(ds, data_folder)

    wall_t0 = time.perf_counter()
    if from_cache:
        if verbose:
            print(f"Loading features from S3 cache {from_cache} (fast path)...")
        X, y, sites, ages, names = load_from_cache(from_cache, tc.DEFAULT_PRESET)
        Xa, _, _, _, alt_names = load_from_cache(from_cache, tc.KAISER_ALT_PRESET)
        extract_sec = 0.0
    else:
        if verbose:
            print("Streaming EDFs from S3 (reviewer-equivalent extraction)...")
        X, Xa, y, sites, ages, names, alt_names, extract_sec = extract_all_from_s3(
            ds, csv_path, verbose,
        )
    ds.close()

    if verbose:
        print(f"Patients={len(y)}  prevalence={y.mean():.3f}  extract_sec={extract_sec:.1f}")

    payload = tc.fit_and_save_model(
        X, y, sites, ages, names, model_folder,
        Xa=Xa, alt_names=alt_names, verbose=verbose,
    )
    payload["timings"]["s3_extract_sec"] = round(extract_sec, 2)
    payload["timings"]["wall_clock_sec"] = round(time.perf_counter() - wall_t0, 2)
    payload["timings"]["data_folder"] = data_folder
    tc.save_model(model_folder, payload)

    if verbose:
        print(f"\nReviewer layout: demographics at {data_folder}/{DEMOGRAPHICS_FILE}")
        print(f"Model saved: {model_folder}/model.sav")
        print(f"Total wall clock: {payload['timings']['wall_clock_sec']:.1f}s")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Full S3 training (~1103 patients).")
    ap.add_argument("-m", "--model-folder", default=os.path.join(ROOT, "model"))
    ap.add_argument("--from-cache", default=None, help="e.g. v3 — skip EDF streaming")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    run(args.model_folder, from_cache=args.from_cache, verbose=args.verbose)


if __name__ == "__main__":
    main()
