#!/usr/bin/env python3
"""Extract per-patient features from S3 (extracted or zip) and cache matrix on S3.

Usage:
  # 1) Unzip to S3 (once, resumable):
  python scripts/unzip_s3_to_s3.py

  # 2) Extract features (streams EDFs, writes cache to S3):
  python scripts/extract_features_s3.py --limit 10   # smoke test
  python scripts/extract_features_s3.py              # full run
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib.util


def _load_claude_team_code():
    path = os.path.join(ROOT, "claude", "team_code.py")
    spec = importlib.util.spec_from_file_location("claude_team_code", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


from helper_code import (  # noqa: E402
    ALGORITHMIC_ANNOTATIONS_SUBFOLDER,
    HEADERS,
    PHYSIOLOGICAL_DATA_SUBFOLDER,
)

from s3_dataset import S3ChallengeDataset  # noqa: E402
from s3_io import (  # noqa: E402
    BUCKET,
    FEATURE_PREFIX,
    download_bytes,
    normalize_prefix,
    s3_key_exists,
    upload_bytes,
)

_claude = _load_claude_team_code()
extract_caisr_features = _claude.extract_caisr_features
extract_demographic_features = _claude.extract_demographic_features
extract_autonomic_features = _claude.extract_autonomic_features
EXTRACT_CACHE_PRESET = _claude.EXTRACT_CACHE_PRESET
PRESETS = _claude.PRESETS
DEFAULT_CSV_PATH = _claude.DEFAULT_CSV_PATH


def resolve_member(ds: S3ChallengeDataset, subfolder: str, site_id: str, pid: str, sess, suffix: str = "") -> str | None:
    base = f"{subfolder}/{site_id}"
    cands = [
        f"{base}/{pid}_ses-{sess}{suffix}.edf",
        f"{base}/{pid}_ses-{int(sess):02d}{suffix}.edf" if str(sess).isdigit() else None,
    ]
    for c in cands:
        if c and member_exists(ds, c):
            return c
    prefix = f"{base}/{pid}_ses-"
    # Search extracted S3 objects first, then zip central directory.
    try:
        from s3_io import list_keys, BUCKET

        s3_prefix = ds.extract_prefix + prefix
        s3_hits = sorted(
            k[len(ds.extract_prefix):]
            for k in list_keys(BUCKET, s3_prefix)
            if k.endswith(f"{suffix}.edf")
        )
        if s3_hits:
            return s3_hits[0]
    except Exception:
        pass
    hits = sorted(
        n for n in ds._zip_file().namelist()
        if n.startswith(prefix) and n.endswith(f"{suffix}.edf")
    )
    return hits[0] if hits else None


def member_exists(ds: S3ChallengeDataset, member: str | None) -> bool:
    if not member:
        return False
    if ds.has_extracted(member):
        return True
    return member in ds._zip_file().namelist()


def extract_record_features(ds: S3ChallengeDataset, record: dict, csv_path: str, preset=EXTRACT_CACHE_PRESET):
    """Extract features for one patient from S3 using a named preset."""
    pid = record[HEADERS["bids_folder"]]
    sid = record[HEADERS["site_id"]]
    sess = record[HEADERS["session_id"]]
    cfg = PRESETS[preset]
    blocks_feat, blocks_name = [], []

    if cfg.get("include_demo"):
        demo = ds.load_demographics_for_record(record)
        include_age = cfg["include_demo"] == "full"
        d_feat, d_names = extract_demographic_features(demo, include_age=include_age)
        blocks_feat.append(d_feat)
        blocks_name.extend(d_names)

    if cfg.get("include_autonomic"):
        phys_member = resolve_member(ds, PHYSIOLOGICAL_DATA_SUBFOLDER, sid, pid, sess)
        try:
            if member_exists(ds, phys_member):
                phys, phys_fs = ds.load_edf_member(phys_member)
                a_feat, a_names = extract_autonomic_features(phys, phys_fs, csv_path)
                del phys
            else:
                a_feat, a_names = extract_autonomic_features({}, {}, csv_path)
        except Exception as exc:
            tqdm.write(f"skip autonomic {pid}: {exc}")
            a_feat, a_names = extract_autonomic_features({}, {}, csv_path)
        blocks_feat.append(a_feat)
        blocks_name.extend(a_names)

    if cfg.get("include_caisr"):
        algo_member = resolve_member(
            ds, ALGORITHMIC_ANNOTATIONS_SUBFOLDER, sid, pid, sess, suffix="_caisr_annotations"
        )
        try:
            if member_exists(ds, algo_member):
                algo, _ = ds.load_edf_member(algo_member)
                c_feat, c_names = extract_caisr_features(algo)
                del algo
            else:
                c_feat, c_names = extract_caisr_features({})
        except Exception as exc:
            tqdm.write(f"skip caisr {pid}: {exc}")
            c_feat, c_names = extract_caisr_features({})
        blocks_feat.append(c_feat)
        blocks_name.extend(c_names)

    feats = np.hstack(blocks_feat).astype(np.float32)
    return feats, blocks_name


def cache_key(version: str) -> str:
    return normalize_prefix(FEATURE_PREFIX) + f"feature_matrix_{version}.npz"


def meta_key(version: str) -> str:
    return normalize_prefix(FEATURE_PREFIX) + f"feature_matrix_{version}.json"


def run(version: str = "v1", limit: int | None = None, force: bool = False) -> None:
    ck = cache_key(version)
    mk = meta_key(version)

    if not force and s3_key_exists(BUCKET, ck):
        print(f"Feature cache already exists: s3://{BUCKET}/{ck}")
        print("Use --force to recompute.")
        return

    csv_path = DEFAULT_CSV_PATH if os.path.exists(DEFAULT_CSV_PATH) else os.path.join(ROOT, "channel_table.csv")
    ds = S3ChallengeDataset()

    records = ds.patient_records()
    if limit:
        records = records[:limit]

    X_rows, y, ages, sites, pids, names = [], [], [], [], [], None
    for rec in tqdm(records, desc="Feature extraction", unit="pt"):
        pid = rec[HEADERS["bids_folder"]]
        label = ds.load_label(pid)
        if label not in (0, 1):
            continue
        try:
            feats, feat_names = extract_record_features(ds, rec, csv_path)
        except Exception as exc:
            tqdm.write(f"skip {pid}: {exc}")
            continue
        if names is None:
            names = feat_names
        demo = ds.load_demographics_for_record(rec)
        age = float(demo.get(HEADERS["age"], np.nan)) if demo else np.nan
        X_rows.append(feats)
        y.append(int(label))
        ages.append(age)
        sites.append(rec[HEADERS["site_id"]])
        pids.append(pid)

    ds.close()

    if not X_rows:
        raise RuntimeError("No features extracted.")

    X = np.vstack(X_rows).astype(np.float32)
    y = np.asarray(y, dtype=np.int8)
    ages = np.asarray(ages, dtype=np.float32)
    sites = np.asarray(sites)
    pids = np.asarray(pids)

    buf = io.BytesIO()
    np.savez_compressed(buf, X=X, y=y, ages=ages, sites=sites, pids=pids, feature_names=np.asarray(names))
    upload_bytes(BUCKET, ck, buf.getvalue(), "application/octet-stream")

    meta = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_patients": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "prevalence": float(y.mean()),
        "feature_names": names,
        "extract_preset": EXTRACT_CACHE_PRESET,
        "s3_npz": f"s3://{BUCKET}/{ck}",
    }
    upload_bytes(BUCKET, mk, json.dumps(meta, indent=2).encode("utf-8"), "application/json")

    print(f"\nSaved {X.shape[0]} x {X.shape[1]} feature matrix")
    print(f"  s3://{BUCKET}/{ck}")
    print(f"  s3://{BUCKET}/{mk}")
    print(f"  prevalence={y.mean():.3f}")


def load_feature_cache(version: str = "v1") -> dict:
    """Load feature cache from S3 into memory."""
    raw = download_bytes(BUCKET, cache_key(version))
    data = np.load(io.BytesIO(raw), allow_pickle=True)
    return {k: data[k] for k in data.files}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract features from S3 Challenge data.")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(version=args.version, limit=args.limit, force=args.force)


if __name__ == "__main__":
    main()
