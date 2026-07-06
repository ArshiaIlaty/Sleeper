#!/usr/bin/env python3
"""Stream-extract the Challenge zip in S3 into individual S3 objects (no local full unzip)."""

from __future__ import annotations

import argparse
import os
import sys
import time
import zipfile
from datetime import datetime, timezone

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from s3_io import (
    BUCKET,
    EXTRACT_PREFIX,
    MANIFEST_KEY,
    S3SeekableReader,
    ZIP_KEY,
    download_bytes,
    load_json,
    normalize_prefix,
    s3_key_exists,
    s3_object_size,
    save_json,
    upload_fileobj,
)


def dest_key(member_name: str) -> str:
    return normalize_prefix(EXTRACT_PREFIX) + member_name.lstrip("/")


def load_manifest() -> dict:
    try:
        return load_json(BUCKET, MANIFEST_KEY)
    except Exception:
        return {"completed": {}, "started_at": None, "updated_at": None}


def save_manifest(manifest: dict) -> None:
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    save_json(BUCKET, MANIFEST_KEY, manifest)


def already_uploaded(member_name: str, uncompressed_size: int, manifest: dict) -> bool:
    key = dest_key(member_name)
    if manifest.get("completed", {}).get(member_name) == uncompressed_size:
        remote = s3_object_size(BUCKET, key)
        if remote == uncompressed_size:
            return True
    return False


def upload_member(zf: zipfile.ZipFile, member_name: str, uncompressed_size: int) -> None:
    key = dest_key(member_name)
    with zf.open(member_name, "r") as src:
        upload_fileobj(BUCKET, key, src)


def run(skip_existing: bool = True, limit: int | None = None) -> None:
    prefix = normalize_prefix(EXTRACT_PREFIX)
    manifest = load_manifest()
    if manifest.get("started_at") is None:
        manifest["started_at"] = datetime.now(timezone.utc).isoformat()

    reader = S3SeekableReader(BUCKET, ZIP_KEY)
    print(f"Source : s3://{BUCKET}/{ZIP_KEY} ({reader._size / 1e9:.2f} GB)")
    print(f"Target : s3://{BUCKET}/{prefix}")

    with zipfile.ZipFile(reader) as zf:
        members = [n for n in zf.namelist() if not n.endswith("/")]
        if limit:
            members = members[:limit]
        print(f"Members: {len(members)}")

        uploaded = skipped = failed = 0
        t0 = time.time()
        for member in tqdm(members, desc="S3 unzip", unit="file"):
            info = zf.getinfo(member)
            if skip_existing and already_uploaded(member, info.file_size, manifest):
                skipped += 1
                continue
            try:
                upload_member(zf, member, info.file_size)
                manifest.setdefault("completed", {})[member] = info.file_size
                uploaded += 1
                if uploaded % 25 == 0:
                    save_manifest(manifest)
            except Exception as exc:
                failed += 1
                tqdm.write(f"FAILED {member}: {exc}")

        save_manifest(manifest)

    elapsed = time.time() - t0
    print(
        f"\nDone in {elapsed/60:.1f} min — uploaded={uploaded}, skipped={skipped}, failed={failed}"
    )
    print(f"Manifest: s3://{BUCKET}/{MANIFEST_KEY}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unzip PhysioNet 2026 data inside S3.")
    parser.add_argument("--no-skip", action="store_true", help="Re-upload even if present")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N files (debug)")
    args = parser.parse_args()
    run(skip_existing=not args.no_skip, limit=args.limit)


if __name__ == "__main__":
    main()
