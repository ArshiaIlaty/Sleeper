#!/usr/bin/env python3
"""Extract small metadata CSVs from the PhysioNet 2026 S3 zip without full download."""

from __future__ import annotations

import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s3_io import BUCKET, ZIP_KEY, S3SeekableReader

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "metadata")

TARGET_SUFFIXES = (
    "demographics.csv",
    "ICD_codes_CI.csv",
    "ICD_Codes_CI.csv",
)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    reader = S3SeekableReader(BUCKET, ZIP_KEY)
    print(f"S3 object size: {reader._size / 1e9:.2f} GB")

    with zipfile.ZipFile(reader) as zf:
        targets = [n for n in zf.namelist() if n.lower().endswith(TARGET_SUFFIXES)]
        print(f"Found {len(targets)} metadata files:")
        for name in targets:
            print(f"  {name}")
            data = zf.read(name)
            out_path = os.path.join(OUT_DIR, name.replace("/", "__"))
            with open(out_path, "wb") as dst:
                dst.write(data)
            print(f"Wrote {out_path} ({len(data):,} bytes)")


if __name__ == "__main__":
    main()
