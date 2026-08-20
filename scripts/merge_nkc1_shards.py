#!/usr/bin/env python3
"""Merge the Phase-2 NK+C1 and reportv2 large-cohort shards into the final CSVs.

The re-extraction (export_combined_features.py, one EDF decode -> both NK+C1 and
report) wrote 3 shards each. This concatenates them into the canonical
`nk_features_large.csv` (194 cols, now WITH the C1 dfa/sampen columns) and
`report_features_large.csv` (132 cols), so they slot straight into
combine_features.py / the large cache builder.

Safety: asserts all 3 shard headers are byte-identical, dedups on
(bids_folder, session), backs up any existing target before overwrite, and prints
row/col counts + C1 column presence. Pure stdlib csv (no pandas needed).

Usage (on pdmle, as arshia_ilaty_physio26):
  python3 merge_nkc1_shards.py --exports /data-temp/physio-viewer/exports
"""
from __future__ import annotations
import argparse, csv, glob, os, sys

JOBS = [  # (shard-glob, final-out, expected-cols)
    ("shards/nkc1_large_s*.csv",     "nk_features_large.csv",     194),
    ("shards/reportv2_large_s*.csv", "report_features_large.csv", 132),
]
KEY = ("bids_folder", "session")


def merge(exports, pattern, out_name, expect_cols):
    shards = sorted(glob.glob(os.path.join(exports, pattern)))
    if len(shards) != 3:
        sys.exit(f"FATAL: expected 3 shards for {pattern}, found {len(shards)}: {shards}")
    header = None
    seen, rows = set(), []
    for sh in shards:
        with open(sh, newline="") as fh:
            rd = csv.reader(fh)
            hdr = next(rd)
            if header is None:
                header = hdr
                if len(header) != expect_cols:
                    sys.exit(f"FATAL: {sh} has {len(header)} cols, expected {expect_cols}")
                ki = [header.index(k) for k in KEY]
            elif hdr != header:
                sys.exit(f"FATAL: {sh} header differs from first shard")
            for row in rd:
                k = tuple(row[i] for i in ki)
                if k in seen:
                    continue
                seen.add(k)
                rows.append(row)
    out = os.path.join(exports, out_name)
    if os.path.exists(out):
        bak = out + ".prev"
        os.replace(out, bak)
        print(f"  backed up existing {out_name} -> {os.path.basename(bak)}")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    c1 = sum(1 for c in header if any(s in c for s in ("dfa_a1", "dfa_a2", "sampen")))
    print(f"  {out_name}: {len(shards)} shards -> {len(rows)} unique rows x {len(header)} cols"
          f"  (C1 dfa/sampen cols: {c1})")
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", required=True)
    args = ap.parse_args()
    counts = []
    for pattern, out_name, cols in JOBS:
        counts.append(merge(args.exports, pattern, out_name, cols))
    if len(set(counts)) != 1:
        print(f"  ! WARNING: row counts differ across families: {counts} "
              f"(join will left-outer on features_large)")
    print("DONE_MERGE")


if __name__ == "__main__":
    main()
