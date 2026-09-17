#!/usr/bin/env python3
"""Concatenate spectral_extra_<dataset>_s*.csv shards into one CSV (single header)."""
import argparse
import csv
import glob
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--dataset", default="standard")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    shards = sorted(glob.glob(os.path.join(a.dir, f"spectral_extra_{a.dataset}_s*.csv")))
    if not shards:
        raise SystemExit(f"no shards matching spectral_extra_{a.dataset}_s*.csv in {a.dir}")
    header, rows, seen = None, [], set()
    for sh in shards:
        with open(sh, newline="") as fh:
            r = csv.reader(fh)
            h = next(r)
            if header is None:
                header = h
            elif h != header:
                raise SystemExit(f"header mismatch in {sh}")
            bi, si = header.index("bids_folder"), header.index("session")
            for row in r:
                key = (row[bi], row[si])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"merged {len(shards)} shards -> {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
