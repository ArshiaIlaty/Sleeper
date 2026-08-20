#!/usr/bin/env python3
"""Combine every per-recording feature CSV into ONE wide table for sharing.

All feature exports share the join keys (dataset, bids_folder, session, site) at
one row per recording, so this is a straight left-join onto `features_` (the only
table carrying demographics + the label). Each other family's columns are
PREFIXED (arch__ / nk__ / micro__ / rep__) so the ~15 names that collide across
tables (e.g. `eeg_channel` lives in nk, micro AND report) stay unambiguous and
every column is traceable to its source CSV. Keys and `label` are kept once.

Nothing is dropped: provenance/meta columns (channel names, *_seconds timing,
arousal_fs, n_epochs_scored, has_resp_caisr) are preserved under their family
prefix. Only the duplicated key/label columns are removed from the joined tables.

Usage (on pdmle, as arshia_ilaty_physio26):
  python3 combine_features.py --exports /data-temp/physio-viewer/exports \
      --cohort standard --out /data-temp/physio-viewer/exports/all_features_standard.csv
  # add --with-dispersion to also fold in the 265-col within-stage spread table
"""
from __future__ import annotations
import argparse, os, sys
import pandas as pd

# columns that identify/label a recording — present in EVERY table, kept ONCE
# (from the base) and dropped from each joined table before prefixing.
DROP_ON_JOIN = ["dataset", "site", "site_name", "label"]
JOIN_KEYS = ["bids_folder", "session"]

# (filename-stem, prefix) for each family joined onto the base. Order = column order.
FAMILIES = [
    ("arch_features",  "arch__"),
    ("nk_features",    "nk__"),
    ("micro_features", "micro__"),
    ("report_features", "rep__"),
]
DISPERSION = ("dispersion_features", "disp__")


def read_keyed(path):
    """Read a feature CSV with the join keys forced to string (so 1 == '1' and a
    blank session lines up across tables), and assert one row per recording."""
    df = pd.read_csv(path, dtype={"bids_folder": str, "session": str}, low_memory=False)
    for k in JOIN_KEYS:
        if k not in df.columns:
            sys.exit(f"FATAL: {path} has no '{k}' column — cannot join")
        df[k] = df[k].fillna("").astype(str).str.strip()
    dup = df.duplicated(JOIN_KEYS).sum()
    if dup:
        sys.exit(f"FATAL: {path} has {dup} duplicate {JOIN_KEYS} rows — not 1/recording")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", required=True, help="dir with the *_<cohort>.csv exports")
    ap.add_argument("--cohort", default="standard")
    ap.add_argument("--out", required=True)
    ap.add_argument("--with-dispersion", action="store_true",
                    help="also join the non-avg within-stage dispersion table")
    args = ap.parse_args()

    base_path = os.path.join(args.exports, f"features_{args.cohort}.csv")
    if not os.path.exists(base_path):
        sys.exit(f"FATAL: base table {base_path} not found")
    base = read_keyed(base_path)
    n0 = len(base)
    print(f"base features_{args.cohort}: {n0} rows x {base.shape[1]} cols")

    fams = list(FAMILIES) + ([DISPERSION] if args.with_dispersion else [])
    combined = base.copy()
    prov = []  # (family, path, n_cols_added, n_matched)
    for stem, prefix in fams:
        path = os.path.join(args.exports, f"{stem}_{args.cohort}.csv")
        if not os.path.exists(path):
            print(f"  ! skip {stem}: {path} not found")
            continue
        df = read_keyed(path)
        # keep the join keys; drop the shared id/label cols; prefix everything else
        drop = [c for c in DROP_ON_JOIN if c in df.columns]
        df = df.drop(columns=drop)
        rename = {c: f"{prefix}{c}" for c in df.columns if c not in JOIN_KEYS}
        df = df.rename(columns=rename)
        before = combined.shape[1]
        combined = combined.merge(df, on=JOIN_KEYS, how="left", validate="one_to_one")
        added = combined.shape[1] - before
        # matched = rows where ANY prefixed col is non-null (i.e. this family joined)
        pref_cols = [c for c in combined.columns if c.startswith(prefix)]
        matched = combined[pref_cols].notna().any(axis=1).sum() if pref_cols else 0
        prov.append((stem, added, int(matched)))
        print(f"  + {stem:<20} prefix={prefix:<8} +{added:>3} cols  matched {matched}/{n0}")

    if len(combined) != n0:
        sys.exit(f"FATAL: row count changed {n0} -> {len(combined)} after joins")

    combined.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print(f"  {len(combined)} rows x {combined.shape[1]} cols")
    # quick provenance footer
    print("  column blocks:")
    print(f"    base (features_ + demographics + label) : "
          f"{base.shape[1]} cols")
    for stem, added, matched in prov:
        print(f"    {stem:<32}: {added} cols")


if __name__ == "__main__":
    main()
