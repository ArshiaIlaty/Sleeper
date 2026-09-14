"""Option-A splice: build drop-in *_large_200hz.csv by overwriting ONLY the re-extracted
records (matched on bids_folder+session) in the existing native-fs CSVs.

Usage:
  python3 splice_emory.py --exports /data-temp/physio-viewer/exports \
      --resampled-dir /tmp/resample200/exports --suffix _200hz

For each family f in {features,nk_features,report_features,arch_features,micro_features}:
  read exports/f_large.csv (native), read resampled-dir/f_large_200hz_raw.csv (re-extracted
  subset), and write exports/f_large_200hz.csv = native rows with any (bids_folder,session)
  present in the resampled file REPLACED by the resampled row. Verifies headers match.
Only families present in BOTH dirs are spliced; others are copied through unchanged so the
five 200hz CSVs are always a complete, self-consistent set for the LOSO A/B.
"""
import argparse, csv, os, shutil

FAMILIES = ["features", "nk_features", "report_features", "arch_features", "micro_features"]
KEY = ("bids_folder", "session")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exports", required=True)
    ap.add_argument("--resampled-dir", required=True)
    ap.add_argument("--suffix", default="_200hz")
    ap.add_argument("--cohort", default="large")
    args = ap.parse_args()
    for fam in FAMILIES:
        native = os.path.join(args.exports, f"{fam}_{args.cohort}.csv")
        rs = os.path.join(args.resampled_dir, f"{fam}_{args.cohort}{args.suffix}_raw.csv")
        out = os.path.join(args.exports, f"{fam}_{args.cohort}{args.suffix}.csv")
        if not os.path.exists(native):
            print(f"  SKIP {fam}: no native {native}"); continue
        if not os.path.exists(rs):
            shutil.copyfile(native, out)
            print(f"  COPY {fam}: no resampled subset -> passthrough ({out})"); continue
        with open(native, newline="") as fh:
            rd = csv.DictReader(fh); hdr = rd.fieldnames; nat = list(rd)
        with open(rs, newline="") as fh:
            rd = csv.DictReader(fh); rhdr = rd.fieldnames
            rep = {tuple((r.get(k, "") or "").strip() for k in KEY): r for r in rd}
        assert rhdr == hdr, f"{fam}: header mismatch native vs resampled"
        n_rep = 0
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=hdr); w.writeheader()
            for r in nat:
                k = tuple((r.get(c, "") or "").strip() for c in KEY)
                if k in rep:
                    w.writerow(rep[k]); n_rep += 1
                else:
                    w.writerow(r)
        print(f"  {fam}: {len(nat)} rows, {n_rep} replaced -> {out}")
    print("DONE_SPLICE")

if __name__ == "__main__":
    main()
