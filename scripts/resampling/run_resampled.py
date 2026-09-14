"""Launcher: install the resample-to-200Hz patch, then run a production extractor UNCHANGED.

  python3 run_resampled.py <export_script.py> [--dataset large --out ..._large_200hz.csv ...]

The patch is installed BEFORE the extractor is imported/executed (via runpy), so the extractor
picks up the patched edfio.read_edf transparently -- no production module is edited. Optional
RESAMPLE_ONLY_BIDS (csv of BidsFolder) filters demographics to those records: TEST-ONLY validation
convenience implemented here in the launcher; leave it UNSET for the real full run.
"""
import os, sys, runpy
import resample_patch
resample_patch.install()

only_bids = set(x for x in os.environ.get("RESAMPLE_ONLY_BIDS", "").split(",") if x)
only_site = set(x for x in os.environ.get("RESAMPLE_ONLY_SITE", "").split(",") if x)
if only_bids or only_site:
    import sources
    _real_demo = sources.Dataset.demographics
    def _filt(self):
        out = _real_demo(self)
        if only_bids: out = [r for r in out if r.get("BidsFolder") in only_bids]
        if only_site: out = [r for r in out if r.get("SiteID") in only_site]
        return out
    sources.Dataset.demographics = _filt
    print(f"[run_resampled] filter bids={len(only_bids)} site={sorted(only_site)}", file=sys.stderr)

if len(sys.argv) < 2:
    sys.exit("usage: run_resampled.py <export_script.py> [args...]")
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
