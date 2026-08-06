"""Dataset sources for the viewer + feature export.

Two cohorts are exposed through one interface:

  * **standard** — the 1,103-recording release on local disk
    (`/data-temp/shared-physionet26-dataset/extracted`).
  * **large**    — the 6,530-recording release in S3
    (`s3://els-thv-nlp-sbox-input-834843060358/physionet26/large-dataset`).

Everything the app needs — demographics rows, the small CAISR annotation EDF, and
the big physiological EDF — is fetched through a `Dataset` object, so the rest of
the code is source-agnostic. S3 access shells out to the `aws` CLI (available to
every account via the instance role) rather than boto3, which is only installed
for one user. Small files stream straight into memory (`aws s3 cp <key> -`); the
large physio EDFs are downloaded to a size-capped local cache on first use.

De-identified data only; nothing leaves the machine except through the SSH tunnel.
"""
import os
import io
import re
import csv
import glob
import threading
import subprocess
from collections import defaultdict

import edfio

_FILE_RE = re.compile(r"sub-([A-Za-z0-9]+)_ses-(\d+)")


def _run_aws(args, capture_binary=False):
    """Run an `aws` CLI command. Returns (rc, stdout, stderr). stdout is bytes if
    capture_binary else str."""
    p = subprocess.run(["aws"] + args, capture_output=True)
    out = p.stdout if capture_binary else p.stdout.decode("utf-8", "replace")
    err = p.stderr.decode("utf-8", "replace")
    return p.returncode, out, err


class Dataset:
    def __init__(self, key, label, description, kind, *, root=None,
                 bucket=None, prefix=None, cache_dir=None, cache_cap_gb=8.0):
        self.key = key                      # "standard" | "large"
        self.label = label                  # human name for the UI
        self.description = description
        self.kind = kind                    # "local" | "s3"
        self.root = root
        self.bucket = bucket
        self.prefix = (prefix or "").rstrip("/")
        self.cache_dir = cache_dir or os.path.join(
            os.path.expanduser("~/.cache/physio-viewer"), key)
        self.cache_cap_bytes = int(cache_cap_gb * (1024 ** 3))
        # per-instance memoisation (a shared lru_cache would evict across the two
        # datasets and re-download the large cohort's demographics/listings)
        self._demo_cache = None
        self._index_cache = {}
        self._cache_lock = threading.Lock()               # guards cache eviction
        self._dl_locks = defaultdict(threading.Lock)      # one lock per basename
        self._dl_locks_guard = threading.Lock()

    # ---- demographics ------------------------------------------------------
    def demographics(self):
        """List of demographics rows (dicts). Cached for the process lifetime."""
        if self._demo_cache is not None:
            return self._demo_cache
        if self.kind == "local":
            with open(os.path.join(self.root, "demographics.csv")) as fh:
                self._demo_cache = list(csv.DictReader(fh))
                return self._demo_cache
        rc, out, err = _run_aws(
            ["s3", "cp", f"s3://{self.bucket}/{self.prefix}/demographics.csv", "-"])
        if rc != 0:
            raise RuntimeError(f"aws s3 cp demographics failed: {err.strip()}")
        self._demo_cache = list(csv.DictReader(io.StringIO(out)))
        return self._demo_cache

    def record(self, bids):
        for r in self.demographics():
            if r.get("BidsFolder") == bids:
                return r
        return None

    # ---- file location -----------------------------------------------------
    def _s3_index(self, subdir):
        """Map basename-parsed (pid, sess) -> full s3 key for one subdir, from a
        single recursive listing. Also keyed by pid alone (first match) as a
        session-tolerant fallback. Cached per-instance per-subdir."""
        if subdir in self._index_cache:
            return self._index_cache[subdir]
        idx = {}
        rc, out, err = _run_aws(
            ["s3", "ls", "--recursive",
             f"s3://{self.bucket}/{self.prefix}/{subdir}/"])
        if rc != 0:
            raise RuntimeError(f"aws s3 ls {subdir} failed: {err.strip()}")
        for line in out.splitlines():
            # `aws s3 ls --recursive` columns: DATE TIME SIZE KEY. The key can in
            # principle contain spaces, so rejoin everything past the 3rd field.
            parts = line.split(maxsplit=3)
            if len(parts) < 4:
                continue
            key = parts[3]
            m = _FILE_RE.search(os.path.basename(key))
            if not m:
                continue
            pid, sess = m.group(1), m.group(2)
            idx[(pid, sess)] = key                 # literal digits from filename
            try:
                idx[(pid, f"n{int(sess)}")] = key  # zero-pad-normalised session
            except ValueError:
                pass
            idx.setdefault((pid, None), key)       # first-listed session fallback
        self._index_cache[subdir] = idx
        return idx

    def _locate(self, subdir, site, bids, sess, suffix):
        """Return a locator (local path or s3 key) for a record's EDF, or None."""
        pid = bids.replace("sub-", "")
        if self.kind == "local":
            folder = os.path.join(self.root, subdir, site)
            cands = [os.path.join(folder, f"sub-{pid}_ses-{sess}{suffix}.edf")]
            try:
                cands.append(os.path.join(folder, f"sub-{pid}_ses-{int(sess):02d}{suffix}.edf"))
            except (TypeError, ValueError):
                pass
            for c in cands:
                if os.path.exists(c):
                    return c
            hits = sorted(glob.glob(os.path.join(folder, f"sub-{pid}_ses-*{suffix}.edf")))
            return hits[0] if hits else None
        idx = self._s3_index(subdir)
        # try the literal session, then the zero-pad-normalised session, then the
        # first-listed session for this subject (single-session subjects only).
        norm = None
        try:
            norm = f"n{int(sess)}"
        except (TypeError, ValueError):
            pass
        return (idx.get((pid, str(sess)))
                or (idx.get((pid, norm)) if norm else None)
                or idx.get((pid, None)))

    # ---- CAISR annotations (small; read into memory) -----------------------
    def open_caisr(self, site, bids, sess):
        """Return an edfio EDF for the CAISR annotations (fully loaded), or None."""
        loc = self._locate("algorithmic_annotations", site, bids, sess,
                            "_caisr_annotations")
        if not loc:
            return None
        if self.kind == "local":
            return edfio.read_edf(loc, lazy_load_data=False)
        rc, data, err = _run_aws(["s3", "cp", f"s3://{self.bucket}/{loc}", "-"],
                                 capture_binary=True)
        if rc != 0 or not data:
            raise RuntimeError(f"aws s3 cp caisr failed: {err.strip()}")
        return edfio.read_edf(io.BytesIO(data), lazy_load_data=False)

    def has_caisr(self, site, bids, sess):
        return self._locate("algorithmic_annotations", site, bids, sess,
                            "_caisr_annotations") is not None

    # ---- physiological EDF (large; local path, downloading+caching for S3) --
    def physio_path(self, site, bids, sess):
        """Return a LOCAL path to the physio EDF, downloading from S3 into the
        size-capped cache if needed. None if the record has no physio file."""
        loc = self._locate("physiological_data", site, bids, sess, "")
        if not loc:
            return None
        if self.kind == "local":
            return loc
        # S3: cache under cache_dir/<basename>; download once.
        os.makedirs(self.cache_dir, exist_ok=True)
        base = os.path.basename(loc)
        dest = os.path.join(self.cache_dir, base)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            os.utime(dest, None)                    # touch -> LRU freshness
            return dest
        # Serialise concurrent downloads of the SAME file (ThreadingHTTPServer):
        # one thread downloads, the rest wait and then find it cached.
        with self._dl_locks_guard:
            lock = self._dl_locks[base]
        with lock:
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                os.utime(dest, None)
                return dest
            tmp = f"{dest}.{os.getpid()}.{threading.get_ident()}.part"
            rc, _, err = _run_aws(["s3", "cp", f"s3://{self.bucket}/{loc}", tmp])
            if rc != 0:
                if os.path.exists(tmp):
                    os.remove(tmp)
                raise RuntimeError(f"aws s3 cp physio failed: {err.strip()}")
            os.replace(tmp, dest)               # atomic publish
        self._evict_cache()
        return dest

    def _evict_cache(self):
        """Keep the cache under its size cap by deleting least-recently-used files.
        Serialised so concurrent downloads don't double-evict."""
        def _size(p):
            try:
                return os.path.getsize(p)
            except OSError:               # vanished mid-scan -> treat as gone
                return 0
        with self._cache_lock:
            try:
                files = [os.path.join(self.cache_dir, f) for f in os.listdir(self.cache_dir)]
                files = [f for f in files if os.path.isfile(f) and f.endswith(".edf")]
            except FileNotFoundError:
                return
            total = sum(_size(f) for f in files)
            if total <= self.cache_cap_bytes:
                return
            for f in sorted(files, key=lambda p: os.path.getmtime(p)):   # oldest first
                if total <= self.cache_cap_bytes:
                    break
                try:
                    sz = os.path.getsize(f)
                    os.remove(f)
                    total -= sz
                except OSError:
                    pass


# --------------------------------------------------------------------------- registry
def build_registry():
    """Construct the dataset registry from env-overridable defaults."""
    std_root = os.environ.get(
        "PHYSIONET_DATA_ROOT", "/data-temp/shared-physionet26-dataset/extracted")
    bucket = os.environ.get("PHYSIONET_S3_BUCKET",
                            "els-thv-nlp-sbox-input-834843060358")
    prefix = os.environ.get("PHYSIONET_S3_PREFIX", "physionet26/large-dataset")
    cache = os.environ.get("PHYSIO_CACHE_DIR")   # None -> per-dataset default
    reg = {}
    reg["standard"] = Dataset(
        "standard", "Standard (1,103)",
        "The 1,103-recording challenge release on local disk.",
        "local", root=std_root)
    reg["large"] = Dataset(
        "large", "Large (6,530)",
        "The 6,530-recording extended release, streamed from S3.",
        "s3", bucket=bucket, prefix=prefix,
        cache_dir=(os.path.join(cache, "large") if cache else None))
    return reg


REGISTRY = build_registry()
DEFAULT_DATASET = "standard"


def get_dataset(key):
    """Return the Dataset for `key`, falling back to the default."""
    return REGISTRY.get(key) or REGISTRY[DEFAULT_DATASET]
