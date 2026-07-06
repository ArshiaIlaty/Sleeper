# S3 Data Pipeline (no local disk required)

The Challenge zip (~130 GB) stays in S3. We never unzip it onto the EC2 disk.

## Layout

| S3 path | Contents |
|---------|----------|
| `physionetchallenge2026data/physionetchallenge2026data.zip` | Original archive |
| `physionetchallenge2026data/extracted/training_set_small/` | Unzipped files (one S3 object per zip member) |
| `physionetchallenge2026data/features/` | Cached feature matrices (`.npz` + `.json`) |

## Environment

Use the `sleeper` conda env:

```bash
export PYTHON=/home/ec2-user/miniconda3/envs/sleeper/bin/python
```

Optional overrides:

```bash
export PHYSIONET_S3_BUCKET=physionet2026
export PHYSIONET_ZIP_KEY=physionetchallenge2026data/physionetchallenge2026data.zip
export PHYSIONET_EXTRACT_PREFIX=physionetchallenge2026data/extracted/training_set_small/
```

## Step 1 — Unzip inside S3 (resumable)

```bash
# Foreground (shows progress)
$PYTHON scripts/unzip_s3_to_s3.py

# Background
bash scripts/start_unzip_background.sh
tail -f logs/unzip_s3_to_s3.log
```

- Streams each zip member from the archive via HTTP Range requests.
- Uploads directly to the `extracted/` prefix.
- Skips files already in the manifest (`_manifest.json`).
- Safe to stop and restart.

## Step 2 — Feature extraction (streams from S3 or zip)

Works **before** unzip finishes (reads from zip fallback) and **after** (reads extracted objects).

```bash
# Smoke test (1 patient)
$PYTHON scripts/extract_features_s3.py --limit 1 --force --version smoke

# Full training set → S3 feature cache
$PYTHON scripts/extract_features_s3.py --version v1 --force
```

Outputs:

- `s3://physionet2026/physionetchallenge2026data/features/feature_matrix_v1.npz`
- `s3://physionetchallenge2026data/features/feature_matrix_v1.json`

Load in Python:

```python
from scripts.extract_features_s3 import load_feature_cache
cache = load_feature_cache("v1")
X, y, ages, sites = cache["X"], cache["y"], cache["ages"], cache["sites"]
```

## Scripts

| Script | Purpose |
|--------|---------|
| `s3_io.py` | Shared S3/zip streaming utilities |
| `unzip_s3_to_s3.py` | Zip → extracted S3 objects |
| `s3_dataset.py` | Read demographics / EDF from S3 |
| `extract_features_s3.py` | Per-patient features → S3 cache |
| `extract_metadata_from_s3.py` | Pull small CSVs locally for EDA |

## Disk usage

- **No** full dataset on disk.
- Temporary EDF cache: `/tmp/sleeper_s3_cache/` (one file at a time during feature extraction; auto-reused by size).
