#!/usr/bin/env bash
# Post-extraction pipeline for advanced-features batch 1 + 2 (run on pdmle as
# arshia_ilaty_physio26, from /data-temp/physio-viewer). Assumes:
#   - nk_features_standard.NEW.csv  (batch-1 C1 re-extraction) is complete
#   - arch_features_standard.csv    (batch-1 F1/A1) already exists
#   - micro_features_standard.csv   is produced here (batch-2 B3/E2/A2)
#   - feature_matrix_local_plus.PREBATCH1.npz snapshot exists (the A/B "before")
set -euo pipefail

VIEW=/data-temp/physio-viewer
EXPORTS=$VIEW/exports
BENCH=$VIEW/bench
REPO=$BENCH/repo
CACHE=$BENCH/cache
DATA=/data-temp/shared-physionet26-dataset/extracted

cd "$VIEW"

echo "== 1. swap in the batch-1 NK re-extraction =="
if [ -f "$EXPORTS/nk_features_standard.NEW.csv" ]; then
  # keep the pre-batch-1 nk CSV as a backup, then promote NEW -> live
  cp -n "$EXPORTS/nk_features_standard.csv" "$EXPORTS/nk_features_standard.PREBATCH1.csv" || true
  mv "$EXPORTS/nk_features_standard.NEW.csv" "$EXPORTS/nk_features_standard.csv"
fi
echo "nk cols: $(head -1 $EXPORTS/nk_features_standard.csv | tr ',' '\n' | wc -l)"

echo "== 2. extract batch-2 micro features (EEG + chin-EMG + arousal) =="
python3 export_micro_features.py --dataset standard \
  --out "$EXPORTS/micro_features_standard.csv"

echo "== 3. rebuild the plus cache (now +arch +micro) =="
python3 "$BENCH/build_local_feature_cache.py" \
  --data-root "$DATA" --exports "$EXPORTS" --repo "$REPO" --out "$CACHE"

echo "== 4. A/B: pre-batch snapshot vs new cache (LOSO reward + AC-AUROC) =="
PYTHONPATH="$REPO" python3 "$BENCH/ab_batch1.py" \
  --old  "$CACHE/feature_matrix_local_plus.PREBATCH1.npz" \
  --new  "$CACHE/feature_matrix_local_plus.npz" \
  --repo "$REPO" --cvdir "$BENCH"

echo "== DONE =="
