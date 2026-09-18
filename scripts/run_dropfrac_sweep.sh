#!/usr/bin/env bash
# Sweep the invariance drop-fraction (variant D of site_agnostic_test) on the large
# cohort under LOSO, to find the point where dropping site-discriminative features
# helps generalization without discarding too much signal.
set -euo pipefail
BENCH=/data-temp/physio-viewer/bench
LOG=/data-temp/physio-viewer/exports/dropfrac_sweep.20260818.log
: > "$LOG"
for DF in 0.05 0.10 0.15 0.20 0.30; do
  echo "########## drop_frac=$DF ##########" >> "$LOG"
  PYTHONPATH="$BENCH/repo" python3 "$BENCH/site_agnostic_test.py" \
    --exports /data-temp/physio-viewer/exports \
    --repo "$BENCH/repo" --cvdir "$BENCH" \
    --cohort large --drop-frac "$DF" 2>&1 \
    | grep -E "=== D:|POOLED reward" >> "$LOG"
done
echo "ALLDONE_DROPFRAC" >> "$LOG"
