#!/bin/bash
# Server-side chain: wait for the scoped exporters to finish, splice Emory 200Hz rows into the
# native CSVs, then run the LOSO A/B. Runs entirely on the box (nohup) so it completes regardless
# of the local host's memory pressure. Writes /tmp/rs_ab.log and touches /tmp/rs_ab.DONE at the end.
set -u
LOG=/tmp/rs_ab.log
rm -f /tmp/rs_ab.DONE
{
  NAT=/data-temp/physio-viewer/exports
  RAW=/tmp/resample200/exports
  ARM=/tmp/resample200/arm200
  REPO=/data-temp/physio-viewer/bench/repo
  FAMS="features nk_features report_features arch_features micro_features"

  echo "[finish_ab] waiting for exporters..."
  while pgrep -f run_resampled.py >/dev/null 2>&1; do sleep 60; done
  echo "[finish_ab] exporters done at $(date -u +%H:%M:%S)"

  mkdir -p "$ARM"
  echo "[finish_ab] staging native CSVs -> $ARM"
  for f in $FAMS; do sudo cp "$NAT/${f}_large.csv" "$ARM/${f}_large.csv"; done
  sudo chmod a+rw "$ARM"/*.csv
  sudo chmod -R a+r "$RAW"

  echo "[finish_ab] splicing Emory 200Hz rows"
  python3 /tmp/rs200/splice_emory.py --exports "$ARM" --resampled-dir "$RAW" --suffix _200hz --cohort large || { echo "[finish_ab] SPLICE FAILED"; touch /tmp/rs_ab.DONE; exit 1; }

  echo "[finish_ab] running LOSO A/B"
  USP=$(python3 -c 'import site;print(site.getusersitepackages())')
  PYTHONPATH="$USP:$REPO:/tmp/rs200" python3 -u /tmp/rs200/resample_ab.py \
      --exports-dir "$ARM" --repo "$REPO" --levers-dir /tmp/rs200 --cohort large
  echo "[finish_ab] ALL_DONE at $(date -u +%H:%M:%S)"
} > "$LOG" 2>&1
touch /tmp/rs_ab.DONE
