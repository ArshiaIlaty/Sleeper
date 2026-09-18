#!/bin/bash
# Orchestrate the spectral-shape screen on the box (run as the ubuntu ssh login).
#   1. extract per-recording spectral-shape features, sharded across cores, as the
#      dataset account (arshia_ilaty_physio26; reads the DUA data + system numpy/scipy/edfio)
#   2. merge shards
#   3. run the LOSO A/B as ubuntu (pandas/sklearn live in ubuntu's ~/.local)
# Per-recording CSV stays on the box; only out/spectral_ab.json leaves. Polls /tmp/spec.DONE.
set -u
DIR=/tmp/spec
VIEW=/data-temp/physio-viewer
REPO=/data-temp/physio-viewer/bench/repo
EXPORTS=/data-temp/physio-viewer/exports
USP=/home/ubuntu/.local/lib/python3.10/site-packages
DS=${DS:-standard}
NSHARDS=${NSHARDS:-8}
DACC=arshia_ilaty_physio26

mkdir -p "$DIR/out"
rm -f /tmp/spec.DONE
{
  echo "[$(date +%T)] extract: $NSHARDS shards of $DS as $DACC"
  for s in $(seq 0 $((NSHARDS-1))); do
    sudo -n -u "$DACC" env PYTHONPATH="$VIEW:$DIR" python3 -u "$DIR/export_spectral_extra.py" \
      --dataset "$DS" --outdir "$DIR" --shard "$s" --nshards "$NSHARDS" --resume &
  done
  wait
  echo "[$(date +%T)] extract done; merging"
  sudo -n -u "$DACC" env PYTHONPATH="$VIEW:$DIR" python3 -u "$DIR/merge_spectral.py" \
    --dir "$DIR" --dataset "$DS" --out "$DIR/spectral_extra_${DS}.csv"
  sudo -n -u "$DACC" chmod 644 "$DIR/spectral_extra_${DS}.csv"
  echo "[$(date +%T)] A/B as ubuntu"
  PYTHONPATH="$USP:$REPO:$DIR" python3 -u "$DIR/spectral_ab.py" \
    --exports "$EXPORTS" --spectral "$DIR/spectral_extra_${DS}.csv" \
    --cohort "$DS" --out "$DIR/out/spectral_ab.json"
} > /tmp/spec.log 2>&1
touch /tmp/spec.DONE
echo "run_spectral.sh finished"
