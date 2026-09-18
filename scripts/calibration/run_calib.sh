#!/bin/bash
# Run the calibration/reliability probe on the box under nohup (survives local memory pressure).
# Writes /tmp/calib.log + /tmp/calib/calib_metrics.json, touches /tmp/calib.DONE at the end.
set -u
USP=/home/ubuntu/.local/lib/python3.10/site-packages
REPO=/data-temp/physio-viewer/bench/repo
DIR=/tmp/rs200
mkdir -p /tmp/calib
rm -f /tmp/calib.DONE
{
  PYTHONPATH="$USP:$REPO:$DIR" python3 -u "$DIR/calib_probe.py" \
    --exports /data-temp/physio-viewer/exports --repo "$REPO" \
    --levers-dir "$DIR" --cohort large --out /tmp/calib/calib_metrics.json
} > /tmp/calib.log 2>&1
touch /tmp/calib.DONE
