#!/bin/bash
# Stage-1 driver: wait for the all-nights pretrain to finish, then embed + gate.
set -u
cd /data-temp/physio-viewer/exports/jepa
CSV=/data-temp/physio-viewer/exports/per_epoch/epoch_eeg_standard.csv
CACHE=/data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz
REPO=/data-temp/physio-viewer/bench/repo
PID="${1:-}"

echo "waiting for pretrain pid=$PID ..."
if [ -n "$PID" ]; then
  while kill -0 "$PID" 2>/dev/null; do sleep 20; done
fi
if [ ! -f encoder_all.pt ]; then
  echo "ERROR: encoder_all.pt not found after pretrain" > stage1_rest.err
  exit 1
fi
echo "pretrain done; embedding ..."
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 python3 ts_jepa.py embed \
  --csv "$CSV" --ckpt encoder_all.pt --out emb_all.npz > embed_all.log 2>&1
echo "gate ..."
PYTHONPATH="$REPO" CUDA_VISIBLE_DEVICES="" python3 jepa_gate_ab.py \
  --cache "$CACHE" --repo "$REPO" --emb emb_all.npz \
  --filter --min-auc 0.6 --n-boot 2000 > jepa_gate.log 2>&1
echo "STAGE1_DONE" > stage1_rest.done
echo "all done"
