#!/bin/bash
# Raw-EEG JEPA V2 stage-1 screen: pretrain-on-all -> embed -> shared-emb LOSO gate.
# V2 = per-channel (CxT) tokens + I-JEPA masking + CLS/register readout (borrowed
# from ECG-JEPA / EEG-VJEPA / PhysioJEPA / SignalJEPA). Same gate as V1 for a
# directly comparable verdict.
set -u
cd /data-temp/physio-viewer/exports/jepa
VENV=./gpuenv/bin/python
PP=/data-temp/physio-viewer/exports/jepa
CACHE=/data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz
REPO=/data-temp/physio-viewer/bench/repo
EPOCHS="${1:-15}"

echo "=== V2 pretrain-all epochs=$EPOCHS $(date) ==="
PYTHONPATH=$PP $VENV raw_eeg_jepa_v2.py pretrain --prefix packed --out encoder_raw_v2_all.pt \
  --epochs "$EPOCHS" --batch 256 --device cuda > pretrain_raw_v2_all.log 2>&1 || { echo PRETRAIN_FAIL; exit 1; }
echo "=== V2 embed $(date) ==="
PYTHONPATH=$PP $VENV raw_eeg_jepa_v2.py embed --prefix packed --ckpt encoder_raw_v2_all.pt \
  --out emb_raw_v2_all.npz --device cuda > embed_raw_v2_all.log 2>&1 || { echo EMBED_FAIL; exit 1; }
echo "=== V2 gate $(date) ==="
PYTHONPATH=$REPO python3 jepa_gate_ab.py --cache "$CACHE" --repo "$REPO" \
  --emb emb_raw_v2_all.npz --filter --min-auc 0.6 --n-boot 2000 > raw_v2_gate.log 2>&1 || { echo GATE_FAIL; exit 1; }
echo "V2_SCREEN_DONE $(date)" > raw_v2_screen.done
tail -n 25 raw_v2_gate.log
