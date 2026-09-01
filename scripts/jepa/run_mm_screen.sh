#!/bin/bash
# Multimodal PSG-JEPA stage-1 screen: pretrain-on-all -> embed -> shared-emb LOSO gate.
# 15-channel presence-aware I-JEPA. Same gate as EEG V1/V2 for a directly comparable
# verdict: does cross-modal (ECG/resp/SpO2/EOG/EMG) structure beat the EEG champion?
set -u
cd /data-temp/physio-viewer/exports/jepa
VENV=./gpuenv/bin/python
PP=/data-temp/physio-viewer/exports/jepa
CACHE=/data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz
REPO=/data-temp/physio-viewer/bench/repo
EPOCHS="${1:-12}"

if [ ! -f packed_mm_index.npz ]; then echo "NO packed_mm -- run extraction first"; exit 1; fi

echo "=== MM pretrain-all epochs=$EPOCHS $(date) ==="
PYTHONPATH=$PP $VENV raw_mm_jepa.py pretrain --prefix packed_mm --out encoder_mm_all.pt \
  --epochs "$EPOCHS" --batch 192 --device cuda > pretrain_mm_all.log 2>&1 || { echo PRETRAIN_FAIL; exit 1; }
echo "=== MM embed $(date) ==="
PYTHONPATH=$PP $VENV raw_mm_jepa.py embed --prefix packed_mm --ckpt encoder_mm_all.pt \
  --out emb_mm_all.npz --device cuda > embed_mm_all.log 2>&1 || { echo EMBED_FAIL; exit 1; }
echo "=== MM gate $(date) ==="
PYTHONPATH=$REPO python3 jepa_gate_ab.py --cache "$CACHE" --repo "$REPO" \
  --emb emb_mm_all.npz --filter --min-auc 0.6 --n-boot 2000 > mm_gate.log 2>&1 || { echo GATE_FAIL; exit 1; }
echo "MM_SCREEN_DONE $(date)" > mm_screen.done
tail -n 25 mm_gate.log
