#!/bin/bash
# Level-2 hierarchical night-JEPA screen (Challenge 2026 CI, pure exploration).
#   L1 dumpep (reuse V2 EEG teacher) -> L2 blocks -> temporal I-JEPA pretrain ->
#   embed (night rep + Exp-C stage-conditioned prediction-error) -> gate LOSO + LOPO.
# Attacks night DYNAMICS -- a different axis than the 8 per-epoch NO-SHIPs.
set -u
cd /data-temp/physio-viewer/exports/jepa
VENV=./gpuenv/bin/python
PP=/data-temp/physio-viewer/exports/jepa
CACHE=/data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz
REPO=/data-temp/physio-viewer/bench/repo
EPOCHS="${1:-60}"

if [ ! -f packed_index.npz ] || [ ! -f encoder_raw_v2_all.pt ]; then
  echo "MISSING packed_index.npz or encoder_raw_v2_all.pt"; exit 1; fi

# L1: per-epoch descriptors from the V2 EEG teacher (only if absent)
if [ ! -f l1_epemb.npy ]; then
  echo "=== L1 dumpep $(date) ==="
  PYTHONPATH=$PP $VENV night_jepa.py dumpep --prefix packed --ckpt encoder_raw_v2_all.pt \
    --out l1_epemb.npy --device cuda > l2_dumpep.log 2>&1 || { echo DUMPEP_FAIL; exit 1; }
fi

# L2: 5-min block sequences (only if absent)
if [ ! -f l2_blocks.npz ]; then
  echo "=== L2 blocks $(date) ==="
  PYTHONPATH=$PP $VENV night_jepa.py blocks --prefix packed --epemb l1_epemb.npy \
    --out l2_blocks.npz > l2_blocks.log 2>&1 || { echo BLOCKS_FAIL; exit 1; }
  tail -n 2 l2_blocks.log
fi

echo "=== L2 pretrain epochs=$EPOCHS $(date) ==="
PYTHONPATH=$PP $VENV night_jepa.py pretrain --blocks l2_blocks.npz --out encoder_l2_all.pt \
  --epochs "$EPOCHS" --batch 64 --device cuda > pretrain_l2_all.log 2>&1 || { echo PRETRAIN_FAIL; exit 1; }
tail -n 3 pretrain_l2_all.log

echo "=== L2 embed $(date) ==="
PYTHONPATH=$PP $VENV night_jepa.py embed --blocks l2_blocks.npz --ckpt encoder_l2_all.pt \
  --out emb_l2_all.npz --device cuda > embed_l2_all.log 2>&1 || { echo EMBED_FAIL; exit 1; }
tail -n 2 embed_l2_all.log

echo "=== L2 gate LOSO $(date) ==="
PYTHONPATH=$REPO python3 jepa_gate_ab.py --cache "$CACHE" --repo "$REPO" \
  --emb emb_l2_all.npz --filter --min-auc 0.6 --n-boot 2000 --cv site \
  > l2_gate_loso.log 2>&1 || { echo GATE_LOSO_FAIL; exit 1; }

echo "=== L2 gate LOPO (patient) $(date) ==="
PYTHONPATH=$REPO python3 jepa_gate_ab.py --cache "$CACHE" --repo "$REPO" \
  --emb emb_l2_all.npz --filter --min-auc 0.6 --n-boot 2000 --cv patient --n-splits 5 \
  > l2_gate_lopo.log 2>&1 || { echo GATE_LOPO_FAIL; exit 1; }

echo "L2_SCREEN_DONE $(date)" > l2_screen.done
echo "########## LOSO ##########"; tail -n 12 l2_gate_loso.log
echo "########## LOPO ##########"; tail -n 12 l2_gate_lopo.log
