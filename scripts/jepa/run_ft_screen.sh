#!/bin/bash
# End-to-end supervised fine-tuning of the V2 EEG encoder (Challenge 2026 CI).
# Fine-tune (LOSO folds) -> eval vs champion; then fine-tune (LOPO folds) -> eval.
# The last JEPA lever: drop the frozen-embedding assumption.
set -u
cd /data-temp/physio-viewer/exports/jepa
VENV=./gpuenv/bin/python
PP=/data-temp/physio-viewer/exports/jepa
CACHE=/data-temp/physio-viewer/bench/cache/feature_matrix_local_plus.npz
REPO=/data-temp/physio-viewer/bench/repo
EPOCHS="${1:-10}"

if [ ! -f packed_index.npz ] || [ ! -f encoder_raw_v2_all.pt ]; then
  echo "MISSING packed_index.npz or encoder_raw_v2_all.pt"; exit 1; fi

for CV in site patient; do
  echo "=== FT fine-tune cv=$CV epochs=$EPOCHS $(date) ==="
  PYTHONPATH=$PP $VENV finetune_v2.py --prefix packed --ckpt encoder_raw_v2_all.pt \
    --out ftprob_${CV}.npz --cv $CV --epochs "$EPOCHS" --device cuda \
    > ft_train_${CV}.log 2>&1 || { echo "FT_TRAIN_FAIL($CV)"; exit 1; }
  tail -n 3 ft_train_${CV}.log
  echo "=== FT eval cv=$CV $(date) ==="
  PYTHONPATH=$REPO python3 ft_eval.py --cache "$CACHE" --repo "$REPO" \
    --ftprob ftprob_${CV}.npz --cv $CV --n-splits 5 --n-boot 2000 \
    > ft_eval_${CV}.log 2>&1 || { echo "FT_EVAL_FAIL($CV)"; exit 1; }
done

echo "FT_SCREEN_DONE $(date)" > ft_screen.done
echo "########## LOSO ##########"; tail -n 14 ft_eval_site.log
echo "########## LOPO ##########"; tail -n 14 ft_eval_patient.log
