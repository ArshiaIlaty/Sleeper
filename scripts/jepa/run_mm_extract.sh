#!/bin/bash
# Full multimodal (15-channel) PSG extraction -> pack into memmap.
# 5 resumable shards in parallel over the cohort, then pack to packed_mm_*.
# Mirrors the EEG raw extraction. Run as arshia_ilaty_physio26 (EDF read + owns dir).
set -u
cd /data-temp/physio-viewer/exports/jepa
PP=/data-temp/physio-viewer
OUT=raw_mm
NSHARDS=5

echo "=== MM extract $NSHARDS shards $(date) ==="
pids=()
for s in $(seq 0 $((NSHARDS - 1))); do
  PYTHONPATH=$PP python3 export_raw_multimodal.py --out "$OUT" --resume \
    --shard "$s" --nshards "$NSHARDS" > "mm_extract_$s.log" 2>&1 &
  pids+=($!)
  sleep 2
done
echo "shard pids: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do
  wait "$p" || fail=1
done
echo "=== all shards finished (fail=$fail) $(date) ==="
n_npz=$(ls "$OUT"/*.npz 2>/dev/null | wc -l)
echo "npz produced: $n_npz"
if [ "$n_npz" -lt 900 ]; then
  echo "MM_EXTRACT_TOO_FEW ($n_npz) -- not packing"; exit 1
fi

echo "=== pack $(date) ==="
PYTHONPATH=$PP python3 raw_mm_jepa.py pack --raw-dir "$OUT" --out-prefix packed_mm \
  > mm_pack.log 2>&1 || { echo PACK_FAIL; exit 1; }
echo "MM_EXTRACT_PACK_DONE $(date) npz=$n_npz" > mm_extract.done
tail -n 5 mm_pack.log
