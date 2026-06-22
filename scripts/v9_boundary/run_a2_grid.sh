#!/bin/bash
# A2 边界损失小网格（复用 A1 的共享嵌入缓存，每组只训投影）
cd ~/ludan/embeddingprofilter || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1
PY=~/anaconda3/envs/torch/bin/python
for lam in 1.0 2.0; do
  for mg in 0.3 0.5; do
    echo "=== A2 lam=$lam margin=$mg ==="
    $PY -u scripts/v9_boundary/train.py --loss boundary --lam "$lam" --margin "$mg" \
        --dims 32 --out "models/v9_a2_l${lam}_m${mg}" 2>&1 | grep -avE 'it/s'
  done
done
echo "===A2_GRID_DONE==="
