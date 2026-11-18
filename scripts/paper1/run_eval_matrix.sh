#!/usr/bin/env bash
# Phase 3: evaluate all trained checkpoints on legacy + mixed 7-task test sets.
set -euo pipefail

cd /path/to/Biod
source .venv/bin/activate
export HF_HOME=/path/to/Biod/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TASKS=tool_select,kb_select,context_retention,role_select,relay_select,premise_judge,overclaim_flag
OUT_ROOT=artifacts/eval/paper1/models
MASTER=artifacts/eval/paper1/eval-matrix.log
mkdir -p "$OUT_ROOT" "$(dirname "$MASTER")"
echo "[eval-matrix] start $(date -Is)" | tee -a "$MASTER"

for run in artifacts/runs/paper1-*-s*; do
  [[ -f "$run/best/heads.pt" ]] || continue
  name=$(basename "$run")
  for data_dir in Datasets/paper1/legacy-test Datasets/paper1/train-v1; do
    if [[ "$data_dir" == *legacy-test ]]; then
      split=legacy
    else
      split=mixed
    fi
    out="$OUT_ROOT/$name/${split}_7task.json"
    if [[ -s "$out" ]]; then
      echo "[skip] $(date -Is) $name $split" | tee -a "$MASTER"
      continue
    fi
    mkdir -p "$(dirname "$out")"
    echo "[start] $(date -Is) $name $split" | tee -a "$MASTER"
    python scripts/eval_multitask.py \
      --checkpoint "$run/best" \
      --data-dir "$data_dir" \
      --split test \
      --tasks "$TASKS" \
      --out "$out" \
      2>&1 | tee "$out.log"
    echo "[done] $(date -Is) $name $split" | tee -a "$MASTER"
  done
done

echo "[eval-matrix] all done $(date -Is)" | tee -a "$MASTER"
