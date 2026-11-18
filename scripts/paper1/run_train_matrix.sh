#!/usr/bin/env bash
# Paper 1 Phase 2: sequential 8-run training matrix on one GPU.
# Order: 2B x2 -> 4B x3 -> 9B x3. Resumable: skips runs with manifest + best/heads.pt.
set -euo pipefail

cd /path/to/Biod
source .venv/bin/activate

export HF_HOME=/path/to/Biod/.hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

DATA_DIR=Datasets/paper1/train-v1
TASKS=tool_select,kb_select,context_retention,role_select,relay_select,premise_judge,overclaim_flag
MASTER=artifacts/runs/paper1-train-matrix.log
STATUS=artifacts/runs/paper1-train-matrix.status.jsonl
mkdir -p artifacts/runs

echo "[matrix] start $(date -Is)" | tee -a "$MASTER"

record() {
  local size="$1" seed="$2" run="$3" started="$4" finished="$5"
  .venv/bin/python - "$run" "$size" "$seed" "$started" "$finished" "$STATUS" <<'PY'
import json, os, sys
run, size, seed, started, finished, status = sys.argv[1:7]
manifest = os.path.join(run, "train_manifest.json")
best = None
steps = None
if os.path.isfile(manifest):
    with open(manifest, encoding="utf-8") as fh:
        m = json.load(fh)
    best = m.get("best_score")
    steps = m.get("global_step")
row = {
    "run": run, "size": size, "seed": int(seed),
    "started": started, "finished": finished,
    "best_score": best, "global_step": steps,
    "ok": bool(best is not None),
}
with open(status, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
print("[record]", json.dumps(row, ensure_ascii=False))
PY
}

run_one() {
  local size="$1" seed="$2"
  local run="artifacts/runs/paper1-${size}-s${seed}"
  if [[ -f "$run/train_manifest.json" && -f "$run/best/heads.pt" ]]; then
    echo "[skip] $(date -Is) $size seed=$seed (already complete)" | tee -a "$MASTER"
    return 0
  fi
  mkdir -p "$run"
  local started finished
  started=$(date -Is)
  echo "[start] $started $size seed=$seed -> $run" | tee -a "$MASTER"

  python -m biod.multitask.train \
    --data-dir "$DATA_DIR" \
    --out-dir "$run" \
    --base-model "Models/foundation-model/Qwen3.5-${size}-Base" \
    --tasks "$TASKS" \
    --epochs 1 \
    --lr-lora 5e-5 \
    --lr-head 2e-4 \
    --seed "$seed" \
    --head-hidden 1024 \
    --kb-tau 0.5 \
    --kb-top-k 3 \
    --device cuda \
    2>&1 | tee "$run/stdout.log"

  finished=$(date -Is)
  echo "[done] $finished $size seed=$seed" | tee -a "$MASTER"
  record "$size" "$seed" "$run" "$started" "$finished"
}

trap 'echo "[error] $(date -Is) line=$LINENO cmd=$BASH_COMMAND" | tee -a "$MASTER"' ERR

for seed in 20261102 20261103; do
  run_one 2B "$seed"
done

for seed in 20261102 20261103 20261104; do
  run_one 4B "$seed"
done

for seed in 20261102 20261103 20261104; do
  run_one 9B "$seed"
done

echo "[matrix] all done $(date -Is)" | tee -a "$MASTER"
