#!/usr/bin/env bash
# Download the released biod-mt-v2 checkpoint (five routing heads) from Hugging Face.
#
# The five checkpoint files are fetched into one directory so that the directory
# fingerprint (serve.py, sha256-tree-v1) matches the value reported in the paper.
set -euo pipefail

DEST="${1:-$PWD/biod-mt-v2-4b}"
REPO="${BIOD_HF_REPO:-wer1477/biod-mt-v2-4b}"

mkdir -p "$DEST"
python - "$REPO" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo, dest = sys.argv[1], sys.argv[2]
snapshot_download(
    repo_id=repo,
    local_dir=dest,
    allow_patterns=["heads.pt", "model_config.json", "adapter/*"],
)
print("downloaded", repo, "to", dest)
PY

echo "checkpoint: $DEST"
echo "export BIOFLOW_BIOD_CKPT=$DEST"
