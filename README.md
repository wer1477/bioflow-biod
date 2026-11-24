# Biod

**A trained decision layer for bioinformatics agents.**

Biod is the small, local decision model behind
[Bioflow Agent](https://github.com/wer1477/bioflow-agent). It answers typed multiple-choice
questions about a running bioinformatics workflow, so that routine catalog and context decisions do
not have to be made implicitly inside the main LLM.

The released model, `biod-mt-v2`, is a frozen Qwen3.5-4B-Base text tower with LoRA adapters and five
routing heads. It runs as a local HTTP sidecar and returns a bounded label, or a score, in one forward
pass per request.

| Decision | Question |
|---|---|
| `tool_select` | Which capability should be loaded next (or `NO_TOOL`)? |
| `kb_select` | Which knowledge entries are relevant? |
| `context_retention` | Keep, summarize or drop each cold-zone context group? |
| `role_select` | Which role should answer this message? |
| `relay_select` | May this role-to-role relay proceed? |

The code also implements two audit heads (`premise_judge`, `overclaim_flag`) for the S2/R1/W1/V1
audit points. They are not part of the released checkpoint: with the five-head checkpoint those
points fail open to their deterministic fallbacks.

## Why a decision layer

A bioinformatics agent makes many small choices per task: which tool, which reference build, which
knowledge, which owner for a message, whether a relay is allowed. Asking the main LLM to make each
one is slow, expensive and hard to audit. Biod moves these bounded choices into a single small model,
leaves open-ended reasoning to the main LLM, and logs every decision with its confidence and latency.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Base tower: Qwen/Qwen3.5-4B-Base from Hugging Face (download it locally)
# 2) Released checkpoint
scripts/download-weights.sh /path/to/biod-mt-v2-4b

# 3) Serve
python -m biod.multitask.serve --ckpt /path/to/biod-mt-v2-4b --host 127.0.0.1 --port 8790
```

Health check: `curl http://127.0.0.1:8790/v1/health`.

`model_config.json` in the released checkpoint records the absolute path of the base tower used
during training. Before serving, either place the base model at that path or edit the `base_model`
entry to `Qwen/Qwen3.5-4B-Base` or to your local copy. Editing that string does not change the
weights.

The service reads its candidate catalog from `Datasets/v2/catalog_snapshot.json` relative to the
repository root. The snapshot is not tracked in this repository; see `Datasets/README.md`.

## Checkpoint provenance

The released checkpoint is the one used by the sidecar in the accompanying manuscript:

| Field | Value |
|---|---|
| Files | 5 (`heads.pt`, `adapter/adapter_model.safetensors`, `adapter/adapter_config.json`, `model_config.json`, `adapter/README.md`) |
| Size | 171,926,438 bytes |
| Tree hash (`sha256-tree-v1`) | `9ff57b3e6c09aa5f49bb08dedcc87286fea5f2039798ec83830b4ca217b3fca6` |
| `heads.pt` SHA-256 | `b0b9be9eb06c44ac57af230381795f60388661cd742540f4adfe379977d0b44f` |
| Base tower | `Qwen/Qwen3.5-4B-Base` |
| LoRA | rank 16, alpha 32, dropout 0.05, 12 target suffixes |
| Training split | 818 train / 42 validation rows from a teacher-exemplar pilot set (196 exemplars, 10 splits) |
| Model selection | best validation score over one epoch; no held-out test set |

Two caveats are part of the provenance: the decision thresholds in
`agent-client/backend/config/biod.json` were inherited from an earlier seven-head service rather than
recalibrated for this checkpoint, and the checkpoint was trained on a pilot-scale teacher-synthesized
set. The manuscript reports the measured behaviour under those conditions and does not claim
independent weight verification.

## Training and evaluation

- `biod/multitask/train.py` — multi-task discriminative SFT.
- `biod/multitask/evaluate.py` — frozen evaluation.
- `scripts/paper1/` — training and evaluation matrix runners for the paired probes.
- `biod/multitask/paper1/` — probe renderers for the evidence-scope and candidate-constraint probes.

The exact data manifests, splits and recipe are described in the paper. Training data is not tracked
in this repository.

## Limitations

- Trained on teacher-synthetic and curated examples; human audit is limited.
- The label space is frozen: new tools or knowledge entries outside the checkpoint vocabulary cannot
  be scored, and the runtime falls back to browsing.
- Thresholds are inherited (see above); refit them when the model, catalog or task distribution
  changes.
- The audit heads are not part of this release.

## Citation

If you use Biod, please cite the paper (in preparation) and this repository. See `CITATION.cff`.

## License

Apache-2.0 for the code and the released heads/adapters. The base model is governed by the Qwen
model license; see `NOTICE`.
