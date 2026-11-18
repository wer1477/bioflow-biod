# Biod

**A trained decision layer for bioinformatics agents.**

Biod is the small, local decision model behind [Bioflow Agent](https://github.com/wer1477/bioflow-agent). It answers typed multiple-choice questions about a running bioinformatics workflow:

| Decision | Question |
|---|---|
| `tool_select` | Which capability should be loaded next? |
| `kb_select` | Which knowledge entries are relevant? |
| `context_retention` | Which context groups should be kept? |
| `role_select` | Which role should answer this message? |
| `relay_select` | May this role-to-role relay proceed? |
| S2 / R1 / W1 / V1 | Premise, plan-step, claim-evidence, and issue-triage audits |

The deployed model is `biod-mt-v2`: a frozen Qwen3.5-4B-Base text tower with LoRA adapters and seven task heads. It runs as an HTTP sidecar and returns a probability for every option in one forward pass.

## Why a decision layer

A bioinformatics agent makes many small choices per task: which tool, which reference build, which knowledge, which owner for a message, whether a relay is allowed. Asking the main LLM to make each one is slow, expensive, and hard to audit. Biod moves these bounded choices into a single small model, leaves open-ended reasoning to the main LLM, and logs every decision with its confidence and latency.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1) Base model (Qwen3.5-4B-Base) from Hugging Face
# 2) Trained heads/adapter from https://huggingface.co/wer1477/biod-mt-v2-4b
# 3) Serve
python -m biod.multitask.serve --ckpt /path/to/biod-mt-v2-4b --port 8790
```

Health check: `curl http://127.0.0.1:8790/v1/health`.

## Training and evaluation

- `biod/multitask/train.py` — multi-task discriminative SFT.
- `biod/multitask/evaluate.py` — frozen test-set metrics.
- `scripts/paper1/` — paired probes for evidence scope (E1) and candidate constraints (E2).
- `biod/multitask/paper1/` — probe renderers.

The training data and the exact manifests are described in the paper. The 4B weights are released on Hugging Face; the 9B variant is not part of the main release.

## Citation

If you use Biod, please cite the paper (in preparation) and this repository. See `CITATION.cff`.

## License

Apache-2.0. The base model is governed by the Qwen model license; see `NOTICE`.
