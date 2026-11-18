"""Inference API for trained Biod SFT v0 checkpoints.

The eval harness imports :class:`BiodPredictor` and calls::

    predictor = BiodPredictor("artifacts/runs/<run_id>")
    tool_scores = predictor.predict_tool(state_text)   # [(tool_id, prob)] x62
    kb_scores = predictor.predict_kb(state_text)       # [(entry_id, prob)] x32

Contract details (agreed with eval-harness):

* ``predict_tool``: 62 slots (61 tools + ``NO_TOOL``), probabilities are the
  masked-softmax over the 62 pointer slots (sum == 1), returned in **canonical**
  option order with ``NO_TOOL`` last.
* ``predict_kb``: 32 entries, per-entry ``sigmoid`` scores (not normalised),
  returned in canonical order.
* ``order``: ``"canonical"`` (default) or ``"reversed"`` re-renders the option
  rows in that display order through the same pointer head; the returned list is
  always canonical-ordered, so the eval harness can compare argmax ids directly
  for the order-flip metric.
* One call = **one** forward pass over ``state + all options`` (latency
  contract). Over-length prompts raise instead of silently truncating.
"""

from __future__ import annotations

import json
import os
from typing import Sequence

import torch

from biod.catalog import Option
from biod.render import (
    NO_TOOL_ID,
    TOOL_QTYPE,
    PointerPrompt,
    render_kb_prompt,
    render_tool_prompt,
)

from .collate import encode_prompt, order_from_name
from .model import PointerHead, _torch_dtype, load_text_tower, pointer_logits

__all__ = ["BiodPredictor", "load_biod_config"]


def load_biod_config(ckpt_dir: str) -> dict:
    path = os.path.join(ckpt_dir, "biod_config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing biod_config.json in {ckpt_dir}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class BiodPredictor:
    """Load a checkpoint directory and score options with the pointer heads.

    Parameters
    ----------
    ckpt_dir:
        Run directory containing ``biod_config.json``, ``heads.pt`` and
        optionally ``adapter/``.
    device / dtype:
        Overrides; defaults come from the checkpoint config (cuda:0 + bf16).
    base_model:
        Override the base model path/id (default: the path recorded at training
        time, falling back to the hub id).
    merge_lora:
        Merge the LoRA adapter into the base weights with
        ``PeftModel.merge_and_unload()`` (exact same function, no adapter-layer
        overhead — restores single-forward latency). Default True.
    """

    def __init__(
        self,
        ckpt_dir: str,
        device: str | None = None,
        dtype: str | None = None,
        base_model: str | None = None,
        merge_lora: bool = True,
    ) -> None:
        self.ckpt_dir = ckpt_dir
        self.config = load_biod_config(ckpt_dir)
        cfg = self.config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or cfg.get("dtype", "bfloat16")
        self._torch_dtype = _torch_dtype(self.dtype)
        self.max_len = int(cfg.get("max_len") or 0)
        q_pos_cfg = cfg.get("q_position", "prompt")
        if isinstance(q_pos_cfg, dict):
            self.q_position_modes = {
                "tool": q_pos_cfg.get("tool", "prompt"),
                "kb": q_pos_cfg.get("kb", "prompt"),
            }
        else:  # legacy / single value
            self.q_position_modes = {"tool": q_pos_cfg, "kb": q_pos_cfg}
        self.q_position_mode = self.q_position_modes["tool"]
        self.hidden_size = int(cfg["hidden_size"])
        self.head_dim = int(cfg["head_dim"])

        base = base_model or cfg.get("base_model_path") or cfg.get("base_model")
        if base_model is None and cfg.get("base_model_path") and not os.path.exists(cfg["base_model_path"]):
            # recorded absolute remote path unavailable on this host
            base = cfg.get("base_model") or base
        if not base:
            raise ValueError("no base model path in checkpoint config")
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        text, resolved = load_text_tower(base, dtype=self.dtype)
        self.base_model_name = resolved
        adapter_dir = os.path.join(ckpt_dir, "adapter")
        lora_enabled = bool(cfg.get("lora", {}).get("enabled", False))
        self.lora_enabled = lora_enabled and os.path.isdir(adapter_dir)
        self.lora_merged = False
        if self.lora_enabled:
            from peft import PeftModel

            text = PeftModel.from_pretrained(text, adapter_dir)
            if merge_lora:
                try:
                    text = text.merge_and_unload()
                    self.lora_merged = True
                except Exception:  # pragma: no cover - fall back to adapter path
                    self.lora_merged = False
        text.to(self.device).eval()
        self.text = text

        self.tool_head = PointerHead(
            self.hidden_size,
            self.head_dim,
            float(cfg.get("head_dropout", 0.0)),
            float(cfg.get("head_norm_eps", 1e-5)),
        )
        self.kb_head = PointerHead(
            self.hidden_size,
            self.head_dim,
            float(cfg.get("head_dropout", 0.0)),
            float(cfg.get("head_norm_eps", 1e-5)),
        )
        heads_path = os.path.join(ckpt_dir, "heads.pt")
        payload = torch.load(heads_path, map_location="cpu", weights_only=True)
        self.tool_head.load_state_dict(payload["tool"])
        self.kb_head.load_state_dict(payload["kb"])
        self.tool_head.to(self.device).eval()
        self.kb_head.to(self.device).eval()

        self.tool_options = self._options_from_cfg(cfg.get("tool", {}), kind="tool")
        self.kb_options = self._options_from_cfg(cfg.get("kb", {}), kind="kb")
        self.tool_id_to_index = {o.id: i for i, o in enumerate(self.tool_options)}
        self.kb_id_to_index = {o.id: i for i, o in enumerate(self.kb_options)}
        self.no_tool_id = cfg.get("tool", {}).get("no_tool_id", NO_TOOL_ID)
        self.tool_canonical_ids = [o.id for o in self.tool_options] + [self.no_tool_id]
        self.kb_canonical_ids = [o.id for o in self.kb_options]

    @staticmethod
    def _options_from_cfg(task_cfg: dict, kind: str) -> list[Option]:
        options = task_cfg.get("options")
        if not options:
            raise ValueError(f"checkpoint config is missing {kind}.options")
        out: list[Option] = []
        for idx, entry in enumerate(options):
            out.append(
                Option(
                    index=idx,
                    name=entry["name"],
                    description=entry.get("description", ""),
                    id=entry["id"],
                    kind=kind,
                    layer=f"{kind}_l2",
                    domain=entry.get("domain", ""),
                )
            )
        return out

    # -- internals -----------------------------------------------------------
    def _enumerate_order(self, order: str, n: int) -> list[int]:
        resolved = order_from_name(order, n)
        if resolved is None:  # "random" is not a valid inference order
            raise ValueError("inference order must be 'canonical' or 'reversed'")
        return resolved

    def _forward_prompt(self, prompt: PointerPrompt) -> torch.Tensor:
        ids, option_positions, state_token = self._encode(prompt)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        positions = torch.tensor([option_positions], dtype=torch.long, device=self.device)
        task_key = "tool" if prompt.qtype == TOOL_QTYPE else "kb"
        mode = self.q_position_modes.get(task_key, "prompt")
        q_index = state_token if mode == "state" else len(ids) - 1
        q_positions = torch.tensor([[q_index]], dtype=torch.long, device=self.device)
        head = self.tool_head if prompt.qtype == TOOL_QTYPE else self.kb_head
        with torch.no_grad():
            logits = pointer_logits(
                self.text,
                head,
                input_ids,
                attention_mask,
                positions,
                q_positions=q_positions,
                autocast_dtype=self._torch_dtype,
                run_body_nograd=True,
            )
        return logits[0].float().cpu()

    def _encode(self, prompt: PointerPrompt) -> tuple[list[int], list[int], int]:
        encoded = encode_prompt(prompt, self.tokenizer, max_len=0)
        if encoded is None:  # pragma: no cover - max_len=0 disables dropping
            raise RuntimeError("unexpected over-length prompt")
        ids, positions, state_token = encoded
        if self.max_len and len(ids) > self.max_len:
            raise RuntimeError(
                f"prompt has {len(ids)} tokens > checkpoint max_len {self.max_len}; "
                "refusing to truncate (contract: over-length samples are dropped, never truncated)"
            )
        return ids, positions, state_token

    # -- public API ----------------------------------------------------------
    def predict_tool(self, state_text: str, order: str = "canonical") -> list[tuple[str, float]]:
        """Score all 61 tools + NO_TOOL for one state; returns 62 (id, prob)."""
        display_order = self._enumerate_order(order, len(self.tool_options))
        prompt = render_tool_prompt(state_text, self.tool_options, order=display_order)
        logits = self._forward_prompt(prompt)
        probs = torch.softmax(logits, dim=-1)
        by_id = {oid: float(p) for oid, p in zip(prompt.option_ids, probs.tolist())}
        return [(oid, by_id[oid]) for oid in self.tool_canonical_ids]

    def predict_kb(self, state_text: str, order: str = "canonical") -> list[tuple[str, float]]:
        """Score all 32 knowledge entries for one state; returns 32 (id, sigmoid)."""
        display_order = self._enumerate_order(order, len(self.kb_options))
        prompt = render_kb_prompt(state_text, self.kb_options, order=display_order)
        logits = self._forward_prompt(prompt)
        probs = torch.sigmoid(logits)
        by_id = {oid: float(p) for oid, p in zip(prompt.option_ids, probs.tolist())}
        return [(oid, by_id[oid]) for oid in self.kb_canonical_ids]

    def predict_options(
        self,
        state_text: str,
        options: Sequence[Option],
        order: str = "canonical",
    ) -> list[tuple[str, float]]:
        """Score an arbitrary option list with the kb pointer head (pointwise).

        Zero-training reuse of the knowledge-base head: the rendered rows are
        ordinary ``<options>`` lines, so the trained pointer head reads each
        row's last hidden state and returns one independent ``sigmoid`` per
        option. Used by ``biod.serve`` to score cold-zone message groups (an
        out-of-distribution input for this head: the score is a *ranking* signal
        and never implies a knowledge-base threshold was met).

        Returns ``[(option.id, sigmoid), ...]`` in the order of ``options``.
        ``order`` follows the same contract as :meth:`predict_kb`
        (``"canonical"``/``"reversed"``; scores are per-row so display order does
        not change them, but it is kept for render/training parity).
        """
        options = list(options)
        if not options:
            return []
        display_order = self._enumerate_order(order, len(options))
        prompt = render_kb_prompt(state_text, options, order=display_order)
        logits = self._forward_prompt(prompt)
        probs = torch.sigmoid(logits).tolist()
        # Map display slot -> caller index (display_order[slot] == options index).
        by_caller_index = {int(idx): float(probs[slot]) for slot, idx in enumerate(display_order)}
        return [(options[i].id, by_caller_index[i]) for i in range(len(options))]

    # convenience for the latency bench
    def predict_tool_batch(self, states: Sequence[str]) -> list[list[tuple[str, float]]]:
        return [self.predict_tool(s) for s in states]

    def predict_kb_batch(self, states: Sequence[str]) -> list[list[tuple[str, float]]]:
        return [self.predict_kb(s) for s in states]

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return (
            f"BiodPredictor({self.ckpt_dir!r}, device={self.device!r}, "
            f"dtype={self.dtype!r}, lora={self.lora_enabled}, "
            f"tool_options={len(self.tool_options)}, kb_options={len(self.kb_options)})"
        )
