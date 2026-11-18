"""Frozen Qwen3.5-4B text tower + LoRA + two fp32 pointer heads.

Architecture facts verified against ``transformers 5.20.2`` on deployment-host
(2026-10-22):

* ``Qwen/Qwen3.5-4B-Base`` is a ``Qwen3_5ForConditionalGeneration`` wrapper;
  ``AutoModel.from_pretrained`` returns ``Qwen3_5Model`` with children
  ``visual`` (333.5M) + ``language_model`` (4.206B, ``Qwen3_5TextModel``).
  We keep only ``language_model`` and run text-only forwards on it.
* ``hidden_size=2560``, 32 layers, ``layer_types`` = 24 ``linear_attention``
  (every layer not divisible by 4) + 8 ``full_attention``.
* LoRA target suffix -> module count on the text tower:
  q/k/v/o_proj = 8 each; gate/up/down_proj = 32 each;
  in_proj_qkv/z/a/b = 24 each; out_proj = 24 (``layers.N.linear_attn.out_proj``).

peft silently ignores target names that match nothing, so
:meth:`BiodModel._apply_lora` asserts every one of the 12 contract targets
matched the same number of modules as the base model, and raises otherwise.
"""

from __future__ import annotations

import gc
import os
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TrainConfig, resolve_base_model, setup_offline_env
from .lora import apply_lora

__all__ = [
    "PointerHead",
    "BiodModel",
    "load_text_tower",
    "DEFAULT_BASE_MODEL_HINT",
]

DEFAULT_BASE_MODEL_HINT = "Qwen/Qwen3.5-4B-Base"


def _torch_dtype(name: str) -> torch.dtype:
    table = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in table:
        raise ValueError(f"unsupported dtype {name!r}")
    return table[name]


def load_text_tower(base_model: str, dtype: str = "bfloat16", device: str | None = None):
    """Load the frozen text tower of the multimodal wrapper.

    Returns ``(text_model, resolved_name)``. The vision tower is dropped so it
    never occupies GPU memory.
    """
    setup_offline_env()
    from transformers import AutoModel

    base_model = resolve_base_model(base_model)
    torch_dtype = _torch_dtype(dtype)
    wrapper = AutoModel.from_pretrained(
        base_model,
        dtype=torch_dtype,
        local_files_only=True,
    )
    resolved = getattr(wrapper.config, "_name_or_path", None) or base_model
    text = getattr(wrapper, "language_model", None)
    if text is None:
        inner = getattr(wrapper, "model", None)
        text = getattr(inner, "language_model", None)
    if text is None:
        # A pure text model (e.g. Qwen3_5ForCausalLM) is passed through as-is.
        text = wrapper
    # Drop the vision tower (contract: text only, never forwarded).
    if getattr(wrapper, "visual", None) is not None:
        wrapper.visual = None
    for param in text.parameters():
        param.requires_grad_(False)
    text.eval()
    if device:
        text.to(device)
    del wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return text, resolved


def pointer_logits(
    text_module: nn.Module,
    head: PointerHead,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    option_positions: torch.Tensor,
    q_positions: torch.Tensor | None = None,
    autocast_dtype: torch.dtype | None = None,
    run_body_nograd: bool = False,
) -> torch.Tensor:
    """Shared forward used by both training and inference.

    Runs the frozen text tower once, gathers the last token (query) and every
    option line's last token, then applies the fp32 pointer head. Keeping this
    in one place guarantees the training and eval readouts are identical.
    """
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)
        if input_ids.is_cuda and autocast_dtype is not None and autocast_dtype != torch.float32
        else nullcontext()
    )
    with autocast_ctx:
        if run_body_nograd:
            with torch.no_grad():
                out = text_module(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            hidden = out.last_hidden_state.detach()
        else:
            out = text_module(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            hidden = out.last_hidden_state
    hidden = hidden.float()
    if q_positions is None:
        h_last = hidden[:, -1, :]
    else:
        q_index = q_positions.view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
        h_last = torch.gather(hidden, 1, q_index)[:, 0, :]
    index = option_positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
    h_options = torch.gather(hidden, 1, index)
    return head(h_last, h_options)


class PointerHead(nn.Module):
    """dot-product pointer head with its own LayerNorms, kept in fp32.

    ``logits_i = (LN(h_last) Wq) · (LN(h_opt_i) Wk) / sqrt(dim)``

    The heads are the only *new* parameters besides LoRA; the body stays
    frozen. LayerNorm has no learnable affine transform in the upstream
    Strands-style head; the contract only requires ``fp32 + dim=256 +
    LayerNorm``, so affine is left off to stay closest to a pure
    pointer/projection readout.
    """

    def __init__(
        self,
        hidden_size: int,
        head_dim: int = 256,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.norm_eps = norm_eps
        self.q_norm = nn.LayerNorm(hidden_size, eps=norm_eps, elementwise_affine=False)
        self.k_norm = nn.LayerNorm(hidden_size, eps=norm_eps, elementwise_affine=False)
        self.wq = nn.Linear(hidden_size, head_dim, bias=False)
        self.wk = nn.Linear(hidden_size, head_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.scale = head_dim ** -0.5
        # fp32 head, as required by the contract
        self.float()

    def forward(self, h_last: torch.Tensor, h_options: torch.Tensor) -> torch.Tensor:
        """``h_last`` [B,H]; ``h_options`` [B,K,H] -> logits [B,K] (fp32)."""
        h_last = h_last.float()
        h_options = h_options.float()
        q = self.wq(self.q_norm(h_last))
        k = self.wk(self.k_norm(h_options))
        q = self.dropout(q)
        k = self.dropout(k)
        return torch.einsum("bd,bkd->bk", q, k) * self.scale


class BiodModel(nn.Module):
    """Frozen text tower (+ optional LoRA) with the two pointer heads."""

    def __init__(self, cfg: TrainConfig, device: str | None = None, init_adapter: str | None = None):
        super().__init__()
        self.cfg = cfg
        self._dtype = _torch_dtype(cfg.dtype)
        text, resolved = load_text_tower(cfg.base_model, dtype=cfg.dtype)
        self.base_model_name = resolved
        hidden_size = int(getattr(text.config, "hidden_size", 2560))
        self.hidden_size = hidden_size
        self.base_head_suffix_counts = self._suffix_counts(text, cfg.lora_targets)

        self.lora_enabled = bool(cfg.lora)
        self.adapter_loaded_from = None
        if init_adapter:
            # Load an existing (frozen) LoRA adapter: the body is adapted to the
            # stage-1 task, but only the heads will train in this run.
            from peft import PeftModel

            text = PeftModel.from_pretrained(text, init_adapter)
            for param in text.parameters():
                param.requires_grad_(False)
            if getattr(cfg, "adapter_trainable", False):
                for name, param in text.named_parameters():
                    if "lora_" in name:
                        param.requires_grad_(True)
            self.adapter_loaded_from = os.path.abspath(init_adapter)
        elif self.lora_enabled:
            text = self._apply_lora(text)
        self.text = text
        self.body_trainable = any(
            p.requires_grad for n, p in self.text.named_parameters() if "lora_" in n
        )
        self.tool_head = PointerHead(hidden_size, cfg.head_dim, cfg.head_dropout, cfg.head_norm_eps)
        self.kb_head = PointerHead(hidden_size, cfg.head_dim, cfg.head_dropout, cfg.head_norm_eps)
        if cfg.gradient_checkpointing and self.body_trainable:
            self.text.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            # With only LoRA trainable, the checkpointed graph needs a grad
            # request on the embedding output.
            if hasattr(self.text, "enable_input_require_grads"):
                self.text.enable_input_require_grads()
        if device:
            self.to(device)

    # -- construction helpers ------------------------------------------------
    @staticmethod
    def _suffix_counts(module: nn.Module, targets) -> dict[str, int]:
        counts = {name: 0 for name in targets}
        for name, _ in module.named_modules():
            suffix = name.rsplit(".", 1)[-1]
            if suffix in counts:
                counts[suffix] += 1
        return counts

    def _apply_lora(self, text: nn.Module) -> nn.Module:
        wrapped, matched = apply_lora(text, self.cfg, self.base_head_suffix_counts)
        self.lora_matched_counts = matched
        return wrapped

    # -- parameters ----------------------------------------------------------
    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def parameter_summary(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.trainable_parameters())
        lora = sum(
            p.numel() for n, p in self.named_parameters() if "lora_" in n and p.requires_grad
        )
        heads = sum(
            p.numel()
            for n, p in self.named_parameters()
            if n.startswith(("tool_head.", "kb_head."))
        )
        return {
            "total": total,
            "trainable": trainable,
            "body_trainable": bool(getattr(self, "body_trainable", False)),
            "adapter_loaded_from": getattr(self, "adapter_loaded_from", None),
            "trainable_pct": round(100.0 * trainable / max(1, total), 4),
            "lora": lora,
            "heads": heads,
            "hidden_size": self.hidden_size,
            "lora_enabled": self.lora_enabled,
            "lora_matched_counts": getattr(self, "lora_matched_counts", None),
        }

    def optimizer_groups(self) -> list[dict]:
        cfg = self.cfg
        lora_params, head_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_" in name:
                lora_params.append(param)
            elif name.startswith(("tool_head.", "kb_head.")):
                head_params.append(param)
            else:  # pragma: no cover - defensive
                raise RuntimeError(f"unexpected trainable parameter outside LoRA/heads: {name}")
        groups = []
        if lora_params:
            groups.append({"params": lora_params, "lr": cfg.lr_lora, "name": "lora"})
        if head_params:
            groups.append(
                {
                    "params": head_params,
                    "lr": cfg.lr_head,
                    "name": "heads",
                    "weight_decay": cfg.head_weight_decay,
                }
            )
        if not groups:
            raise RuntimeError("no trainable parameters (head-only with empty heads?)")
        return groups

    # -- forward -------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        option_positions: torch.Tensor,
        task: str,
        q_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return pointer logits ``[B, num_slots]`` in fp32."""
        if task not in {"tool", "kb"}:
            raise ValueError(f"unknown task {task!r}")
        head = self.tool_head if task == "tool" else self.kb_head
        return pointer_logits(
            self.text,
            head,
            input_ids,
            attention_mask,
            option_positions,
            q_positions=q_positions,
            autocast_dtype=self._dtype,
            # Frozen body: skip the backward graph entirely (forward-only cost);
            # heads still receive gradients.
            run_body_nograd=not self.body_trainable,
        )

    # -- checkpoint helpers --------------------------------------------------
    def save_heads(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "format": "biod-sft-v0-heads",
                "head_dim": self.cfg.head_dim,
                "hidden_size": self.hidden_size,
                "head_dropout": self.cfg.head_dropout,
                "head_norm_eps": self.cfg.head_norm_eps,
                "tool": self.tool_head.state_dict(),
                "kb": self.kb_head.state_dict(),
            },
            path,
        )

    def load_heads(self, path: str) -> dict:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if int(payload["hidden_size"]) != self.hidden_size:
            raise ValueError(
                f"head hidden_size {payload['hidden_size']} != model {self.hidden_size}"
            )
        if int(payload["head_dim"]) != self.cfg.head_dim:
            raise ValueError(f"head_dim {payload['head_dim']} != config {self.cfg.head_dim}")
        self.tool_head.load_state_dict(payload["tool"])
        self.kb_head.load_state_dict(payload["kb"])
        return payload

    def save_adapter(self, directory: str) -> None:
        if not self.lora_enabled:
            return
        os.makedirs(directory, exist_ok=True)
        self.text.save_pretrained(directory)
