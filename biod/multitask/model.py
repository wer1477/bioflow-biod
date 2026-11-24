"""Multitask decision model: frozen text tower + LoRA + five heads.

One forward per decision. The three new heads are small MLPs on top of the
frozen (LoRA-adapted) hidden states; tool/kb keep the v0 dot-product pointer
readout so their behaviour and latency stay comparable with biod-v0.
"""

from __future__ import annotations

import json
import os
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from biod.train.config import TrainConfig
from biod.train.lora import apply_lora, suffix_counts
from biod.train.model import PointerHead, load_text_tower

from .tasks import OVERCLAIM_FLAGS, PREMISE_LABELS, ROUTING_TASKS, TASKS

HEAD_NAMES = (
    "tool_head",
    "kb_head",
    "context_head",
    "role_head",
    "relay_head",
    "reason_head",
    "premise_head",
    "flag_head",
)


class ClassHead(nn.Module):
    """LayerNorm -> MLP -> n_classes, fp32."""

    def __init__(self, hidden_size: int, n_classes: int, hidden: int = 1024, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False),
            nn.Linear(hidden_size, hidden, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes, bias=True),
        )
        self.float()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h.float())


class GroupHead(nn.Module):
    """Shared 3-class head applied to every group span."""

    def __init__(self, hidden_size: int, n_classes: int = 3, hidden: int = 1024, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False),
            nn.Linear(hidden_size, hidden, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes, bias=True),
        )
        self.float()

    def forward(self, h_groups: torch.Tensor) -> torch.Tensor:
        # [B, G, H] -> [B, G, 3]
        return self.net(h_groups.float())


def _gather(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Gather ``hidden[b, positions[b, k]]`` for every b,k."""
    if positions.dim() == 1:
        positions = positions.unsqueeze(1)
    index = positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
    return torch.gather(hidden, 1, index)


class MultiTaskBiodModel(nn.Module):
    def __init__(
        self,
        cfg: TrainConfig,
        device: str | None = None,
        init_adapter: str | None = None,
        use_reason_head: bool = False,
        routing_only: bool = False,
        train_adapter: bool = False,
    ):
        super().__init__()
        self.cfg = cfg
        self.tasks = ROUTING_TASKS if routing_only else TASKS
        self._dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32
        text, resolved = load_text_tower(cfg.base_model, dtype=cfg.dtype)
        self.base_model_name = resolved
        self.hidden_size = int(text.config.hidden_size)
        base_counts = suffix_counts(text, cfg.lora_targets)
        self.lora_matched_counts = None
        if init_adapter:
            from peft import PeftModel

            text = PeftModel.from_pretrained(text, init_adapter, is_trainable=train_adapter)
        elif cfg.lora:
            text, matched = apply_lora(text, cfg, base_counts)
            self.lora_matched_counts = matched
        self.text = text
        self.body_trainable = any(p.requires_grad for n, p in self.text.named_parameters() if "lora_" in n)

        self.tool_head = PointerHead(self.hidden_size, cfg.head_dim, cfg.head_dropout, cfg.head_norm_eps)
        self.kb_head = PointerHead(self.hidden_size, cfg.head_dim, cfg.head_dropout, cfg.head_norm_eps)
        self.context_head = GroupHead(self.hidden_size, 3, hidden=cfg.head_hidden, dropout=cfg.head_dropout)
        self.role_head = ClassHead(self.hidden_size, 5, hidden=cfg.head_hidden, dropout=cfg.head_dropout)
        self.relay_head = ClassHead(self.hidden_size, 3, hidden=cfg.head_hidden, dropout=cfg.head_dropout)
        self.premise_head = None
        self.flag_head = None
        if not routing_only:
            self.premise_head = ClassHead(self.hidden_size, len(PREMISE_LABELS), hidden=cfg.head_hidden, dropout=cfg.head_dropout)
            self.flag_head = ClassHead(self.hidden_size, len(OVERCLAIM_FLAGS), hidden=cfg.head_hidden, dropout=cfg.head_dropout)
        self.use_reason_head = use_reason_head
        self.reason_head = ClassHead(self.hidden_size, 6, hidden=cfg.head_hidden, dropout=cfg.head_dropout) if use_reason_head else None

        if cfg.gradient_checkpointing and self.body_trainable:
            self.text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(self.text, "enable_input_require_grads"):
                self.text.enable_input_require_grads()
        if device:
            self.to(device)

    # -- forward -------------------------------------------------------------
    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self._dtype)
            if input_ids.is_cuda and self._dtype != torch.float32
            else nullcontext()
        )
        with autocast_ctx:
            out = self.text(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return out.last_hidden_state

    def forward(self, batch: dict) -> dict:
        task = batch["task"]
        if task not in self.tasks:
            raise ValueError(f"task {task!r} is not supported by this model")
        hidden = self._encode(batch["input_ids"], batch["attention_mask"])
        if task == "tool_select":
            q = _gather(hidden, batch["q_positions"])[:, 0, :]
            opts = _gather(hidden, batch["option_positions"])
            return {"logits": self.tool_head(q, opts)}
        if task == "kb_select":
            q = _gather(hidden, batch["q_positions"])[:, 0, :]
            opts = _gather(hidden, batch["option_positions"])
            return {"logits": self.kb_head(q, opts)}
        if task == "context_retention":
            groups = _gather(hidden, batch["group_positions"])
            return {"logits": self.context_head(groups)}
        q = _gather(hidden, batch["q_positions"])[:, 0, :]
        if task == "role_select":
            return {"logits": self.role_head(q)}
        if task == "relay_select":
            out = {"logits": self.relay_head(q)}
            if self.reason_head is not None:
                out["reason_logits"] = self.reason_head(q)
            return out
        if task == "premise_judge":
            return {"logits": self.premise_head(q)}
        if task == "overclaim_flag":
            return {"logits": self.flag_head(q)}
        raise ValueError(f"unknown task {task!r}")

    # -- losses --------------------------------------------------------------
    def compute_loss(self, batch: dict, outputs: dict) -> tuple[torch.Tensor, dict]:
        task = batch["task"]
        logits = outputs["logits"]
        if task == "tool_select":
            loss = F.cross_entropy(logits, batch["labels"])
            pred = logits.argmax(-1)
            acc = (pred == batch["labels"]).float().mean()
        elif task == "kb_select":
            loss = F.binary_cross_entropy_with_logits(logits, batch["kb_soft"])
            acc = torch.tensor(0.0, device=logits.device)
        elif task == "context_retention":
            weight = torch.tensor([1.0, 1.0, 1.3], device=logits.device)
            loss = F.cross_entropy(logits.view(-1, 3), batch["labels"].view(-1), weight=weight,
                                   ignore_index=-100, label_smoothing=0.03)
            mask = batch["labels"] != -100
            pred = logits.argmax(-1)
            acc = ((pred == batch["labels"]) & mask).float().sum() / mask.float().sum().clamp_min(1)
        elif task == "premise_judge":
            loss = F.cross_entropy(logits, batch["labels"], label_smoothing=0.03)
            pred = logits.argmax(-1)
            acc = (pred == batch["labels"]).float().mean()
        elif task == "overclaim_flag":
            target = batch["labels"]
            pos = target.sum(0)
            neg = target.shape[0] - pos
            pos_weight = (neg / pos.clamp_min(1.0)).clamp(0.5, 5.0)
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
            pred = (logits.sigmoid() >= 0.5).float()
            acc = (pred == target).float().mean()
        else:
            if task == "role_select":
                weight = torch.tensor([1.0, 1.0, 1.0, 1.2, 1.6], device=logits.device)
            else:
                weight = torch.tensor([1.0, 1.8, 1.5], device=logits.device)
            loss = F.cross_entropy(logits, batch["labels"], weight=weight, label_smoothing=0.03)
            pred = logits.argmax(-1)
            acc = (pred == batch["labels"]).float().mean()
        return loss, {"loss": float(loss.detach()), "acc": float(acc.detach())}

    # -- parameters ----------------------------------------------------------
    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def optimizer_groups(self) -> list[dict]:
        cfg = self.cfg
        lora, heads = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "lora_" in name:
                lora.append(p)
            elif name.startswith(tuple(f"{h}." for h in HEAD_NAMES)):
                heads.append(p)
            else:
                raise RuntimeError(f"unexpected trainable parameter: {name}")
        groups = []
        if lora:
            groups.append({"params": lora, "lr": cfg.lr_lora, "name": "lora"})
        if heads:
            groups.append({"params": heads, "lr": cfg.lr_head, "name": "heads", "weight_decay": cfg.head_weight_decay})
        if not groups:
            raise RuntimeError("no trainable parameters")
        return groups

    def parameter_summary(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.trainable_parameters())
        return {
            "base_model": self.base_model_name,
            "hidden_size": self.hidden_size,
            "total": total,
            "trainable": trainable,
            "trainable_pct": round(100.0 * trainable / max(1, total), 4),
            "lora": sum(p.numel() for n, p in self.named_parameters() if "lora_" in n and p.requires_grad),
            "heads": sum(p.numel() for n, p in self.named_parameters() if n.startswith(tuple(f"{h}." for h in HEAD_NAMES))),
            "lora_matched_counts": self.lora_matched_counts,
        }

    # -- checkpoint ----------------------------------------------------------
    def save_checkpoint(self, out_dir: str, meta: dict | None = None) -> None:
        os.makedirs(out_dir, exist_ok=True)
        if self.cfg.lora:
            self.text.save_pretrained(os.path.join(out_dir, "adapter"))
        heads = {
            "format": "biod-multitask-v1-heads",
            "hidden_size": self.hidden_size,
            "head_dim": self.cfg.head_dim,
            "head_hidden": self.cfg.head_hidden,
            "head_dropout": self.cfg.head_dropout,
            "head_norm_eps": self.cfg.head_norm_eps,
            "tool": self.tool_head.state_dict(),
            "kb": self.kb_head.state_dict(),
            "context": self.context_head.state_dict(),
            "role": self.role_head.state_dict(),
            "relay": self.relay_head.state_dict(),
        }
        if self.premise_head is not None:
            heads["premise"] = self.premise_head.state_dict()
        if self.flag_head is not None:
            heads["flag"] = self.flag_head.state_dict()
        if self.reason_head is not None:
            heads["reason"] = self.reason_head.state_dict()
        torch.save(heads, os.path.join(out_dir, "heads.pt"))
        config = {
            "base_model": self.cfg.base_model,
            "resolved_base_model": self.base_model_name,
            "hidden_size": self.hidden_size,
            "head_dim": self.cfg.head_dim,
            "head_hidden": self.cfg.head_hidden,
            "tasks": list(self.tasks),
            "lora": {
                "r": self.cfg.lora_r, "alpha": self.cfg.lora_alpha, "dropout": self.cfg.lora_dropout,
                "targets": list(self.cfg.lora_targets),
            },
            "meta": meta or {},
        }
        with open(os.path.join(out_dir, "model_config.json"), "w", encoding="utf-8") as fh:
            json.dump(config, fh, ensure_ascii=False, indent=1)

    def load_heads(self, path: str) -> dict:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if int(payload["hidden_size"]) != self.hidden_size:
            raise ValueError(f"head hidden_size {payload['hidden_size']} != {self.hidden_size}")
        self.tool_head.load_state_dict(payload["tool"])
        self.kb_head.load_state_dict(payload["kb"])
        self.context_head.load_state_dict(payload["context"])
        self.role_head.load_state_dict(payload["role"])
        self.relay_head.load_state_dict(payload["relay"])
        if self.premise_head is not None and "premise" in payload:
            self.premise_head.load_state_dict(payload["premise"])
        if self.flag_head is not None and "flag" in payload:
            self.flag_head.load_state_dict(payload["flag"])
        if self.reason_head is not None and "reason" in payload:
            self.reason_head.load_state_dict(payload["reason"])
        return payload
