"""Hyper-parameters and paths for Biod SFT v0.

Defaults follow ``docs/SFT-v0-执行契约.md`` §3 and
``docs/决策模型训练路线-从零训练到SFT.md`` §4.5. Everything is a plain dataclass
so it serialises straight into ``train_manifest.json`` / ``biod_config.json``.
"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass, field

# --------------------------------------------------------------------------- #
# frozen assets (contract §1)
# --------------------------------------------------------------------------- #
TOOL_CATALOG_DIR = "other/archive-pre-paper1-20261103/baselines/toollibrary_baseline/v1/tool_catalog"
KB_CATALOG_DIR = "other/archive-pre-paper1-20261103/baselines/kb_baseline/v1/knowledge_source"
TOOL_CATALOG_VERSION = "f74720a19c557b52"
KB_CATALOG_VERSION = "66383258b2d773c7"
TOOL_NUM_OPTIONS = 61
KB_NUM_OPTIONS = 32
NO_TOOL_ID = "NO_TOOL"

DEFAULT_DATA_DIR = "Datasets/v2/multitask"
DEFAULT_BASE_MODEL = "Qwen/Qwen3.5-4B-Base"
DEFAULT_HF_HOME = os.path.expanduser("~/.cache/huggingface")

# Contract §3: r=16, alpha=32, dropout=0.05, exactly these 12 target suffixes.
LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class TrainConfig:
    """All knobs for one SFT v0 run."""

    # --- data ---
    data_dir: str = DEFAULT_DATA_DIR
    task: str = "both"  # tool | kb | both
    train_split: str = "train"
    val_split: str = "val"
    max_train_rows: int = 0  # 0 = all (smoke sets this)
    max_val_rows: int = 200
    # max_len: 0 = auto (p99 of rendered token lengths, rounded to 256,
    # clamped to [4096, 8192]). Over-length samples are DROPPED, never truncated.
    max_len: int = 0
    min_len: int = 0
    option_shuffle: bool = True  # per-epoch option reorder + label remap
    order_seed: int = 0  # 0 -> seed

    # --- model ---
    base_model: str = DEFAULT_BASE_MODEL
    dtype: str = "bfloat16"
    head_dim: int = 256
    head_hidden: int = 1024  # multitask MLP heads (context/role/relay)
    head_dropout: float = 0.05
    head_norm_eps: float = 1e-5
    head_weight_decay: float = 0.01  # separate wd for the head projections
    # query position: "prompt" = last token of the whole prompt (contract),
    # "state" = last token of the <state> block (less option-text contamination)
    q_position: str = "prompt"
    lora: bool = True  # --head-only sets this False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = LORA_TARGETS
    adapter_trainable: bool = False  # when --init-adapter is used, keep LoRA trainable
    gradient_checkpointing: bool = True
    attn_impl: str = ""  # empty = transformers default

    # --- optimisation (contract §3) ---
    epochs: float = 1.0
    micro_batch: int = 4
    grad_accum: int = 8
    lr_lora: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    clip_grad: float = 1.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    seed: int = 20261022
    log_every: int = 1
    max_steps: int = 0  # 0 = epochs-driven; smoke sets a step cap
    save_every_epoch: bool = True
    # dev early stopping (contract §0: dev may be used for early stopping)
    eval_every_epoch: bool = True
    early_stop_patience: int = 2
    divergence_factor: float = 30.0  # stop if a step loss exceeds this x the first-3 average

    # --- runtime / bookkeeping ---
    out_dir: str = ""
    run_id: str = ""
    smoke: bool = False
    smoke_rows: int = 300  # gate: 200-500 rows
    smoke_steps: int = 40  # gate: 30-100 steps
    device: str = "cuda"
    num_threads: int = 8
    log_jsonl: str = ""
    notes: str = ""

    def resolved_order_seed(self) -> int:
        return self.order_seed or self.seed

    def to_dict(self) -> dict:
        d = asdict(self)
        d["lora_targets"] = list(self.lora_targets)
        return d

    def hyperparams(self) -> dict:
        """The subset that belongs in train_manifest.json."""
        return {
            "task": self.task,
            "epochs": self.epochs,
            "micro_batch": self.micro_batch,
            "grad_accum": self.grad_accum,
            "effective_batch": self.micro_batch * self.grad_accum,
            "lr_lora": self.lr_lora,
            "lr_head": self.lr_head,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "clip_grad": self.clip_grad,
            "seed": self.seed,
            "max_len": self.max_len,
            "head_dim": self.head_dim,
            "head_dropout": self.head_dropout,
            "head_norm_eps": self.head_norm_eps,
            "head_weight_decay": self.head_weight_decay,
            "q_position": self.q_position,
            "dtype": self.dtype,
            "gradient_checkpointing": self.gradient_checkpointing,
            "lora": {
                "enabled": self.lora,
                "r": self.lora_r,
                "alpha": self.lora_alpha,
                "dropout": self.lora_dropout,
                "targets": list(self.lora_targets),
            },
            "option_shuffle": self.option_shuffle,
            "order_seed": self.resolved_order_seed(),
            "max_steps": self.max_steps,
        }


def new_run_id(task: str, smoke: bool = False, now: float | None = None) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now if now is not None else time.time()))
    return f"{'smoke' if smoke else 'sft'}-{task}-{stamp}"


def setup_offline_env() -> None:
    """HF must be offline on the host (network hangs); cache lives in HF_HOME."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if "HF_HOME" not in os.environ and os.path.isdir(DEFAULT_HF_HOME):
        os.environ["HF_HOME"] = DEFAULT_HF_HOME


def resolve_base_model(explicit: str | None = None) -> str:
    """Resolve the base model path/id: explicit > BIOD_BASE_MODEL > HF hub id."""
    setup_offline_env()
    if explicit:
        return explicit
    env = os.environ.get("BIOD_BASE_MODEL")
    if env:
        return env
    return DEFAULT_BASE_MODEL
