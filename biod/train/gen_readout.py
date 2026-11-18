"""Generative (CLM) readout — backup for the pointer head.

Motivation: the pointer head only learns at the junction; if the frozen-body
pointer model underperforms the zero-training baseline, this module gives a
second, independent readout that the LoRA body can drive directly:

* rendering is unchanged (``biod.render.render_tool_prompt``);
* the answer is appended after ``<answer>`` as the option number on that row
  (``" 17"``) or ``" NO_TOOL"``;
* loss = standard causal-LM cross-entropy **only on the answer tokens**
  (prompt tokens are masked with -100);
* inference prefills the prompt and greedily decodes 1-3 answer tokens under a
  trie constraint (only valid answer strings), then maps the sequence back to a
  option id. ``GenPredictor.predict_tool`` returns the same
  ``[(id, prob)]`` shape as ``BiodPredictor`` (one-hot over the decoded slot),
  so the eval harness can swap it in without other changes.

Training CLI::

    .venv/bin/python -m biod.train.gen_readout --task tool \\
        --data-dir artifacts/datasets/v1 --epochs 1 --out artifacts/runs/biod-v0-gen

Checkpoint layout: ``adapter/`` + ``gen_config.json`` + ``train_manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import torch
import torch.nn as nn
import torch.nn.functional as F

from biod.render import render_tool_prompt

from .collate import make_order
from .config import (
    NO_TOOL_ID,
    TOOL_CATALOG_VERSION,
    TrainConfig,
    new_run_id,
    resolve_base_model,
    setup_offline_env,
)
from .data import load_catalog, load_task_records, sha256_file
from .model import _torch_dtype, load_text_tower

ANSWER_FORMAT_VERSION = 1
ANSWER_SEPARATOR = " "  # answer text = " " + number / "NO_TOOL"

__all__ = [
    "answer_string",
    "render_gen_prompt",
    "GenExample",
    "build_gen_example",
    "pad_gen_batch",
    "gen_ce_loss",
    "BiodGenModel",
    "GenPredictor",
    "train_main",
]


# --------------------------------------------------------------------------- #
# rendering / tokenization
# --------------------------------------------------------------------------- #
def answer_string(display_pos: int, num_options: int) -> str:
    """Option number for a display slot; the last slot is NO_TOOL."""
    if display_pos >= num_options:
        return NO_TOOL_ID
    return str(display_pos + 1)


def render_gen_prompt(
    state: str,
    options,
    order: list[int] | None = None,
):
    """Return ``(prefix, answers)``; prefix ends with ``<answer>``."""
    prompt = render_tool_prompt(state, options, order=order)
    prefix = prompt.text  # ends with "<answer>"
    answers = [answer_string(i, len(options)) for i in range(prompt.num_options)]
    return prefix, answers, prompt


@dataclass
class GenExample:
    record_id: str
    input_ids: list[int]
    predict_positions: list[int]  # hidden positions used to predict answer tokens
    answer_ids: list[int]
    slot: int  # display slot of the gold option (last = NO_TOOL)
    gold_id: str
    prompt_len: int
    option_ids: list[str]
    order: list[int]

    @property
    def num_answer_tokens(self) -> int:
        return len(self.answer_ids)


def _tokenize(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def build_gen_example(
    record: dict,
    options,
    id_to_index: dict[str, int],
    tokenizer,
    max_len: int = 0,
    order: list[int] | None = None,
) -> GenExample | None:
    prefix, answers, prompt = render_gen_prompt(record["state_text"], options, order=order)
    gold = record["gold_primary"]
    if gold == NO_TOOL_ID:
        slot = prompt.meta["no_tool_slot"]
    else:
        slot = prompt.canonical_indices.index(id_to_index[gold])
    if prompt.option_ids[slot] != gold:
        raise AssertionError(f"{record['id']}: generative label mismatch")
    prefix_ids = _tokenize(tokenizer, prefix)
    answer_ids = _tokenize(tokenizer, ANSWER_SEPARATOR + answers[slot])
    ids = prefix_ids + answer_ids
    if max_len and len(ids) > max_len:
        return None
    answer_start = len(prefix_ids)
    predict_positions = [answer_start + i - 1 for i in range(len(answer_ids))]
    return GenExample(
        record_id=record["id"],
        input_ids=ids,
        predict_positions=predict_positions,
        answer_ids=answer_ids,
        slot=slot,
        gold_id=gold,
        prompt_len=len(ids),
        option_ids=list(prompt.option_ids),
        order=list(prompt.canonical_indices),
    )


def build_gen_examples(
    records,
    options,
    id_to_index,
    tokenizer,
    max_len=0,
    rng: random.Random | None = None,
    shuffle_options: bool = True,
):
    examples, dropped = [], 0
    n = len(options)
    for record in records:
        order = make_order(n, rng) if (shuffle_options and rng is not None) else None
        ex = build_gen_example(record, options, id_to_index, tokenizer, max_len, order)
        if ex is None:
            dropped += 1
            continue
        examples.append(ex)
    return examples, {"built": len(examples), "dropped": dropped}


def pad_gen_batch(examples, pad_token_id: int, device: str | None = None) -> dict:
    """Left-pad; answer prediction positions are shifted by the pad width."""
    if not examples:
        raise ValueError("empty batch")
    width = max(len(ex.input_ids) for ex in examples)
    max_answer = max(ex.num_answer_tokens for ex in examples)
    batch_size = len(examples)
    input_ids = torch.full((batch_size, width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, width), dtype=torch.long)
    predict_positions = torch.zeros((batch_size, max_answer), dtype=torch.long)
    answer_ids = torch.full((batch_size, max_answer), -100, dtype=torch.long)
    slots = torch.zeros((batch_size,), dtype=torch.long)
    for row, ex in enumerate(examples):
        pad = width - len(ex.input_ids)
        input_ids[row, pad:] = torch.tensor(ex.input_ids, dtype=torch.long)
        attention_mask[row, pad:] = 1
        n = ex.num_answer_tokens
        predict_positions[row, :n] = torch.tensor([p + pad for p in ex.predict_positions], dtype=torch.long)
        answer_ids[row, :n] = torch.tensor(ex.answer_ids, dtype=torch.long)
        slots[row] = ex.slot
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "predict_positions": predict_positions,
        "answer_ids": answer_ids,
        "slots": slots,
        "gold_ids": [ex.gold_id for ex in examples],
        "option_ids": [ex.option_ids for ex in examples],
        "record_ids": [ex.record_id for ex in examples],
    }
    if device:
        for key, value in batch.items():
            if hasattr(value, "to"):
                batch[key] = value.to(device, non_blocking=True)
    return batch


def gen_ce_loss(logits: torch.Tensor, answer_ids: torch.Tensor) -> torch.Tensor:
    """``logits`` [B,A,V], ``answer_ids`` [B,A] (-100 = padding)."""
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        answer_ids.reshape(-1),
        ignore_index=-100,
    )


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class BiodGenModel(nn.Module):
    """Frozen text tower (+ optional LoRA) with a tied-embedding LM readout."""

    def __init__(self, cfg: TrainConfig, device: str | None = None):
        super().__init__()
        self.cfg = cfg
        self._dtype = _torch_dtype(cfg.dtype)
        text, resolved = load_text_tower(cfg.base_model, dtype=cfg.dtype)
        self.base_model_name = resolved
        self.hidden_size = int(getattr(text.config, "hidden_size", 2560))
        self.vocab_size = int(getattr(text.config, "vocab_size", 248320))
        self.lora_enabled = bool(cfg.lora)
        if self.lora_enabled:
            from peft import LoraConfig, get_peft_model

            text = get_peft_model(
                text,
                LoraConfig(
                    r=cfg.lora_r,
                    lora_alpha=cfg.lora_alpha,
                    lora_dropout=cfg.lora_dropout,
                    bias="none",
                    target_modules=list(cfg.lora_targets),
                    task_type="FEATURE_EXTRACTION",
                ),
            )
            if cfg.gradient_checkpointing:
                text.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                if hasattr(text, "enable_input_require_grads"):
                    text.enable_input_require_grads()
        self.text = text
        self.embed_weight = text.get_input_embeddings().weight  # tied with LM head
        if device:
            self.to(device)

    def forward(self, batch: dict) -> torch.Tensor:
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self._dtype)
            if batch["input_ids"].is_cuda and self._dtype != torch.float32
            else torch.enable_grad()
        )
        with autocast_ctx:
            out = self.text(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            hidden = out.last_hidden_state
            index = batch["predict_positions"].unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
            h = torch.gather(hidden, 1, index).float()
        logits = F.linear(h, self.embed_weight.float())
        return logits

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def optimizer_groups(self) -> list[dict]:
        params = []
        for name, p in self.named_parameters():
            if p.requires_grad and "lora_" in name:
                params.append(p)
        if not params:
            raise RuntimeError("no trainable LoRA parameters")
        return [{"params": params, "lr": self.cfg.lr_lora, "name": "lora"}]

    def save_adapter(self, directory: str) -> None:
        if self.lora_enabled:
            os.makedirs(directory, exist_ok=True)
            self.text.save_pretrained(directory)


# --------------------------------------------------------------------------- #
# constrained greedy inference
# --------------------------------------------------------------------------- #
class _AnswerTrie:
    """Token trie over the valid answer strings of a fixed option set."""

    def __init__(self, tokenizer, num_options: int):
        self.tokenizer = tokenizer
        self.num_options = num_options
        self.answers: list[str] = [answer_string(i, num_options) for i in range(num_options + 1)]
        self.token_seqs: list[list[int]] = [
            _tokenize(tokenizer, ANSWER_SEPARATOR + a) for a in self.answers
        ]
        self.root: dict = {}
        for slot, seq in enumerate(self.token_seqs):
            node = self.root
            for tok in seq:
                node = node.setdefault(tok, {})
            node["__slot__"] = slot

    def allowed(self, path: list[int]) -> list[int]:
        node = self.root
        for tok in path:
            node = node.get(tok)
            if node is None:
                return []
        return [k for k in node.keys() if k != "__slot__"]

    def slot_matches(self, path: list[int]) -> list[int]:
        node = self.root
        for tok in path:
            node = node.get(tok)
            if node is None:
                return []
        slots = []
        if "__slot__" in node:
            slots.append(node["__slot__"])
        return slots


class GenPredictor:
    """Greedy constrained-decoding readout with the BiodPredictor interface."""

    def __init__(self, ckpt_dir: str, device: str | None = None, dtype: str | None = None):
        self.ckpt_dir = ckpt_dir
        with open(os.path.join(ckpt_dir, "gen_config.json"), encoding="utf-8") as fh:
            self.config = json.load(fh)
        cfg = self.config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or cfg.get("dtype", "bfloat16")
        self._torch_dtype = _torch_dtype(self.dtype)
        self.max_len = int(cfg.get("max_len") or 0)
        base = cfg.get("base_model_path") or cfg["base_model"]
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        text, resolved = load_text_tower(base, dtype=self.dtype)
        adapter_dir = os.path.join(ckpt_dir, "adapter")
        self.lora_enabled = bool(cfg.get("lora", {}).get("enabled", True)) and os.path.isdir(adapter_dir)
        if self.lora_enabled:
            from peft import PeftModel

            text = PeftModel.from_pretrained(text, adapter_dir)
        text.to(self.device).eval()
        self.text = text
        self.embed_weight = text.get_input_embeddings().weight
        self.options = [
            self._option(entry) for entry in cfg["tool"]["options"]
        ]
        self.option_ids = [o.id for o in self.options] + [NO_TOOL_ID]
        self.trie = _AnswerTrie(self.tokenizer, len(self.options))
        self._max_new_tokens = max(len(s) for s in self.trie.token_seqs)

    @staticmethod
    def _option(entry: dict):
        from biod.catalog import Option

        return Option(
            index=0,
            name=entry["name"],
            description=entry.get("description", ""),
            id=entry["id"],
            kind="tool",
        )

    @torch.no_grad()
    def _next_logits(self, ids: list[int]) -> torch.Tensor:
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.autocast(
            device_type="cuda",
            dtype=self._torch_dtype,
            enabled=self.device.startswith("cuda") and self._torch_dtype != torch.float32,
        ):
            out = self.text(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            h = out.last_hidden_state[:, -1, :].float()
        return F.linear(h, self.embed_weight.float())[0]

    def predict_tool(self, state_text: str, order: str = "canonical") -> list[tuple[str, float]]:
        from biod.render import reversed_order

        n = len(self.options)
        display = list(range(n)) if order == "canonical" else reversed_order(n)
        prefix, answers, prompt = render_gen_prompt(state_text, self.options, order=display)
        ids = _tokenize(self.tokenizer, prefix)
        if self.max_len and len(ids) + 8 > self.max_len:
            raise RuntimeError("over-length prompt for generative readout")
        path: list[int] = []
        slot: int | None = None
        for _ in range(self._max_new_tokens):
            allowed = self.trie.allowed(path)
            if not allowed:
                break
            logits = self._next_logits(ids + path)
            masked = torch.full_like(logits, float("-inf"))
            allowed_t = torch.tensor(allowed, dtype=torch.long, device=logits.device)
            masked[allowed_t] = logits[allowed_t]
            tok = int(masked.argmax().item())
            path.append(tok)
            matches = self.trie.slot_matches(path)
            if matches:
                slot = matches[0]
                break
        probs_by_display = [0.0] * prompt.num_options
        if slot is not None:
            probs_by_display[slot] = 1.0
        id_to_prob = {oid: p for oid, p in zip(prompt.option_ids, probs_by_display)}
        return [(oid, float(id_to_prob.get(oid, 0.0))) for oid in self.option_ids]


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def _git_info() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20
        ).stdout.strip()
    except Exception:  # pragma: no cover
        commit = ""
    return {"commit": commit}


def train_main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train generative (CLM) tool readout")
    p.add_argument("--data-dir", default="artifacts/datasets/v1")
    p.add_argument("--out", default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-len", type=int, default=0)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--micro-batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr-lora", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=20261022)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--verify-ckpt", dest="verify_ckpt", action="store_true", default=True)
    p.add_argument("--no-verify-ckpt", dest="verify_ckpt", action="store_false")
    p.add_argument("--notes", default="")
    args = p.parse_args(argv)

    setup_offline_env()
    run_id = args.run_id or new_run_id("tool-gen", smoke=False)
    out_dir = args.out or os.path.join("artifacts", "runs", run_id)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train_log.jsonl")
    start = time.time()

    def log(event: str, **fields) -> None:
        payload = {"ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "event": event, **fields}
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        scalars = " ".join(f"{k}={v}" for k, v in fields.items() if not isinstance(v, (dict, list)))
        print(f"[gen] {event} {scalars}".rstrip(), flush=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda:0" if args.device == "cuda" else args.device
    base_model = resolve_base_model(args.model)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    catalog = load_catalog("tool")
    records = load_task_records(args.data_dir, "tool", "train", catalog, limit=args.max_rows)
    try:
        val_records = load_task_records(args.data_dir, "tool", "val", catalog, limit=200)
    except FileNotFoundError:
        val_records = []
    log("run_start", run_id=run_id, n_train=len(records), n_val=len(val_records), base_model=base_model)

    # max_len from answer-inclusive lengths
    if not args.max_len:
        lengths = []
        for rec in records:
            _, _, prompt = render_gen_prompt(rec["state_text"], catalog.options)
            lengths.append(len(_tokenize(tokenizer, prompt.text)) + 4)
        lengths.sort()
        p99 = lengths[min(len(lengths) - 1, int(round(0.99 * (len(lengths) - 1))))]
        args.max_len = max(4096, min(8192, int(math.ceil(p99 / 256.0) * 256)))
    log("max_len", value=args.max_len)

    cfg = TrainConfig()
    cfg.base_model = args.model or base_model
    cfg.dtype = args.dtype
    cfg.lora = True
    cfg.lora_r = args.lora_r
    cfg.lora_alpha = args.lora_alpha
    cfg.gradient_checkpointing = True
    cfg.lr_lora = args.lr_lora
    model = BiodGenModel(cfg, device=device)
    model.train()
    trainable = sum(p.numel() for p in model.trainable_parameters())
    log("model", trainable=trainable, hidden_size=model.hidden_size)

    steps_per_epoch = max(1, math.ceil(math.ceil(len(records) / args.micro_batch) / args.grad_accum))
    total_steps = args.max_steps or max(1, int(math.ceil(steps_per_epoch * args.epochs)))
    warmup = max(1, int(total_steps * args.warmup_ratio))
    opt = torch.optim.AdamW(model.optimizer_groups(), lr=args.lr_lora, weight_decay=args.weight_decay)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    examples: list[GenExample] = []
    losses: list[float] = []
    step = 0
    epoch = 0
    cursor = 0
    t0 = time.time()
    while step < total_steps:
        if cursor >= len(examples):
            rng = random.Random(args.seed * 100_003 + epoch * 1009)
            order = list(range(len(records)))
            rng.shuffle(order)
            examples, stats = build_gen_examples(
                [records[i] for i in order], catalog.options, catalog.id_to_index,
                tokenizer, max_len=args.max_len, rng=rng, shuffle_options=True,
            )
            cursor = 0
            log("epoch_data", epoch=epoch, **stats)
            epoch += 1
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.grad_accum):
            chunk = examples[cursor : cursor + args.micro_batch]
            cursor += args.micro_batch
            if not chunk:
                break
            batch = pad_gen_batch(chunk, tokenizer.pad_token_id or tokenizer.eos_token_id, device)
            logits = model(batch)
            loss = gen_ce_loss(logits, batch["answer_ids"])
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite gen loss at step {step}")
            (loss / args.grad_accum).backward()
            step_loss += float(loss.item()) / args.grad_accum
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), args.clip_grad)
        opt.step()
        sched.step()
        step += 1
        losses.append(step_loss)
        if step % 5 == 0 or step == total_steps:
            log(
                "step", step=step, loss=round(step_loss, 4),
                s_per_step=round((time.time() - t0) / max(1, step), 2),
                max_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2)
                if torch.cuda.is_available() else 0.0,
            )

    model.save_adapter(os.path.join(out_dir, "adapter"))
    gen_config = {
        "format": "biod-gen-v0",
        "answer_format_version": ANSWER_FORMAT_VERSION,
        "base_model": cfg.base_model,
        "base_model_path": model.base_model_name,
        "dtype": args.dtype,
        "hidden_size": model.hidden_size,
        "max_len": args.max_len,
        "seed": args.seed,
        "lora": {"enabled": True, "r": args.lora_r, "alpha": args.lora_alpha},
        "git": _git_info(),
        "catalog_version": TOOL_CATALOG_VERSION,
        "tool": {
            "num_options": len(catalog.options),
            "no_tool_id": NO_TOOL_ID,
            "options": [
                {"id": o.id, "name": o.name, "description": o.description}
                for o in catalog.options
            ],
        },
    }
    with open(os.path.join(out_dir, "gen_config.json"), "w", encoding="utf-8") as fh:
        json.dump(gen_config, fh, ensure_ascii=False, indent=2)
    manifest = {
        "run_id": run_id,
        "format": "biod-gen-v0",
        "git": _git_info(),
        "base_model": cfg.base_model,
        "data": {
            "train": os.path.join(args.data_dir, "tool_train.jsonl"),
            "train_sha256": sha256_file(os.path.join(args.data_dir, "tool_train.jsonl")),
            "count": len(records),
        },
        "hyperparams": {
            "lr_lora": args.lr_lora,
            "micro_batch": args.micro_batch,
            "grad_accum": args.grad_accum,
            "epochs": args.epochs,
            "seed": args.seed,
            "max_len": args.max_len,
        },
        "loss": {"history": [round(x, 5) for x in losses]},
        "elapsed_s": round(time.time() - start, 1),
        "notes": args.notes,
    }
    with open(os.path.join(out_dir, "train_manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    log("train_done", steps=step, loss_last=round(sum(losses[-5:]) / max(1, len(losses[-5:])), 4))
    if args.verify_ckpt:
        try:
            predictor = GenPredictor(out_dir)
            scores = predictor.predict_tool(records[0]["state_text"])
            top = sorted(scores, key=lambda kv: -kv[1])[:3]
            log("verify", ok=True, n=len(scores), top3=top)
        except Exception as exc:  # pragma: no cover
            log("verify_failed", error=f"{type(exc).__name__}: {exc}")
    print("GEN_TRAIN_RESULT " + json.dumps({"out": out_dir, "steps": step,
                                            "loss_last": round(sum(losses[-5:]) / max(1, len(losses[-5:])), 4)}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(train_main())
