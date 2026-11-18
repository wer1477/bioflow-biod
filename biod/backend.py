"""Model backends.

The readout layer only needs one primitive: raw next-token logits for a small
set of candidate tokens, for a batch of prompts. Keeping the interface this
narrow means:

* :class:`HfBackend` can stay a thin transformers wrapper (with left padding so
  the last position is always a real token), and
* :class:`MockBackend` can exercise the *entire* decision/replay pipeline in
  unit tests without torch, a GPU, or model weights.

Real backend
------------
``HfBackend`` lazily imports torch/transformers so that the rest of the package
(and the tests) run on a plain Python install.
"""

from __future__ import annotations

import hashlib
import math
import struct
from abc import ABC, abstractmethod
from typing import Sequence


class Backend(ABC):
    """Minimal interface required by the readout layer."""

    name: str = "backend"

    @abstractmethod
    def score_tokens(self, prompts: Sequence[str], token_texts: Sequence[str]) -> list[dict[str, float]]:
        """Return raw logits for ``token_texts`` at the last position of each prompt.

        Shape: ``len(prompts)`` dicts mapping ``token_text -> logit``.
        """

    def close(self) -> None:  # pragma: no cover - optional hook
        pass


# --------------------------------------------------------------------------- #
# deterministic mock backend (tests / pipeline smoke, no GPU)
# --------------------------------------------------------------------------- #
class MockBackend(Backend):
    """Deterministic pseudo-logits derived from ``sha256(prompt, token)``.

    It is *not* meant to be meaningful. Its only job is to let the rendering,
    readout, decision and replay code be tested end-to-end without a model.
    A prompt that literally contains a candidate's name/description gets a
    small boost, so mock runs still produce interpretable rankings.
    """

    def __init__(self, seed: str = "biod"):
        self.name = "mock"
        self.seed = seed
        self._vocab: dict[str, int] = {}

    def _hash(self, text: str) -> float:
        digest = hashlib.sha256((self.seed + "\x00" + text).encode("utf-8")).digest()
        # map first 8 bytes to [0, 1)
        (val,) = struct.unpack(">Q", digest[:8])
        return val / 2**64

    def score_tokens(self, prompts: Sequence[str], token_texts: Sequence[str]) -> list[dict[str, float]]:
        out: list[dict[str, float]] = []
        for prompt in prompts:
            row: dict[str, float] = {}
            for tok in token_texts:
                if tok not in self._vocab:
                    self._vocab[tok] = len(self._vocab)
                # base deterministic noise in [0, 2]
                val = self._hash(prompt + "\x00" + tok) * 2.0
                # crude lexical prior: option text appearing in the prompt
                if len(tok) > 2 and tok in prompt:
                    val += 0.5
                row[tok] = val
            out.append(row)
        return out

    def vocab_size(self) -> int:
        return len(self._vocab)


# --------------------------------------------------------------------------- #
# transformers backend
# --------------------------------------------------------------------------- #
class HfBackend(Backend):
    """Transformers backend with left padding.

    Parameters
    ----------
    model_id:
        Local path or hub id (use ``HF_ENDPOINT=https://hf-mirror.com``).
    device:
        ``cuda`` / ``cpu``.
    dtype:
        ``bfloat16`` / ``float16`` / ``float32``.
    """

    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "bfloat16", max_length: int = 4096):
        import torch  # noqa: F401  (imported lazily on purpose)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = model_id
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # left padding keeps the *last* position a real token for every row
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device if device != "cpu" else None,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self._token_cache: dict[str, int] = {}

    # -- token resolution --------------------------------------------------- #
    def resolve_token(self, text: str) -> int | None:
        """Resolve ``text`` to a single token id, trying common prefix variants."""
        if text in self._token_cache:
            return self._token_cache[text]
        for variant in (text, " " + text, text.strip()):
            ids = self.tokenizer.encode(variant, add_special_tokens=False)
            if len(ids) == 1:
                self._token_cache[text] = ids[0]
                return ids[0]
        # fall back to the first token of the prefixed form (logged by caller)
        ids = self.tokenizer.encode(" " + text, add_special_tokens=False)
        if ids:
            self._token_cache[text] = ids[0]
            return ids[0]
        return None

    # -- forward ------------------------------------------------------------ #
    def score_tokens(self, prompts: Sequence[str], token_texts: Sequence[str]) -> list[dict[str, float]]:
        import torch

        token_ids = [(t, self.resolve_token(t)) for t in token_texts]
        token_ids = [(t, i) for t, i in token_ids if i is not None]
        if not token_ids:
            return [{} for _ in prompts]

        enc = self.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.no_grad():
            out = self.model(**enc)
            last = out.logits[:, -1, :]  # [B, V]
            cols = torch.tensor([i for _, i in token_ids], device=last.device)
            gathered = last.index_select(1, cols)  # [B, T]

        rows = gathered.float().cpu().tolist()
        names = [t for t, _ in token_ids]
        result: list[dict[str, float]] = []
        for row in rows:
            result.append({name: float(val) for name, val in zip(names, row)})
        return result


def load_backend(
    model_id: str | None,
    device: str = "cuda",
    dtype: str = "bfloat16",
    mock: bool = False,
) -> Backend:
    """Factory used by the CLI."""
    if mock or not model_id:
        return MockBackend()
    return HfBackend(model_id, device=device, dtype=dtype)
