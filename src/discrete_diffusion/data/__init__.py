"""Task data as numpy batches, one module per ``data.source``: ``.npy`` row arrays per
split read through memory maps, or seeded generators for graph coloring. A batch holds
``input_ids`` and, per task, ``cond_mask`` (the clean anchors) and ``attention_mask``
(the graph adjacency).
"""
from __future__ import annotations

import importlib
from collections.abc import Iterator
from itertools import count
from types import ModuleType
from typing import Any

import numpy as np
from omegaconf import DictConfig

# Offset that gives each split its own sampling stream off the run seed.
SPLIT_STREAM = {"train": 1, "val": 2, "test": 3}


class StaticTokenizer:
    """The tokenizer attributes the model and the losses read, for a fixed
    vocabulary: ``num_classes`` content ids first, task symbols and the mask after."""

    def __init__(self, vocab_size: int, mask_token_id: int, num_classes: int):
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.vocab_size


def epoch_batches(n: int, batch_size: int, seed: int, start: int
                  ) -> Iterator[tuple[np.ndarray, np.random.Generator]]:
    """Row indices and the batch's own generator, in reshuffled epochs forever. Both
    are seeded from the epoch and batch number, so batch ``k`` is the same whatever
    ran before it and ``start`` seeks to where an interrupted run stopped."""
    pos = start * batch_size
    epoch, perm = -1, None
    for k in count(start):
        idx = np.empty(batch_size, dtype=np.int64)
        filled = 0
        while filled < batch_size:
            e, offset = divmod(pos, n)
            if e != epoch:
                epoch, perm = e, np.random.default_rng([seed, e]).permutation(n)
            take = min(batch_size - filled, n - offset)
            idx[filled:filled + take] = perm[offset:offset + take]
            filled += take
            pos += take
        yield idx, np.random.default_rng([seed, epoch, k])


def _task(cfg: DictConfig) -> ModuleType:
    # Imported on first use: the task modules import this package's helpers.
    return importlib.import_module(f"{__name__}.{cfg.data.source}")


def _rows(arrays: dict[str, np.ndarray], idx: np.ndarray) -> dict[str, np.ndarray]:
    return {name: array[idx] for name, array in arrays.items()}


def create_tokenizer(cfg: DictConfig) -> Any:
    """The tokenizer of ``data.source``."""
    return _task(cfg).tokenizer(cfg)


def train_batches(cfg: DictConfig, start: int) -> Iterator[dict]:
    """Training batches from batch ``start`` on, augmented by the task's ``make_batch``.
    Batch ``k`` depends only on the config, so a resumed run continues the stream."""
    task, batch_size, seed = _task(cfg), cfg.training.global_batch_size, cfg.training.seed
    if cfg.data.source == "graphcolor":
        return (task.instances(cfg, seed, "train", k * batch_size, batch_size)
                for k in count(start))
    arrays = task.load(cfg, "train")
    n = len(next(iter(arrays.values())))
    return (task.make_batch(_rows(arrays, idx), rng, cfg)
            for idx, rng in epoch_batches(n, batch_size, seed + SPLIT_STREAM["train"], start))


def eval_batches(cfg: DictConfig, split: str, batch_size: int, n: int, seed: int
                 ) -> Iterator[dict]:
    """``n`` unaugmented instances of ``split`` in batches: the first ``n`` text
    records in stored order, a ``seed``-drawn subset of a logic split (cycled past
    its size), or a prefix of the ``seed`` graph stream."""
    if n % batch_size:
        raise ValueError(f"n={n} is not a multiple of batch_size={batch_size}")
    task, starts = _task(cfg), range(0, n, batch_size)
    if cfg.data.source == "graphcolor":
        return (task.instances(cfg, seed, split, i, batch_size) for i in starts)
    arrays = task.load(cfg, split)
    size = len(next(iter(arrays.values())))
    idx = (np.arange(n) if cfg.data.source == "text"
           else np.resize(np.random.default_rng(seed).permutation(size), n))
    return (task.make_batch(_rows(arrays, idx[i:i + batch_size]), None, cfg) for i in starts)


def solved(cfg: DictConfig, batch: dict, pred: np.ndarray) -> np.ndarray:
    """Per instance of a logic batch, whether ``pred`` solves it."""
    return _task(cfg).solved(batch, np.asarray(pred), cfg)
