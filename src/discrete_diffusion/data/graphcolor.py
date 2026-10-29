"""Graph k-coloring with a planted solution: n vertices split into k equal color
classes, and each cross-class edge present independently w.p. p. The vertex colors
diffuse, and the adjacency travels as the ``[B, n, n]`` ``attention_mask``. A
prediction solves an instance when no edge joins two vertices of one color.
"""
from __future__ import annotations

import numpy as np
from omegaconf import DictConfig

from . import SPLIT_STREAM, StaticTokenizer

CHUNK = 1024  # instances per generator draw, so a stream does not depend on the batch size


def tokenizer(cfg: DictConfig) -> StaticTokenizer:
    k = cfg.data.graphcolor.k
    return StaticTokenizer(k + 1, mask_token_id=k, num_classes=k)


def generate_instances(
    rng: np.random.Generator, batch: int, n: int, k: int, p: float
) -> tuple[np.ndarray, np.ndarray]:
    """``batch`` planted instances (k | n): colors [B, n] and edges [B, n(n-1)/2] in
    ``np.triu_indices(n, 1)`` order."""
    base = np.repeat(np.arange(k), n // k)
    perm = np.argsort(rng.random((batch, n)), axis=1)
    colors = base[perm]                                            # [B, n]
    iu, ju = np.triu_indices(n, 1)
    cross = colors[:, iu] != colors[:, ju]                         # [B, E]
    edges = cross & (rng.random((batch, iu.size)) < p)
    return colors, edges


def instances(cfg: DictConfig, seed: int, split: str, start: int, count: int) -> dict:
    """Instances ``[start, start + count)`` of the stream of ``(seed, split)``; the key
    holds them apart, so no seed's training stream replays another's evaluation graphs."""
    g = cfg.data.graphcolor
    first = start // CHUNK
    chunks = [generate_instances(np.random.default_rng([seed, SPLIT_STREAM[split], c]),
                                 CHUNK, g.n, g.k, g.p)
              for c in range(first, -(-(start + count) // CHUNK))]
    window = slice(start - first * CHUNK, start - first * CHUNK + count)
    colors = np.concatenate([c for c, _ in chunks])[window]
    edges = np.concatenate([e for _, e in chunks])[window]
    iu, ju = np.triu_indices(g.n, 1)
    adj = np.zeros((count, g.n, g.n), dtype=bool)
    adj[:, iu, ju] = edges
    adj[:, ju, iu] = edges
    return {"input_ids": colors.astype(np.int32),
            "cond_mask": np.zeros(colors.shape, dtype=bool),
            "attention_mask": adj}


def solved(batch: dict, pred: np.ndarray, cfg: DictConfig) -> np.ndarray:
    adj = batch["attention_mask"]
    return ~(adj & (pred[:, :, None] == pred[:, None, :])).any((1, 2))
