"""Sudoku as solution infilling: the 81 solution cells diffuse, and the puzzle's given
cells are clean anchors. Arrays ``{split}_sol.npy`` (digits 1-9 as 0-8) and
``{split}_clue.npy`` (given cells), int8 [N, 81]. The solutions are unique, so a
prediction solves a puzzle exactly when it matches the solution.
"""
from __future__ import annotations

import os

import numpy as np
from omegaconf import DictConfig

from . import StaticTokenizer

N_DIGITS = 9


def tokenizer(cfg: DictConfig) -> StaticTokenizer:
    return StaticTokenizer(N_DIGITS + 1, mask_token_id=N_DIGITS, num_classes=N_DIGITS)


def load(cfg: DictConfig, split: str) -> dict[str, np.ndarray]:
    return {name: np.load(os.path.join(cfg.data.dataset_path, f"{split}_{name}.npy"),
                          mmap_mode="r")
            for name in ("sol", "clue")}


def augment_sudoku(sol: np.ndarray, clue: np.ndarray, rng: np.random.Generator
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Validity-preserving augmentation, drawn independently per sample: digit
    relabeling, row-band + within-band row permutations, column-stack +
    within-stack column permutations, and transpose. ``sol``/``clue`` are
    [B, 81]; positions permute jointly, digit relabeling touches values only."""
    B = sol.shape[0]
    sol9 = sol.reshape(B, 9, 9)
    clue9 = clue.reshape(B, 9, 9)
    bands = np.argsort(rng.random((B, 3)), axis=1)
    rows = (3 * bands[:, :, None] + np.argsort(rng.random((B, 3, 3)), axis=2)
            ).reshape(B, 9)
    stacks = np.argsort(rng.random((B, 3)), axis=1)
    cols = (3 * stacks[:, :, None] + np.argsort(rng.random((B, 3, 3)), axis=2)
            ).reshape(B, 9)
    b = np.arange(B)[:, None, None]
    s9 = sol9[b, rows[:, :, None], cols[:, None, :]]
    c9 = clue9[b, rows[:, :, None], cols[:, None, :]]
    t = (rng.integers(2, size=B) == 1)[:, None, None]
    s9 = np.where(t, s9.transpose(0, 2, 1), s9)
    c9 = np.where(t, c9.transpose(0, 2, 1), c9)
    relabel = np.argsort(rng.random((B, 9)), axis=1).astype(sol.dtype)
    s_out = np.take_along_axis(relabel, s9.reshape(B, 81), axis=1)
    return s_out, c9.reshape(B, 81)


def make_batch(rows: dict, rng: np.random.Generator | None, cfg: DictConfig) -> dict:
    sol, clue = rows["sol"], rows["clue"]
    if rng is not None:
        sol, clue = augment_sudoku(sol, clue, rng)
    return {"input_ids": np.asarray(sol, dtype=np.int32),
            "cond_mask": np.asarray(clue, dtype=bool)}


def solved(batch: dict, pred: np.ndarray, cfg: DictConfig) -> np.ndarray:
    return (pred == batch["input_ids"]).all(1)
