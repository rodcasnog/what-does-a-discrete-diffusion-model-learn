"""Maze shortest paths as plan infilling: ``{split}.npy`` int8 rows ``[board (H*W) |
moves]`` whose board cells are clean anchors, with the whole move region diffused and
scored; the geometry is ``data.maze``. Ids: the moves 0-3, STAY 4, ``BOARD_CHARS`` from
5, then the mask. A plan solves a maze when its walk from S stays on the floor until it
reaches G; moves after G are ignored.
"""
from __future__ import annotations

import os

import numpy as np
from omegaconf import DictConfig

from . import StaticTokenizer
from .grid import MOVE_DELTAS, N_MOVES, board_batch, decode_actions

N_CONTENT = N_MOVES + 1  # diffused alphabet: moves + STAY

BOARD_CHARS = "# SG"  # wall, floor, start, goal
CHAR_TO_ID = {c: N_CONTENT + i for i, c in enumerate(BOARD_CHARS)}
VOCAB_SIZE = N_CONTENT + len(BOARD_CHARS) + 1
MASK_ID = VOCAB_SIZE - 1


def tokenizer(cfg: DictConfig) -> StaticTokenizer:
    return StaticTokenizer(VOCAB_SIZE, mask_token_id=MASK_ID, num_classes=N_CONTENT)


def load(cfg: DictConfig, split: str) -> dict[str, np.ndarray]:
    return {"seq": np.load(os.path.join(cfg.data.dataset_path, f"{split}.npy"), mmap_mode="r")}


def make_batch(rows: dict, rng: np.random.Generator | None, cfg: DictConfig) -> dict:
    m = cfg.data.maze
    return board_batch(rows["seq"], rng, m.grid_height, m.grid_width)


def _board_state(seq_row: np.ndarray, height: int, width: int
                 ) -> tuple[np.ndarray, tuple, tuple]:
    """Decode the board prefix into (walls, start (r, c), goal (r, c))."""
    g = np.asarray(seq_row[:height * width]).reshape(height, width)
    walls = g == CHAR_TO_ID["#"]
    sr, sc = np.argwhere(g == CHAR_TO_ID["S"])[0]
    gr, gc = np.argwhere(g == CHAR_TO_ID["G"])[0]
    return walls, (int(sr), int(sc)), (int(gr), int(gc))


def solves(seq_row: np.ndarray, moves: np.ndarray, height: int, width: int) -> bool:
    """Whether the walk of ``moves`` (ids 0..3) from S reaches G on the board of
    ``seq_row``; a step off the board or into a wall fails the whole plan."""
    walls, (r, c), goal = _board_state(seq_row, height, width)
    for m in moves:
        dr, dc = MOVE_DELTAS[int(m)]
        r, c = r + dr, c + dc
        if not (0 <= r < height and 0 <= c < width) or walls[r, c]:
            return False
        if (r, c) == goal:
            return True
    return False


def solved(batch: dict, pred: np.ndarray, cfg: DictConfig) -> np.ndarray:
    h, w = cfg.data.maze.grid_height, cfg.data.maze.grid_width
    return np.array([solves(row, decode_actions(p[h * w:]), h, w)
                     for row, p in zip(batch["input_ids"], pred, strict=True)])
