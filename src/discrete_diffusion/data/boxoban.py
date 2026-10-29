"""Boxoban (Sokoban) as plan infilling: ``{split}.npy`` int8 rows ``[board (100) |
moves (128)]`` whose board cells are clean anchors, with the whole move region diffused
and scored. Ids: the moves 0-3, STAY 4, ``BOARD_CHARS`` from 5, then the mask. A plan
solves a level when every move is legal and every box ends on a goal.
"""
from __future__ import annotations

import os

import numpy as np
from omegaconf import DictConfig

from . import StaticTokenizer
from .grid import MOVE_DELTAS, N_MOVES, board_batch, decode_actions

GRID_H = GRID_W = 10
BOARD = GRID_H * GRID_W
N_CONTENT = N_MOVES + 1  # diffused alphabet: moves + STAY

BOARD_CHARS = "# .$@*+"  # wall, floor, goal, box, player, box on goal, player on goal
CHAR_TO_ID = {c: N_CONTENT + i for i, c in enumerate(BOARD_CHARS)}
VOCAB_SIZE = N_CONTENT + len(BOARD_CHARS) + 1
MASK_ID = VOCAB_SIZE - 1


def tokenizer(cfg: DictConfig) -> StaticTokenizer:
    return StaticTokenizer(VOCAB_SIZE, mask_token_id=MASK_ID, num_classes=N_CONTENT)


def load(cfg: DictConfig, split: str) -> dict[str, np.ndarray]:
    return {"seq": np.load(os.path.join(cfg.data.dataset_path, f"{split}.npy"), mmap_mode="r")}


def make_batch(rows: dict, rng: np.random.Generator | None, cfg: DictConfig) -> dict:
    return board_batch(rows["seq"], rng, GRID_H, GRID_W)


def _board_state(seq_row: np.ndarray) -> tuple[np.ndarray, tuple, np.ndarray, np.ndarray]:
    """Decode the board prefix of a row into (walls, player (r, c), boxes, targets)."""
    g = np.asarray(seq_row[:BOARD]).reshape(GRID_H, GRID_W)
    walls = g == CHAR_TO_ID["#"]
    targets = (g == CHAR_TO_ID["."]) | (g == CHAR_TO_ID["*"]) | (g == CHAR_TO_ID["+"])
    boxes = (g == CHAR_TO_ID["$"]) | (g == CHAR_TO_ID["*"])
    pr, pc = np.argwhere((g == CHAR_TO_ID["@"]) | (g == CHAR_TO_ID["+"]))[0]
    return walls, (int(pr), int(pc)), boxes, targets


def solves(seq_row: np.ndarray, moves: np.ndarray) -> bool:
    """Whether ``moves`` (ids 0..3) solve the board of ``seq_row``: a move off the
    board, into a wall or against an unpushable box fails the whole plan."""
    walls, (pr, pc), boxes, targets = _board_state(seq_row)
    for m in moves:
        dr, dc = MOVE_DELTAS[int(m)]
        nr, nc = pr + dr, pc + dc
        if not (0 <= nr < GRID_H and 0 <= nc < GRID_W) or walls[nr, nc]:
            return False
        if boxes[nr, nc]:
            br, bc = nr + dr, nc + dc
            if not (0 <= br < GRID_H and 0 <= bc < GRID_W) or walls[br, bc] or boxes[br, bc]:
                return False
            boxes[nr, nc] = False
            boxes[br, bc] = True
        pr, pc = nr, nc
    return bool(np.all(boxes == (boxes & targets)) and boxes.sum() == targets.sum())


def solved(batch: dict, pred: np.ndarray, cfg: DictConfig) -> np.ndarray:
    return np.array([solves(row, decode_actions(p[BOARD:]))
                     for row, p in zip(batch["input_ids"], pred, strict=True)])
