"""Move alphabet and D4 augmentation of the ``[board | moves]`` tasks (Boxoban, maze):
a square board of anchor cells, then a scored move region of up/right/down/left and a
STAY filler. The board carries no directional symbol, so each of the eight rotations
and reflections maps a solvable instance to a solvable one once the moves are remapped.
"""
from __future__ import annotations

import numpy as np

N_MOVES = 4  # up/right/down/left = 0/1/2/3
STAY_ID = 4  # "no more moves" -- scored content, never padding

# move id -> (d_row, d_col); row 0 is the top of the board, so up decreases row.
MOVE_DELTAS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int32)


def _move_perms() -> np.ndarray:
    """Move-id remap per D4 element (k rotations x optional fliplr), derived from
    MOVE_DELTAS; STAY maps to itself."""
    perms = np.empty((8, STAY_ID + 1), dtype=np.int8)
    for e in range(8):
        k, f = e % 4, e >= 4
        for m in range(N_MOVES):
            dr, dc = MOVE_DELTAS[m]
            if f:                       # fliplr: (r, c) -> (r, W-1-c)
                dc = -dc
            for _ in range(k):          # np.rot90 CCW: (r, c) -> (W-1-c, r)
                dr, dc = -dc, dr
            perms[e, m] = int(np.argwhere((MOVE_DELTAS == (dr, dc)).all(1))[0, 0])
        perms[e, STAY_ID] = STAY_ID
    return perms


_PERMS = _move_perms()


def augment_d4(seq: np.ndarray, rng: np.random.Generator,
               height: int, width: int) -> np.ndarray:
    """An independent random D4 transform of each ``[board | moves]`` row, its moves
    remapped to match the board."""
    board = height * width
    seq = np.array(seq)
    elems = rng.integers(0, 8, size=seq.shape[0])
    boards = seq[:, :board].reshape(-1, height, width)
    moves = seq[:, board:]
    for e in range(8):
        m = elems == e
        if not m.any():
            continue
        k, f = e % 4, e >= 4
        b = boards[m]
        if f:
            b = b[:, :, ::-1]
        if k:
            b = np.rot90(b, k, axes=(1, 2))
        boards[m] = b
        moves[m] = _PERMS[e][moves[m]]
    seq[:, :board] = boards.reshape(-1, board)
    seq[:, board:] = moves
    return seq


def board_batch(seq: np.ndarray, rng: np.random.Generator | None,
                height: int, width: int) -> dict:
    """A batch of ``[board | moves]`` rows with the board cells as anchors, each
    row D4-transformed when ``rng`` is given."""
    seq = np.asarray(seq, dtype=np.int32)
    if rng is not None:
        seq = augment_d4(seq, rng, height, width)
    cond = np.zeros(seq.shape, dtype=bool)
    cond[:, :height * width] = True
    return {"input_ids": seq, "cond_mask": cond}


def encode_example(board_grid: np.ndarray, moves: np.ndarray, max_move_len: int
                   ) -> np.ndarray | None:
    """The ``[board | moves]`` row of an [H, W] grid of board-symbol ids and its moves
    (ids 0-3), STAY after the last; ``None`` unless a final STAY fits."""
    board = board_grid.size
    s = len(moves)
    if s + 1 > max_move_len:
        return None
    seq = np.full(board + max_move_len, STAY_ID, dtype=np.int8)
    seq[:board] = board_grid.reshape(-1)
    seq[board:board + s] = np.asarray(moves, dtype=np.int8)
    return seq


def decode_actions(pred_action_row: np.ndarray) -> np.ndarray:
    """Move-region prediction [A] (ids 0..4) -> move list up to the first STAY."""
    pred = np.asarray(pred_action_row)
    end = np.argmax(pred == STAY_ID) if np.any(pred == STAY_ID) else len(pred)
    return pred[:end]
