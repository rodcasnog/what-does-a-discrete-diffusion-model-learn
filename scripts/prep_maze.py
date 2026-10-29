"""Build 40x40 maze shortest paths: ``{train,val,test}.npy``, int8 [N, 1600 + 192].

Each cell is a wall w.p. 0.38. The start is a uniform floor cell, and the goal is
uniform over the cells at BFS distance 120 to 191 from it. The plan is the
shortest path traced back from the goal, each step to the first neighbour (up,
right, down, left) one step closer to the start. Every split is generated in
shards with their own spawned seeds, concatenated in shard order, so the arrays
do not depend on the number of processes.

    uv run scripts/prep_maze.py --out data/maze-40x40
"""
from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import os

import numpy as np

from discrete_diffusion.data.grid import MOVE_DELTAS, encode_example
from discrete_diffusion.data.maze import CHAR_TO_ID

GRID = 40
MOVE_SLOTS = 192
WALL_P = 0.38
MIN_LEN = 120
SHARDS = 32
SEEDS = {"train": 20260809, "val": 20260811, "test": 20260812}
SIZES = {"train": 500_000, "val": 25_000, "test": 25_000}
MOVE_OF = {tuple(delta): move for move, delta in enumerate(MOVE_DELTAS.tolist())}


def bfs_dist(walls: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """Grid BFS distances from ``start`` via numpy frontier dilation; -1 marks
    unreachable cells (and walls). 4-neighborhood, matching MOVE_DELTAS."""
    dist = np.full(walls.shape, -1, dtype=np.int32)
    frontier = np.zeros(walls.shape, dtype=bool)
    frontier[start] = True
    dist[start] = 0
    d = 0
    while frontier.any():
        d += 1
        grown = np.zeros_like(frontier)
        grown[1:, :] |= frontier[:-1, :]
        grown[:-1, :] |= frontier[1:, :]
        grown[:, 1:] |= frontier[:, :-1]
        grown[:, :-1] |= frontier[:, 1:]
        frontier = grown & ~walls & (dist < 0)
        dist[frontier] = d
    return dist


def instance(rng: np.random.Generator) -> np.ndarray | None:
    """One ``[board | moves]`` row, or None if the draw has no admissible goal."""
    walls = rng.random((GRID, GRID)) < WALL_P
    floor = np.argwhere(~walls)
    s = tuple(floor[rng.integers(len(floor))])
    dist = bfs_dist(walls, s)
    far = np.argwhere((dist >= MIN_LEN) & (dist <= MOVE_SLOTS - 1))
    if len(far) == 0:
        return None
    g = tuple(far[rng.integers(len(far))])
    # backtrack the shortest path from g (any neighbor one step closer to s)
    cells, cur = [g], g
    while cur != s:
        nbs = [(cur[0] + dd[0], cur[1] + dd[1]) for dd in MOVE_DELTAS]
        nb = next(p for p in nbs if 0 <= p[0] < GRID and 0 <= p[1] < GRID
                  and dist[p] == dist[cur] - 1)
        cells.append(nb)
        cur = nb
    cells = cells[::-1]
    moves = np.array([MOVE_OF[(b[0] - a[0], b[1] - a[1])] for a, b in itertools.pairwise(cells)],
                     dtype=np.int8)
    grid = np.full((GRID, GRID), CHAR_TO_ID[" "], dtype=np.int8)
    grid[walls] = CHAR_TO_ID["#"]
    grid[s] = CHAR_TO_ID["S"]
    grid[g] = CHAR_TO_ID["G"]
    return encode_example(grid, moves, MOVE_SLOTS)


def shard(task: tuple[np.random.SeedSequence, int]) -> np.ndarray:
    """``count`` accepted instances from one shard's seed."""
    seed, count = task
    rng = np.random.default_rng(seed)
    rows = []
    while len(rows) < count:
        row = instance(rng)
        if row is not None:
            rows.append(row)
    return np.stack(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output directory")
    out = parser.parse_args().out
    os.makedirs(out, exist_ok=True)
    with mp.Pool() as pool:
        for split, seed in SEEDS.items():
            n = SIZES[split]
            counts = [n // SHARDS + (i < n % SHARDS) for i in range(SHARDS)]
            tasks = zip(np.random.SeedSequence(seed).spawn(SHARDS), counts, strict=True)
            np.save(os.path.join(out, f"{split}.npy"), np.concatenate(pool.map(shard, tasks)))


if __name__ == "__main__":
    main()
