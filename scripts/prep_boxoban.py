"""Build Boxoban with A* plans: ``{train,val,test}.npy``, int8 [N, 100 + 128].

Joins the unfiltered levels of ``google-deepmind/boxoban-levels`` with the A*
solutions of ``AlignmentResearch/boxoban-astar-solutions`` on (file, level) and
keeps the plans that fit the move region and solve their level. The official
test split is small, so validation and test are carved from the official
validation split, leaving out its boards that occur in the official test split;
the rest of the validation split joins the training levels.

    uv run scripts/prep_boxoban.py --out data/boxoban
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
import re
import tarfile
import urllib.request

import numpy as np
from huggingface_hub import hf_hub_download

from discrete_diffusion.data.boxoban import BOARD, CHAR_TO_ID, solves
from discrete_diffusion.data.grid import encode_example

LEVELS_URL = ("https://github.com/google-deepmind/boxoban-levels/archive/"
              "78011d8c86ad423bfb2e1005012393b4c3c24409.tar.gz")
SOLUTIONS_REPO = "AlignmentResearch/boxoban-astar-solutions"
SOLUTIONS_REVISION = "4b30b9c3eda5e937dc16b9f6e58c8a27952aa6cd"
MOVE_SLOTS = 128
CARVE_SEED = 0
EVAL_SIZE = 25_000
TRAIN_SIZE = 936_408


def read_levels(archive: tarfile.TarFile, split: str) -> dict[tuple[int, int], np.ndarray]:
    """``(file, level) -> board`` of one unfiltered split; each ``NNN.txt`` holds
    levels headed ``; <level>``."""
    boards = {}
    for member in archive.getmembers():
        if f"/unfiltered/{split}/" not in member.name or not member.name.endswith(".txt"):
            continue
        file_idx = int(os.path.basename(member.name)[:-4])
        for level in archive.extractfile(member).read().decode().split(";")[1:]:
            header, *rows = level.split("\n")
            boards[file_idx, int(header)] = np.array(
                [[CHAR_TO_ID[c] for c in row] for row in rows if row], np.int8)
    return boards


def build(split: str, boards: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    """The verified ``[board | moves]`` rows of one split, in solution-file order."""
    path = hf_hub_download(SOLUTIONS_REPO, f"unfiltered_{split}.csv.gz",
                           repo_type="dataset", revision=SOLUTIONS_REVISION)
    rows = []
    with gzip.open(path, "rt", newline="") as f:
        for plan in csv.DictReader(f):
            # Levels A* did not solve carry a status marker instead of moves.
            actions = plan["Actions"] or ""
            if not re.fullmatch("[0-3]+", actions):
                continue
            board = boards.get((int(plan["File"]), int(plan["Level"])))
            if board is None:
                continue
            moves = np.frombuffer(actions.encode(), dtype=np.uint8) - ord("0")
            seq = encode_example(board, moves, MOVE_SLOTS)
            # Some public plans do not solve the level they are filed under.
            if seq is not None and solves(seq, moves):
                rows.append(seq)
    return np.stack(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output directory")
    out = parser.parse_args().out
    os.makedirs(out, exist_ok=True)
    # Decompressed up front, so reading members in any order needs no gzip seeks.
    with urllib.request.urlopen(LEVELS_URL) as response:
        levels = io.BytesIO(gzip.decompress(response.read()))
    with tarfile.open(fileobj=levels) as archive:
        train, valid, official_test = (build(split, read_levels(archive, split))
                                       for split in ("train", "valid", "test"))

    official = {row[:BOARD].tobytes() for row in official_test}
    keep = np.array([row[:BOARD].tobytes() not in official for row in valid])
    order = np.random.default_rng(CARVE_SEED).permutation(len(valid))
    order = order[keep[order]]
    splits = {
        "test": valid[np.sort(order[:EVAL_SIZE])],
        "val": valid[np.sort(order[EVAL_SIZE:2 * EVAL_SIZE])],
        "train": np.concatenate([train, valid[np.sort(order[2 * EVAL_SIZE:])]]),
    }
    assert len(splits["train"]) == TRAIN_SIZE, len(splits["train"])
    for split, rows in splits.items():
        np.save(os.path.join(out, f"{split}.npy"), rows)


if __name__ == "__main__":
    main()
