"""Build Sudoku-Extreme: ``{train,val,test}_sol.npy`` (the solution, digits 1-9 as
0-8) and ``{train,val,test}_clue.npy`` (the puzzle's given cells), int8 [N, 81].

The validation and test splits are disjoint subsets of the upstream test split,
selected by published index files whose sha256 is checked.

    uv run scripts/prep_sudoku.py --out data/sudoku-extreme
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os

import numpy as np
from huggingface_hub import hf_hub_download

REPO = "sapientinc/sudoku-extreme"
REVISION = "58942f96baeb572ca3127e2a9e9c70f330783d6b"
SPLIT_REPO = "rodcasnog/sudoku-extreme-split"
SPLIT_SHA256 = {
    "val": "06414bf101f9d34a93b8656738272579fd02839766930a4448a0f98a2033f44d",
    "test": "dbff47578f4b65ff4dcbf6f7d084922109d8732d73a425fb3562409015ebb762",
}


def read(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Solutions and given-cell masks of one upstream CSV, whose ``question`` is the
    81-character puzzle (``.`` for a blank) and ``answer`` the solution."""
    path = hf_hub_download(REPO, f"{split}.csv", repo_type="dataset", revision=REVISION)
    sols, clues = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            sols.append([int(c) - 1 for c in row["answer"]])    # 1-9 -> 0-8
            clues.append([c != "." for c in row["question"]])   # given-cell mask
    return np.array(sols, np.int8), np.array(clues, np.int8)


def save(out: str, split: str, sol: np.ndarray, clue: np.ndarray) -> None:
    np.save(os.path.join(out, f"{split}_sol.npy"), sol)
    np.save(os.path.join(out, f"{split}_clue.npy"), clue)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output directory")
    out = parser.parse_args().out
    os.makedirs(out, exist_ok=True)
    save(out, "train", *read("train"))
    sol, clue = read("test")
    for split, sha256 in SPLIT_SHA256.items():
        path = hf_hub_download(SPLIT_REPO, f"{split}_index.npy", repo_type="dataset")
        with open(path, "rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != sha256:
                raise ValueError(f"{path} is not the published {split} index file")
        idx = np.load(path)
        save(out, split, sol[idx], clue[idx])


if __name__ == "__main__":
    main()
