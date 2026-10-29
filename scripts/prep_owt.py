"""Build OpenWebText as 256-token GPT-2 records: ``{train,val,test}.npy``, uint16 [N, 256].

The source holds the EOS-delimited GPT-2 token stream of OpenWebText cut into
shuffled 1024-token rows; each row becomes four consecutive records. Files 0-71
are the training split. The last file is held out and cut into 64 contiguous
blocks: the even blocks are the validation split and the odd blocks the test split.

    uv run scripts/prep_owt.py --out data/openwebtext-gpt2-len256
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "apollo-research/Skylion007-openwebtext-tokenizer-gpt2"
REVISION = "f02886b54795e8acabceb637ca119f9ae8f19d3f"
NUM_FILES = 73
SOURCE_LEN = 1024
SEQ_LEN = 256
HELD_OUT_BLOCKS = 64
ROWS = {"train": 34_812_860, "val": 241_792, "test": 241_716}


def records(path: str) -> np.ndarray:
    """The rows of one source file, each cut into consecutive ``SEQ_LEN`` records."""
    column = pq.read_table(path, columns=["input_ids"]).column("input_ids").combine_chunks()
    ids = column.flatten().to_numpy()
    return ids.reshape(len(column) * SOURCE_LEN // SEQ_LEN, SEQ_LEN).astype(np.uint16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="output directory")
    out = parser.parse_args().out
    os.makedirs(out, exist_ok=True)
    *train_files, held_out_file = (
        hf_hub_download(REPO, f"data/train-{i:05d}-of-{NUM_FILES:05d}.parquet",
                        repo_type="dataset", revision=REVISION)
        for i in range(NUM_FILES))

    train = np.lib.format.open_memmap(os.path.join(out, "train.npy"), mode="w+",
                                      dtype=np.uint16, shape=(ROWS["train"], SEQ_LEN))
    pos = 0
    for path in train_files:
        part = records(path)
        train[pos:pos + len(part)] = part
        pos += len(part)
    assert pos == ROWS["train"], pos
    train.flush()

    held_out = records(held_out_file)
    # Blocks of whole source rows, the last one shorter.
    per_row = SOURCE_LEN // SEQ_LEN
    block = per_row * -(-len(held_out) // (per_row * HELD_OUT_BLOCKS))
    blocks = [held_out[i:i + block] for i in range(0, len(held_out), block)]
    for split, part in (("val", blocks[0::2]), ("test", blocks[1::2])):
        rows = np.concatenate(part)
        assert len(rows) == ROWS[split], (split, len(rows))
        np.save(os.path.join(out, f"{split}.npy"), rows)


if __name__ == "__main__":
    main()
