"""Text as fixed-length records of a pretrained tokenizer: ``{split}.npy``, [N, L]."""
from __future__ import annotations

import os

import numpy as np
from omegaconf import DictConfig


def tokenizer(cfg: DictConfig):
    """The pretrained tokenizer with a mask token appended after its vocabulary."""
    from transformers import AutoTokenizer  # heavy, and only text runs need it

    tok = AutoTokenizer.from_pretrained(cfg.data.tokenizer_name)
    num_classes = len(tok)
    tok.add_special_tokens({"mask_token": f"<|diffusion_mask_{num_classes}|>"})
    tok.num_classes = num_classes
    return tok


def load(cfg: DictConfig, split: str) -> dict[str, np.ndarray]:
    path = os.path.join(cfg.data.dataset_path, f"{split}.npy")
    return {"input_ids": np.load(path, mmap_mode="r")}


def make_batch(rows: dict, rng: np.random.Generator | None, cfg: DictConfig) -> dict:
    return {"input_ids": np.asarray(rows["input_ids"], dtype=np.int32)}
