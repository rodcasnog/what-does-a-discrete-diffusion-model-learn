"""Generative perplexity, word entropy and MAUVE of text samples:
``python -m discrete_diffusion.eval.text samples=<sample output dir>/samples.json``.

One float32 judge scores the samples. GenPPL is the exponentiated judge NLL pooled
over tokens, the first token of each text unscored. MAUVE compares the samples
with as many test records of the sampled model's dataset, decoded the same way,
on the judge's last hidden state at each text's final token, averaged over
quantization seeds. Whitespace-only texts are dropped on both sides.
``reference=true`` scores those test records instead.
"""
from __future__ import annotations

import json
import math

import hydra
import mauve
import numpy as np
import torch
from mauve.utils import featurize_tokens_from_model
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..checkpoints import load_config
from ..data import create_tokenizer, eval_batches
from . import write_result


@torch.no_grad()
def gen_nll(texts: list[str], judge, tokenizer, max_length: int, batch_size: int) -> float:
    """Judge NLL per token, pooled over ``texts``."""
    nll, count = 0.0, 0
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], return_tensors="pt", padding=True,
                          truncation=True, max_length=max_length).to(judge.device)
        logits = judge(batch.input_ids, attention_mask=batch.attention_mask).logits[:, :-1]
        target, keep = batch.input_ids[:, 1:], batch.attention_mask[:, 1:].bool()
        token_nll = torch.logsumexp(logits, -1) - logits.gather(-1, target[..., None])[..., 0]
        nll += float((token_nll * keep).sum())
        count += int(keep.sum())
    return nll / count


def word_entropy(texts: list[str]) -> float:
    """Mean over texts of the entropy of each text's whitespace-word histogram."""
    def entropy(words: list[str]) -> float:
        _, counts = np.unique(words, return_counts=True)
        p = counts / counts.sum()
        return float(-(p * np.log(p)).sum())
    return float(np.mean([entropy(text.split()) for text in texts]))


def mauve_score(texts: list[str], refs: list[str], judge, tokenizer, max_length: int,
                batch_size: int, seeds) -> float:
    """Mean MAUVE over the quantization ``seeds``."""
    refs_features, texts_features = (featurize_tokens_from_model(
        judge.base_model,
        [tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length)
         for text in side], batch_size) for side in (refs, texts))
    scores = [mauve.compute_mauve(p_features=refs_features, q_features=texts_features,
                                  num_buckets="auto", seed=seed).mauve for seed in seeds]
    return float(np.mean(scores))


@hydra.main(config_path="../../../configs/eval", config_name="text", version_base="1.3")
def main(cfg: DictConfig) -> None:
    with open(cfg.samples) as f:
        sampled = json.load(f)
    texts = [text for text in sampled["samples"] if text.strip()]
    run_cfg = load_config(sampled["checkpoint_dir"])
    tok = create_tokenizer(run_cfg)
    # Text evaluation batches are the first records in stored order; no seed enters.
    ids = next(eval_batches(run_cfg, cfg.split, len(texts), len(texts), 0))["input_ids"]
    refs = [ref for ref in (tok.decode(row, skip_special_tokens=True) for row in ids.tolist())
            if ref.strip()]

    tokenizer = AutoTokenizer.from_pretrained(cfg.judge, revision=cfg.judge_revision)
    tokenizer.pad_token = tokenizer.eos_token      # padded positions are masked and unscored
    judge = AutoModelForCausalLM.from_pretrained(
        cfg.judge, revision=cfg.judge_revision, dtype=torch.float32)
    judge = judge.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    scored = refs if cfg.reference else texts
    result = {"gen_ppl": math.exp(gen_nll(scored, judge, tokenizer, cfg.max_length,
                                          cfg.batch_size)),
              "word_entropy": word_entropy(scored)}
    if not cfg.reference:
        result["mauve"] = mauve_score(
            texts, refs, judge, tokenizer, cfg.max_length, cfg.batch_size, cfg.mauve_seeds)
    write_result(result)


if __name__ == "__main__":
    main()
