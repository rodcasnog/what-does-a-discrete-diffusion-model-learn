"""Ancestral and re-corrupt sampling on a survival grid.

Text samples: ``python -m discrete_diffusion.sample checkpoint_dir=<run>/checkpoints/<step>``
writes ``samples.json`` to the Hydra output directory.
"""
from __future__ import annotations

import itertools
import json
import os
from functools import partial

import flax.nnx as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from hydra.core.hydra_config import HydraConfig
from jax import random
from omegaconf import DictConfig

from .checkpoints import load_checkpoint
from .coords import cavity_logits, denoiser_logits, log_or_mask
from .process import (
    Diffusion,
    _bernoulli,
    model_time_cond,
    renoise,
    split_keys,
    survival_grid,
)


def _categorical(keys, logits):
    # One key per row; high-dynamic-range Gumbels avoid implicit sharpening of small
    # probabilities.
    return jax.vmap(lambda key, row: random.categorical(key, row, mode="high"))(
        keys, logits)


def decode_tokens(rng: jnp.ndarray, logits: jnp.ndarray, temperature: float) -> jnp.ndarray:
    return _categorical(rng, logits / temperature)


@partial(nn.jit, static_argnames=("ctx",))
def predict(
    model: nn.Module,
    z: jnp.ndarray,
    r: jnp.ndarray,
    cond_mask: jnp.ndarray,
    attention_mask: jnp.ndarray | None,
    board: tuple[jnp.ndarray, ...] | None,
    ctx: Diffusion,
) -> jnp.ndarray:
    """Compiled model forward at survival ``r``, read as denoiser logits."""
    t = 1.0 - r
    sigma = jnp.broadcast_to(model_time_cond(t, ctx.t_min, ctx.t_max), (z.shape[0],))
    extra = {} if board is None else {"board": board}
    logits = model(z, sigma, attention_mask=attention_mask, cond_mask=cond_mask, **extra)
    return denoiser_logits(ctx, logits, z, r, t)


def _masked_step(ctx, temperature, dl, x, seq, cond, r_t, r_s, rng):
    """Unmask each masked position w.p. ``(r_s - r_t) / (1 - r_t)`` from the denoiser."""
    u_rng, t_rng = split_keys(rng)
    p_un = jnp.clip((r_s - r_t) / (1.0 - r_t), 0.0, 1.0)
    do_un = (x == ctx.mask_id) & _bernoulli(u_rng, p_un, x.shape)
    x = jnp.where(do_un, decode_tokens(t_rng, dl, temperature), x)
    return jnp.where(cond, seq, x)


def _gidd_step(ctx, temperature, dl, z, seq, cond, r_t, r_s, rng):
    """Exact finite bridge for the GIDD family: ``z_s = y`` with probability
    proportional to ``q(z_t | y) m_s(y)``, where ``m_s = r_s mu + (1 - r_s) nu`` is the
    cavity's noisy marginal at ``r_s``, and the channel from ``s`` to ``t`` keeps a
    state w.p. ``r_t / r_s`` and otherwise redraws it from ``nu``."""
    C, lam = ctx.num_classes, ctx.gidd_lambda
    has_mask = lam < 1.0
    # Temperature shapes the denoiser; the bridge consumes its cavity.
    mu = jax.nn.softmax(cavity_logits(ctx, dl / temperature, z, r_t,
                                      1.0 - r_t, head="denoiser"), axis=-1)
    r_ts = r_t / r_s
    is_mask_z = (z == ctx.mask_id) if has_mask else jnp.zeros_like(z, bool)
    nu_zt = jnp.where(is_mask_z, 1.0 - lam, lam / C)
    onehot_zt = jax.nn.one_hot(jnp.clip(z, 0, C - 1), C) * (~is_mask_z[..., None])
    # Over the content candidates y: q(z_t | y) and m_s(y).
    trans = r_ts * onehot_zt + (1.0 - r_ts) * nu_zt[..., None]
    m_s = r_s * mu + (1.0 - r_s) * lam / C
    post = trans * m_s
    if has_mask:
        # The mask as a candidate, appended as column C.
        trans_mask = r_ts * is_mask_z + (1.0 - r_ts) * nu_zt
        m_s_mask = (1.0 - r_s) * (1.0 - lam)
        post = jnp.concatenate([post, (trans_mask * m_s_mask)[..., None]], axis=-1)
    idx = _categorical(rng, log_or_mask(post))
    z_s = jnp.where(idx == C, ctx.mask_id, idx) if has_mask else idx
    return jnp.where(cond, seq, z_s)


def _recorrupt_step(ctx, temperature, dl, z, seq, cond, r_t, r_s, rng):
    """Decode clean data, then forward-noise it to the next survival level."""
    p_rng, n_rng = split_keys(rng)
    x0 = jnp.where(cond, seq, decode_tokens(p_rng, dl, temperature))
    z, _ = renoise(ctx, x0, r_s, n_rng, cond)
    return z


@partial(nn.jit, static_argnames=("ctx", "temperature", "step"))
def _level(model, z, board, seq, cond, attention_mask, r_t, r_s, rng, *, ctx, temperature, step):
    """One reverse level from survival ``r_t`` to ``r_s``."""
    dl = predict(model, z, r_t, cond, attention_mask, board, ctx=ctx)
    return step(ctx, temperature, dl, z, seq, cond, r_t, r_s, rng)


@nn.jit
def _encode_board(model, seq):
    return model.encode_board(seq)


def sample(
    sampler: str,
    model: nn.Module,
    ctx: Diffusion,
    seq: jnp.ndarray,
    cond_mask: jnp.ndarray,
    attention_mask: jnp.ndarray | None,
    keys: jnp.ndarray,
    grid: np.ndarray,
    temperature: float,
) -> jnp.ndarray:
    """Sample one sequence per row of ``keys`` on the ascending survival ``grid``.

    The sweep starts from the kernel's limiting prior and takes one level per grid
    step; the forward at the last level is decoded directly, and a masked kernel
    keeps the tokens it already revealed. The anchors (``cond_mask``) are clamped
    to ``seq`` throughout; a maze board is encoded once.
    """
    temperature = float(temperature)
    step = {"ancestral": _masked_step if ctx.noise_type == "masked" else _gidd_step,
            "re_corrupt": _recorrupt_step}[sampler]
    grid = np.asarray(grid, dtype=np.float32)
    board = _encode_board(model, seq) if model.config.architecture == "maze_dit" else None
    keys, init_keys = split_keys(keys)
    z, _ = renoise(ctx, seq, 0.0, init_keys, cond_mask)
    for r_t, r_s in itertools.pairwise(grid):
        keys, step_keys = split_keys(keys)
        z = _level(model, z, board, seq, cond_mask, attention_mask, r_t, r_s, step_keys,
                   ctx=ctx, temperature=temperature, step=step)
    dl = predict(model, z, grid[-1], cond_mask, attention_mask, board, ctx=ctx)
    pred = decode_tokens(keys, dl, temperature)
    if ctx.noise_type == "masked":
        pred = jnp.where(z == ctx.mask_id, pred, z)
    return jnp.where(cond_mask, seq, pred)


@hydra.main(config_path="../../configs", config_name="sample", version_base="1.3")
def main(cfg: DictConfig) -> None:
    model, run_cfg, tok, ctx, _ = load_checkpoint(cfg.checkpoint_dir, cfg.read_as,
                                                  cfg.compute_dtype)
    if run_cfg.data.source != "text":
        raise ValueError("this command samples text; the logic tasks are sampled and "
                         "verified by discrete_diffusion.eval.solve")
    grid = survival_grid(cfg.nfe, ctx.t_min, ctx.t_max, cfg.spacing)
    keys = random.split(random.PRNGKey(cfg.seed), cfg.num_samples)
    chunks = []
    for start in range(0, cfg.num_samples, cfg.batch_size):
        batch_keys = keys[start:start + cfg.batch_size]
        seq = jnp.zeros((len(batch_keys), run_cfg.data.max_seq_len), jnp.int32)
        chunks.append(np.asarray(sample(cfg.sampler, model, ctx, seq, jnp.zeros_like(seq, bool),
                                        None, batch_keys, grid, cfg.temperature)))
        print(f"[sample] {start + len(batch_keys)}/{cfg.num_samples}", flush=True)
    tokens = np.concatenate(chunks)

    out_dir = HydraConfig.get().runtime.output_dir
    with open(os.path.join(out_dir, "samples.json"), "w") as f:
        json.dump({"samples": [tok.decode(row, skip_special_tokens=True) for row in tokens.tolist()],
                   "checkpoint_dir": os.path.abspath(cfg.checkpoint_dir)}, f, indent=1)
    print(f"Wrote {len(tokens)} samples to {out_dir}")


if __name__ == "__main__":
    main()
