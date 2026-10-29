"""The forward process of one model: its kernel, the per-row noising and the clock,
with survival ``r = 1 - t``."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, get_args

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from omegaconf import DictConfig

from .coords import PARAMETRIZATIONS, Parametrization

NoiseType = Literal["masked", "gidd", "uniform"]
NOISE_TYPES = get_args(NoiseType)
GridSpacing = Literal["uniform", "cosine"]


@dataclass(frozen=True)
class Diffusion:
    """Kernel, trained coordinate and training window ``[t_min, t_max]`` of one model.

    GIDD noises towards ``(1 - gidd_lambda) delta_mask + gidd_lambda unif(C)``;
    uniform diffusion is ``gidd_lambda = 1``, and masked diffusion ignores it (1).
    """

    noise_type: NoiseType
    parametrization: Parametrization
    gidd_lambda: float
    num_classes: int          # content alphabet: only [0, num_classes) diffuses
    mask_id: int              # num_classes: the head's mask logit follows the content logits
    t_min: float
    t_max: float

    def __post_init__(self):
        # from_config reads every kernel but gidd as lambda = 1, so a typo would train uniform.
        assert self.noise_type in NOISE_TYPES and self.parametrization in PARAMETRIZATIONS, self

    @classmethod
    def from_config(cls, cfg: DictConfig, tok) -> Diffusion:
        """From the ``diffusion`` config and the task tokenizer."""
        d = cfg.diffusion
        return cls(
            noise_type=d.noise_type,
            parametrization=d.parametrization,
            gidd_lambda=float(d.gidd_lambda) if d.noise_type == "gidd" else 1.0,
            num_classes=int(tok.num_classes),
            mask_id=int(tok.mask_token_id),
            t_min=float(d.t_min),
            t_max=float(d.t_max),
        )


def split_keys(keys: jnp.ndarray, count: int = 2) -> jnp.ndarray:
    """Split each row key into ``count``, stacked on a new leading axis."""
    return jnp.swapaxes(jax.vmap(lambda key: random.split(key, count))(keys), 0, 1)


def _bernoulli(keys, p, shape):
    p = jnp.asarray(p, jnp.float32)
    p = jnp.broadcast_to(p.reshape(p.shape + (1,) * (len(shape) - p.ndim)), shape)
    return jax.vmap(lambda key, row: random.bernoulli(key, row))(keys, p)


def renoise(ctx: Diffusion, x0: jnp.ndarray, r_s: jnp.ndarray | float, rng: jnp.ndarray,
            cond: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Forward-noise clean tokens ``x0`` to survival ``r_s`` (a scalar, or one level
    per sequence or per position) with one key per row; anchors (``cond``) stay
    clean. Also return the cells the kernel replaced; at ``r_s = 0`` that is every
    free cell, which draws the limiting prior."""
    r_s = jnp.asarray(r_s, jnp.float32)
    rate = 1.0 - (r_s[:, None] if r_s.ndim == 1 else r_s)
    scored = ~cond
    if ctx.noise_type == "masked":
        remask = scored & _bernoulli(rng, rate, x0.shape)
        return jnp.where(remask, ctx.mask_id, x0), remask
    # replace w.p. 1-r_s by a draw from the noise law (mask w.p. 1-lam).
    rep_rng, u_rng, m_rng = split_keys(rng, 3)
    rep = scored & _bernoulli(rep_rng, rate, x0.shape)
    repl = jax.vmap(lambda key: random.randint(key, x0.shape[1:], 0, ctx.num_classes))(u_rng)
    if ctx.gidd_lambda < 1.0:
        to_mask = _bernoulli(m_rng, 1.0 - ctx.gidd_lambda, x0.shape)
        repl = jnp.where(to_mask, ctx.mask_id, repl)
    return jnp.where(rep, repl, x0), rep


def sample_time(rng: jnp.ndarray, batch_size: int, t_min: float, t_max: float) -> jnp.ndarray:
    """Low-discrepancy uniform times on ``[t_min, t_max]``, one per randomly shifted
    equal-width stratum: the shift keeps batch position and noise level unrelated."""
    jitter_rng, shift_rng = random.split(rng)
    u = random.uniform(jitter_rng, (batch_size,), dtype=jnp.float32)
    shift = random.randint(shift_rng, (), 0, batch_size)
    strata = (jnp.arange(batch_size, dtype=jnp.int32) + shift) % batch_size
    u = (strata.astype(jnp.float32) + u) / batch_size
    return t_min + u * (t_max - t_min)


def model_time_cond(t: jnp.ndarray, t_min: float, t_max: float) -> jnp.ndarray:
    """Model input: physical time ``t`` normalized to the training window."""
    t = jnp.asarray(t, jnp.float32)
    return (t - t_min) / (t_max - t_min)


def survival_grid(nfe: int, t_min: float, t_max: float, spacing: GridSpacing) -> np.ndarray:
    """Exactly ``nfe`` ascending survival levels from noisy to clean; ``cosine`` spaces
    them as ``sin^2`` of a uniform grid, which concentrates them at both ends."""
    u = np.linspace(1.0, 0.0, nfe)
    s = {"uniform": u, "cosine": np.sin(0.5 * math.pi * u) ** 2}[spacing]
    r = 1.0 - (t_min + (t_max - t_min) * s)
    r[-1], r[0] = 1.0 - t_min, 1.0 - t_max   # exact ends, and NFE 1 keeps the noisy one
    return r
