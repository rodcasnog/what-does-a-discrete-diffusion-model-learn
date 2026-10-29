"""The three coordinates of one reverse process, and the exact conversions between them.

At a position whose current state is ``z``:

- denoiser ``pi(l)``: the posterior of the clean token ``l`` given everything, ``z``
  included;
- cavity ``mu(l)``: the belief about ``l`` from the other positions alone, which a
  bridge plug-in head learns. Bayes links the two: ``pi(l) ∝ mu(l) q(z | l)``;
- score ``s(z, y) = m(y) / m(z)``: ratios of the cavity's noisy marginal
  ``m = r mu + (1 - r) nu``, which set the reverse jump rates from ``z`` to ``y``.

Bridges consume ``mu``, decoding consumes ``pi`` and CTMC rates consume ``s``.

Noise: the GIDD family, with uniform diffusion at ``lam = 1``. The noise law is
``nu = (1 - lam) delta_mask + lam unif(C)`` and the forward channel
``q(y | l) = r 1{y = l} + (1 - r) nu(y)``. Times are the survival ``r`` and
``t = 1 - r`` (scalars or one per row); the readouts return ``[B, L, C]`` logits.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal, get_args

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from .process import Diffusion

Parametrization = Literal["denoiser", "cavity", "score"]
PARAMETRIZATIONS = get_args(Parametrization)
LOGIT_MASK = -1e9  # finite log-zero for softmax and categorical operations


def _bcast(x, target: jnp.ndarray) -> jnp.ndarray:
    """Add trailing axes so a time scalar broadcasts over ``target``."""
    x = jnp.asarray(x, dtype=jnp.float32)
    return x.reshape(x.shape + (1,) * (target.ndim - x.ndim))


def log_or_mask(x: jnp.ndarray) -> jnp.ndarray:
    """Elementwise log with exact zero as ``LOGIT_MASK``; the inner ``where`` keeps
    both branches' derivatives finite without flooring any positive mass."""
    positive = x > 0.0
    return jnp.where(positive, jnp.log(jnp.where(positive, x, 1.0)), LOGIT_MASK)


def _loglik_shift(ctx: Diffusion, z, r, t) -> jnp.ndarray:
    """``log q(z | l)`` over the clean tokens ``l``, up to an ``l``-independent
    constant. At a content ``z``, ``q(z | l)`` is ``q_other = t lam / C`` for
    ``l != z`` and ``q_other + r`` for ``l = z``, so the shift is
    ``log(1 + r / q_other)`` on ``z``'s own entry. At a mask it is constant: zero."""
    C = ctx.num_classes
    q_other = _bcast(t, z) * ctx.gidd_lambda / C
    log_match = jnp.log1p(_bcast(r, z) / q_other)
    one_hot = jax.nn.one_hot(jnp.clip(z, 0, C - 1), C)
    return jnp.where((z != ctx.mask_id)[..., None], one_hot * log_match[..., None], 0.0)


def gidd_score_logits(ctx: Diffusion, logits, z, r, t) -> jnp.ndarray:
    """Genuine ``log s(z, y)`` of a score head over the destinations
    ``[C content, mask if lam < 1]``.

    The head outputs a residual ``f`` around the score of a uniform cavity:
    ``log s(z, y) = f_y + log m0(y) - log m0(z)``, where ``m0 = r unif(C) + (1 - r) nu``
    is ``(r + t lam) / C`` on every content token and ``t (1 - lam)`` on the mask.
    Under uniform diffusion the shift vanishes. The current state's own entry is
    ``log s(z, z) = 0``.
    """
    C, lam, mask_id = ctx.num_classes, ctx.gidd_lambda, ctx.mask_id
    own = jnp.clip(z, 0, C - 1)  # destination column of the current state
    log_s = logits[..., :C]
    if lam < 1.0:
        own = jnp.where(z == mask_id, C, own)
        r_b, t_b = _bcast(r, z), _bcast(t, z)
        log_m0_content = jnp.log(r_b + t_b * lam) - jnp.log(C)
        log_m0_mask = jnp.log(t_b * (1.0 - lam))
        log_m0_z = jnp.where(z == mask_id, log_m0_mask, log_m0_content)
        to_content = log_s + (log_m0_content - log_m0_z)[..., None]
        to_mask = logits[..., mask_id] + log_m0_mask - log_m0_z
        log_s = jnp.concatenate([to_content, to_mask[..., None]], axis=-1)
    return jnp.where(jax.nn.one_hot(own, log_s.shape[-1], dtype=jnp.bool_), 0.0, log_s)


def cavity_logits(ctx: Diffusion, logits, z, r, t, head: Parametrization | None = None):
    """Context-only belief logits from a ``head`` output (default: the trained
    coordinate). Denoiser -> cavity divides out the local likelihood; score ->
    cavity normalizes the genuine score to ``m_t`` and inverts the forward channel,
    a law exactly on the channel image. A masked head is read raw, so its cavity
    is untrained at visible positions."""
    C, head = ctx.num_classes, head or ctx.parametrization
    if ctx.noise_type == "masked" or head == "cavity":
        return logits[..., :C]
    if head == "denoiser":
        return logits[..., :C] - _loglik_shift(ctx, z, r, t)
    # s(z, .) = m(.) / m(z), so its softmax over the destinations is m; on the content
    # tokens m = r mu + t lam / C, which inverts to mu.
    m = jax.nn.softmax(gidd_score_logits(ctx, logits, z, r, t), axis=-1)
    noise = _bcast(t, m) * (ctx.gidd_lambda / C)
    return log_or_mask((m[..., :C] - noise) / _bcast(r, m))


def denoiser_logits(ctx: Diffusion, logits, z, r, t) -> jnp.ndarray:
    """Clean-token posterior logits from the trained head. Under masked noise a
    visible token is the clean one; at a mask every head's softmax is the denoiser."""
    C = ctx.num_classes
    if ctx.noise_type == "masked":
        point = jnp.where(jax.nn.one_hot(jnp.clip(z, 0, C - 1), C, dtype=jnp.bool_),
                          0.0, LOGIT_MASK)
        return jnp.where((z != ctx.mask_id)[..., None], point, logits[..., :C])
    if ctx.parametrization == "denoiser":
        return logits[..., :C]
    return cavity_logits(ctx, logits, z, r, t) + _loglik_shift(ctx, z, r, t)
