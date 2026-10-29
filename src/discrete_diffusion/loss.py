"""The negative ELBO of a CTMC diffusion: path term, reconstruction and prior KL, in
nats per scored token.

The path term integrates in corruption time ``t = 1 - r`` over ``[t_min, t_max]``. It
is estimated with one stratified uniform ``t`` per sequence, so each integrand sample
is weighted by the window length ``t_max - t_min``. ``denoiser`` and ``cavity`` heads
are softmax laws; a ``score`` head reads ``exp(logits)`` as the rate ratio with
kernel-known prefactors folded out (coords.py).
"""
from __future__ import annotations

import math

import flax.nnx as nn
import jax
import jax.numpy as jnp
import optax
from jax import random
from jax.scipy.special import xlogy

from .coords import denoiser_logits, gidd_score_logits
from .process import Diffusion, model_time_cond, renoise, sample_time

DECODE_SMOOTHING = 1e-8


def _phi(true_ratio: jnp.ndarray, model_ratio: jnp.ndarray,
         log_model: jnp.ndarray | None) -> jnp.ndarray:
    """Local rate divergence Phi(a, b) = b - a + a·(log a - log b), elementwise.
    ``log_model`` may be passed when it is available exactly (score heads:
    log exp(logits) = logits), avoiding a redundant log."""
    if log_model is None:
        return (model_ratio - true_ratio
                + xlogy(true_ratio, true_ratio)
                - xlogy(true_ratio, model_ratio))
    return (model_ratio - true_ratio + xlogy(true_ratio, true_ratio)
            - true_ratio * log_model)


def _scored(cond_mask: jnp.ndarray) -> jnp.ndarray:
    """Positions the NELBO scores: all but the clean anchors."""
    return (~cond_mask).astype(jnp.float32)


def _scored_mean(per_token: jnp.ndarray, scored: jnp.ndarray) -> jnp.ndarray:
    """Mean of ``per_token`` over the scored positions (a batch with none gives 0)."""
    return jnp.sum(per_token * scored) / jnp.sum(scored).clip(1.0)


def _corrupt(ctx: Diffusion, tokens: jnp.ndarray, rng: jnp.ndarray, cond_mask: jnp.ndarray):
    """Proposal times ``t``, the tokens forward-corrupted with per-row keys, the cells
    the kernel replaced, and the model's time conditioning."""
    noise_rng, corrupt_rng = random.split(rng)
    t = sample_time(noise_rng, tokens.shape[0], ctx.t_min, ctx.t_max)
    z, changed = renoise(ctx, tokens, 1.0 - t, random.split(corrupt_rng, tokens.shape[0]),
                         cond_mask)
    return t, z, changed, model_time_cond(t, ctx.t_min, ctx.t_max)


def _masked_path(model: nn.Module, tokens: jnp.ndarray, rng: jnp.ndarray, ctx: Diffusion,
                 attention_mask: jnp.ndarray | None, cond_mask: jnp.ndarray) -> jnp.ndarray:
    """Masked path term, with survival ``r = 1-t`` and mask probability ``t``.

    Denoiser and cavity coincide at masked positions, so for the distribution
    heads the path integrand is ``J_t = 1/t · 1{masked} · xent``. A ``score`` head
    predicts the SEDD residual with its uniform ``1/C`` scale folded out, whose
    Phi-sum is the same cross-entropy with the softmax log-partition replaced by
    the linear partition: ``mean_l exp(logits_l) - 1 - logits_truth + log(C)``
    (SEDD's masked denoising score entropy).
    """
    t, noisy_tokens, masked, sigma = _corrupt(ctx, tokens, rng, cond_mask)
    logits = model(noisy_tokens, sigma, attention_mask=attention_mask, cond_mask=cond_mask)
    # Anchor labels can lie outside the alphabet; anchors are never scored.
    logits, labels = logits[..., :ctx.num_classes], jnp.minimum(tokens, ctx.num_classes - 1)
    scored = _scored(cond_mask)

    if ctx.parametrization == "score":
        # Positions that pay nothing read zero logits, so exp cannot overflow there.
        score_logits = jnp.where((masked & (scored > 0))[..., None], logits, 0.0)
        logit_truth = jnp.take_along_axis(score_logits, labels[..., None], axis=-1)[..., 0]
        path_ce = (jnp.mean(jnp.exp(score_logits), axis=-1) - 1.0 - logit_truth
                   + math.log(ctx.num_classes))
    else:
        path_ce = optax.softmax_cross_entropy_with_integer_labels(logits, labels)

    rate = 1.0 / t
    j_t = rate[:, None] * masked.astype(jnp.float32) * path_ce
    return _scored_mean(j_t * (ctx.t_max - ctx.t_min), scored)


def _gidd_path(model: nn.Module, tokens: jnp.ndarray, rng: jnp.ndarray, ctx: Diffusion,
               attention_mask: jnp.ndarray | None, cond_mask: jnp.ndarray) -> jnp.ndarray:
    """GIDD path term; uniform diffusion is ``lam = 1``.

    Static noise law ``nu = (1-lam) delta_mask + lam unif(C)`` over the C content
    tokens; forward marginal ``q(y|e_l) = r 1{y=e_l} + (1-r) nu(y)`` with survival
    ``r = 1-t``. The jump-destination rate in direct time is ``nu(y)/r``, so the
    per-position integrand over reverse targets ``y != z_t`` is

        J_t = nu(z_t)/r · Σ_{y != z_t} Phi( d(y)/d(z_t), g_theta(y) ),

    with ``d(y) = q(y|e_{z_0})`` the true noisy marginal, and the coordinate
    entering only through ``g_theta``:

    * cavity:   ``g = m(y)/m(z_t)``, ``m(y) = r mu(y) + (1-r) nu(y)``;
    * denoiser: ``g(y) = Σ_l pi_theta(l) q(y|e_l)/q(z_t|e_l)``;
    * score:    ``g(y) = m_0(y)/m_0(z_t) exp(logits_y)`` for every ``y != z_t``,
                with ``g(z_t)=1`` and ``m_0 = r unif(C) + (1-r) nu`` the noisy
                marginal induced by a uniform cavity.

    For the cavity coordinate this Phi-sum has the closed form
    ``(KL(d||m) + IS(d(z_t)||m(z_t))) / d(z_t)``. A zero-initialized cavity head
    is exactly uniform, with NELBO ``log C`` per token.
    """
    lam = float(ctx.gidd_lambda)
    C, has_mask = ctx.num_classes, lam < 1.0
    t, noisy_tokens, _, sigma = _corrupt(ctx, tokens, rng, cond_mask)
    r = 1.0 - t
    logits_full = model(noisy_tokens, sigma, attention_mask=attention_mask,
                        cond_mask=cond_mask)
    logits = logits_full[..., :C]
    labels = jnp.minimum(tokens, C - 1)
    noisy_ids = jnp.minimum(noisy_tokens, C - 1)
    is_mask_z = (noisy_tokens == ctx.mask_id) if has_mask else jnp.zeros_like(tokens, bool)
    scored = _scored(cond_mask)

    # The forward channel q(y | e_l), one value per row: q_same at y = l, q_other at
    # every other content token, q_mask at the mask whatever l is.
    q_other = t * lam / C
    q_same = r + q_other
    q_mask = t * (1.0 - lam)

    # The true noisy marginal d at the current state.
    d_zt = jnp.where(noisy_ids == labels, q_same[:, None], q_other[:, None])
    if has_mask:
        d_zt = jnp.where(is_mask_z, q_mask[:, None], d_zt)
    lse = jax.nn.logsumexp(logits, axis=-1)
    logit_zt = jnp.take_along_axis(logits, noisy_ids[..., None], axis=-1)[..., 0]

    if ctx.parametrization == "cavity":
        # Closed form (KL(d||m) + IS(d(z_t)||m(z_t))) / d(z_t), with
        # IS(a||b) = a/b - log(a/b) - 1. On content m = r mu + q_other, and on the mask
        # m = d = q_mask, which adds nothing to the KL.
        log_r, log_q_other = jnp.log(r)[:, None], jnp.log(q_other)[:, None]
        # log m(y) = logaddexp(log r + log mu(y), log q_other); its sum S over the
        # content tokens is the only reduction over a row.
        S = jnp.sum(jnp.logaddexp(log_r[..., None] + logits - lse[..., None],
                                  log_q_other[..., None]), axis=-1)
        logit_x0 = jnp.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
        log_m_x0 = jnp.logaddexp(log_r + logit_x0 - lse, log_q_other)
        log_m_zt = jnp.logaddexp(log_r + logit_zt - lse, log_q_other)
        if has_mask:
            log_m_zt = jnp.where(is_mask_z, jnp.log(q_mask)[:, None], log_m_zt)
        # KL(d||m) = sum d log d - sum d log m, where d is q_same at x0 and q_other
        # elsewhere, and q_same - q_other = r.
        d_log_d = q_same * jnp.log(q_same) + (C - 1) * q_other * jnp.log(q_other)
        kl = d_log_d[:, None] - r[:, None] * log_m_x0 - q_other[:, None] * S
        log_ratio = jnp.log(d_zt) - log_m_zt
        phi_sum = (kl + jnp.exp(log_ratio) - log_ratio - 1.0) / d_zt
    else:
        # True ratios d(y) / d(z_t): to every content target [B, L, C] and to the mask [B, L].
        d = jnp.where(jax.nn.one_hot(labels, C, dtype=jnp.bool),
                      q_same[:, None, None], q_other[:, None, None])
        true_ratio = d / d_zt[..., None]
        true_ratio_mask = q_mask[:, None] / d_zt if has_mask else None

        # Model ratios g(y), and their logs where these are exact.
        log_model_ratio, log_model_ratio_mask, model_ratio_mask = None, None, None
        if ctx.parametrization == "score":
            # Unscored positions read zero, so exp stays finite where the weight is zero.
            log_g = gidd_score_logits(ctx, logits_full, noisy_tokens, r, t)
            log_g = jnp.where(scored[..., None] > 0, log_g, 0.0)
            model_ratio, log_model_ratio = jnp.exp(log_g[..., :C]), log_g[..., :C]
            if has_mask:
                model_ratio_mask, log_model_ratio_mask = jnp.exp(log_g[..., C]), log_g[..., C]
        else:  # denoiser
            # g(y) = sum_l pi(l) q(y|e_l) / q(z_t|e_l). As q(y|e_l) = q_other + r 1{l = y},
            # g(y) = q_other S + r pi(y) / q(z_t|e_y) with S = sum_l pi(l) / q(z_t|e_l).
            pi = jnp.exp(logits - lse[..., None])
            # From the logits: gathering from pi would materialize it.
            pi_zt = jnp.exp(logit_zt - lse)
            # q(z_t|e_l) is q_same at l = z_t and q_other elsewhere for a content z_t, and
            # q_mask for every l at a mask.
            S_content = (1.0 - pi_zt) / q_other[:, None] + pi_zt / q_same[:, None]
            S = jnp.where(is_mask_z, 1.0 / q_mask[:, None], S_content) if has_mask else S_content
            # q(z_t|e_y) at a target y != z_t.
            q_zt_given_y = jnp.where(is_mask_z[..., None], q_mask[:, None, None],
                                     q_other[:, None, None])
            model_ratio = (q_other[:, None, None] * S[..., None]
                           + r[:, None, None] * pi / q_zt_given_y)
            if has_mask:
                model_ratio_mask = q_mask[:, None] * S

        # Sum over the reverse targets y != z_t: every content column but the current
        # token, plus the mask column when the current state is clean.
        noisy_one_hot = jax.nn.one_hot(noisy_ids, C, dtype=jnp.bool) & ~is_mask_z[..., None]
        phi_sum = jnp.sum(_phi(true_ratio, model_ratio, log_model_ratio), axis=-1,
                          where=~noisy_one_hot)
        if has_mask:
            phi_mask = _phi(true_ratio_mask, model_ratio_mask, log_model_ratio_mask)
            phi_sum = phi_sum + jnp.where(is_mask_z, 0.0, phi_mask)

    path_rate = jnp.where(is_mask_z, 1.0 - lam, lam / C) / r[:, None]  # nu(z_t) / r
    j_t = path_rate * phi_sum
    return _scored_mean(j_t * (ctx.t_max - ctx.t_min), scored)


def path_term(model: nn.Module, ctx: Diffusion, batch: dict, rng: jnp.ndarray) -> jnp.ndarray:
    """Monte Carlo path term of the NELBO, one proposal time per sequence. ``batch``
    holds ``input_ids`` and, per task, ``cond_mask`` (clean anchors; text has none)
    and ``attention_mask``."""
    path = _masked_path if ctx.noise_type == "masked" else _gidd_path
    tokens = batch["input_ids"]
    return path(model, tokens, rng, ctx, batch.get("attention_mask"),
                batch.get("cond_mask", jnp.zeros(tokens.shape, bool)))


def decoder_log_probs(dl: jnp.ndarray) -> jnp.ndarray:
    """Log law of the decoder every consumer of a head shares: the denoiser smoothed
    towards uniform by ``DECODE_SMOOTHING``, as the score-to-cavity inverse can place
    exact zeros."""
    log_pi = jax.nn.log_softmax(dl, axis=-1)
    return jnp.logaddexp(log_pi + math.log1p(-DECODE_SMOOTHING),
                         math.log(DECODE_SMOOTHING / dl.shape[-1]))


def reconstruction(model: nn.Module, ctx: Diffusion, batch: dict,
                   rng: jnp.ndarray) -> jnp.ndarray:
    """Monte Carlo reconstruction term ``E[-log pi_theta(x_0 | z_{t_min})]``, decoded
    in the trained coordinate. Under masked noise only the masked positions pay."""
    tokens = batch["input_ids"]
    cond_mask = batch.get("cond_mask", jnp.zeros(tokens.shape, bool))
    t = ctx.t_min
    z, open_positions = renoise(ctx, tokens, 1.0 - t, random.split(rng, tokens.shape[0]),
                                cond_mask)
    t_b = jnp.full((tokens.shape[0],), t, dtype=jnp.float32)
    logits = model(z, model_time_cond(t_b, ctx.t_min, ctx.t_max),
                   attention_mask=batch.get("attention_mask"), cond_mask=cond_mask)
    scored = _scored(cond_mask)
    if ctx.noise_type != "masked":
        open_positions = scored
    labels = jnp.minimum(tokens, ctx.num_classes - 1)
    log_pi = decoder_log_probs(denoiser_logits(ctx, logits, z, 1.0 - t, t))
    log_pi_truth = jnp.take_along_axis(log_pi, labels[..., None], axis=-1)[..., 0]
    return -_scored_mean(log_pi_truth * open_positions.astype(jnp.float32), scored)


def prior_kl(ctx: Diffusion) -> float:
    """KL from the forward marginal ``q(z_{t_max} | x_0)`` to the prior
    ``q(z_{t_max} | x_0 ~ unif(C))``, per token and in float64. The mask column
    cancels, so masked diffusion pays ``r log C`` at ``r = 1 - t_max``."""
    r, C, lam = 1.0 - ctx.t_max, ctx.num_classes, ctx.gidd_lambda
    if ctx.noise_type == "masked":
        return r * math.log(C)
    # The forward marginal holds r + floor on x0 and floor on each other content token;
    # the prior holds r / C + floor on every content token.
    floor = (1.0 - r) * lam / C
    prior = r / C + floor
    return ((r + floor) * math.log((r + floor) / prior)
            + (C - 1.0) * (floor * math.log(floor / prior)))
