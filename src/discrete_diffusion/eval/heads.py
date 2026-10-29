"""Agreement and own-token sensitivity of one kernel's heads:
``python -m discrete_diffusion.eval.heads "views=[A,A:denoiser,B]"``.

A view is a checkpoint, optionally read in another coordinate (``<checkpoint>:denoiser``).
Held-out sequences are corrupted at per-sequence times and read through the smoothed
decoder; a statistic is a per-token Jensen-Shannon divergence in nats, averaged per
batch, with its standard error over batches:

* ``agreement``: every pair of views read as denoisers, over the free positions
  (under masked noise only the masked ones: a visible token's denoiser is a point mass);
* ``sensitivity``: one position per sequence is redrawn from the forward kernel, and
  each view, read as a cavity and as a denoiser, is compared at the two inputs where
  the token changed. An exact cavity does not move; a denoiser moves by the Bayes
  factor of its own token.
"""
from __future__ import annotations

import dataclasses
import itertools
import math
from functools import partial

import flax.nnx as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from omegaconf import DictConfig

from ..checkpoints import load_checkpoint
from ..coords import cavity_logits, denoiser_logits
from ..data import eval_batches
from ..loss import decoder_log_probs
from ..process import model_time_cond, renoise, sample_time
from . import write_result

READS = {"cavity": cavity_logits, "denoiser": denoiser_logits}


def _js(log_a: jnp.ndarray, log_b: jnp.ndarray) -> jnp.ndarray:
    """Jensen-Shannon divergence between two log laws on the last axis."""
    log_m = jnp.logaddexp(log_a, log_b) - math.log(2.0)
    return 0.5 * (jnp.exp(log_a) * (log_a - log_m)
                  + jnp.exp(log_b) * (log_b - log_m)).sum(-1)


@nn.jit
def _forward(model, z, sigma, attention_mask, cond_mask):
    return model(z, sigma, attention_mask=attention_mask, cond_mask=cond_mask)


@partial(jax.jit, static_argnames=("ctx_a", "ctx_b"))
def _agreement(logits_a, logits_b, z, r, t, ctx_a, ctx_b):
    """Per-position JS ``[B, L]`` between two heads read as denoisers."""
    return _js(decoder_log_probs(denoiser_logits(ctx_a, logits_a, z, r, t)),
               decoder_log_probs(denoiser_logits(ctx_b, logits_b, z, r, t)))


@partial(jax.jit, static_argnames=("ctx", "read"))
def _sensitivity(logits, logits_alt, z, z_alt, pos, r, t, ctx, read):
    """Per-row JS ``[B]`` at position ``pos`` between one head's laws at ``z`` and
    at ``z_alt``, read in the coordinate ``read``."""
    rows = jnp.arange(pos.shape[0])
    log_a, log_b = (
        decoder_log_probs(READS[read](ctx, h[rows, pos][:, None], x[rows, pos][:, None],
                                      r, t))[:, 0]
        for h, x in ((logits, z), (logits_alt, z_alt)))
    return _js(log_a, log_b)


@hydra.main(config_path="../../../configs/eval", config_name="heads", version_base="1.3")
def main(cfg: DictConfig) -> None:
    runs, views = {}, {}
    for view in cfg.views:
        path, _, read_as = view.partition(":")
        if path not in runs:
            model, run_cfg, _, ctx, _ = load_checkpoint(path, None, cfg.compute_dtype)
            runs[path] = (model, run_cfg, ctx)
        ctx = runs[path][2]
        views[view] = (path, dataclasses.replace(ctx, parametrization=read_as) if read_as else ctx)
    _, run_cfg, ctx = next(iter(runs.values()))
    if any((c.noise_type, c.gidd_lambda, c.t_min, c.t_max, rc.data)
           != (ctx.noise_type, ctx.gidd_lambda, ctx.t_min, ctx.t_max, run_cfg.data)
           for _, rc, c in runs.values()):
        raise ValueError("the views must share one kernel, training window and dataset")
    if cfg.stat == "agreement":
        items = [(f"{a} | {b}", views[a], views[b]) for a, b in itertools.combinations(views, 2)]
    elif cfg.stat == "sensitivity":
        if ctx.noise_type == "masked":
            raise ValueError("a masked head's cavity is untrained at visible tokens, so "
                             "it has no own-token sensitivity")
        items = [(f"{view} as {read}", views[view], read) for view in views for read in READS]
    else:
        raise ValueError(f"unknown stat {cfg.stat!r}; use agreement or sensitivity")

    def stats(times) -> dict:
        values = {label: [] for label, *_ in items}
        for i, batch in enumerate(eval_batches(run_cfg, cfg.split, cfg.batch_size, cfg.n,
                                               cfg.seed)):
            time_key, corrupt_key, redraw_key = random.split(
                random.fold_in(random.PRNGKey(cfg.seed), i), 3)
            x0 = jnp.asarray(batch["input_ids"])
            attention_mask = batch.get("attention_mask")
            anchors = jnp.asarray(batch.get("cond_mask", np.zeros(x0.shape, bool)))
            B, L = x0.shape
            t = times(time_key, B)
            r = 1.0 - t
            sigma = model_time_cond(t, ctx.t_min, ctx.t_max)
            z, _ = renoise(ctx, x0, r, random.split(corrupt_key, B), anchors)
            logits = {path: _forward(model, z, sigma, attention_mask, anchors)
                      for path, (model, _, _) in runs.items()}
            if cfg.stat == "agreement":
                free = ~anchors & (z == ctx.mask_id) if ctx.noise_type == "masked" else ~anchors
                readings = {label: (_agreement(logits[pa], logits[pb], z, r, t, ca, cb), free)
                            for label, (pa, ca), (pb, cb) in items}
            else:
                pos_key, draw_key = random.split(redraw_key)
                pos = random.randint(pos_key, (B,), 0, L)
                fresh, _ = renoise(ctx, x0, r, random.split(draw_key, B), anchors)
                rows = jnp.arange(B)
                z_alt = z.at[rows, pos].set(fresh[rows, pos])
                changed = z_alt[rows, pos] != z[rows, pos]
                logits_alt = {path: _forward(model, z_alt, sigma, attention_mask, anchors)
                              for path, (model, _, _) in runs.items()}
                readings = {label: (_sensitivity(logits[path], logits_alt[path], z, z_alt, pos,
                                                 r, t, c, read), changed)
                            for label, (path, c), read in items}
            for label, (js, keep) in readings.items():
                kept = np.asarray(keep)
                if kept.any():      # a batch with nothing to compare carries no reading
                    values[label].append(np.asarray(js, np.float64)[kept].mean())
        return {label: {"js": float(np.mean(v)),
                        "se": float(np.std(v, ddof=1) / math.sqrt(len(v)))}
                for label, v in values.items()}

    result = {}
    if cfg.t is None:
        result["proposal"] = stats(lambda key, B: sample_time(key, B, ctx.t_min, ctx.t_max))
    else:
        for t in cfg.t:
            result[f"t={t}"] = stats(lambda key, B, t=t: jnp.full((B,), t, jnp.float32))
    write_result(result)


if __name__ == "__main__":
    main()
