"""Held-out NELBO of a checkpoint in nats and bits per scored token:
``python -m discrete_diffusion.eval.elbo checkpoint_dir=<run>/checkpoints/<step>``.

The training-time estimator: the path term on the training time proposal and the
reconstruction at ``t_min``, both Monte Carlo per batch, plus the exact prior KL.
``read_as`` prices a head read in a coordinate it was not trained in.
"""
from __future__ import annotations

import math

import flax.nnx as nn
import hydra
import numpy as np
from jax import random
from omegaconf import DictConfig

from ..checkpoints import load_checkpoint
from ..data import eval_batches
from ..loss import path_term, prior_kl, reconstruction
from . import write_result


@hydra.main(config_path="../../../configs/eval", config_name="elbo", version_base="1.3")
def main(cfg: DictConfig) -> None:
    model, run_cfg, _, ctx, _ = load_checkpoint(cfg.checkpoint_dir, cfg.read_as,
                                                cfg.compute_dtype)
    path_fn = nn.jit(lambda m, batch, rng: path_term(m, ctx, batch, rng))
    rec_fn = nn.jit(lambda m, batch, rng: reconstruction(m, ctx, batch, rng))
    paths, recs = [], []
    for i, batch in enumerate(eval_batches(run_cfg, cfg.split, cfg.batch_size, cfg.n, cfg.seed)):
        path_rng, rec_rng = random.split(random.fold_in(random.PRNGKey(cfg.seed), i))
        paths.append(path_fn(model, batch, path_rng))
        recs.append(rec_fn(model, batch, rec_rng))
    paths, recs = np.asarray(paths, np.float64), np.asarray(recs, np.float64)
    prior = prior_kl(ctx)
    nelbo = float(paths.mean() + recs.mean() + prior)
    se = float(np.std(paths + recs, ddof=1) / math.sqrt(len(paths)))
    write_result({
        "path": float(paths.mean()), "reconstruction": float(recs.mean()), "prior_kl": prior,
        "nelbo": nelbo, "se": se, "nelbo_bits": nelbo / math.log(2), "se_bits": se / math.log(2),
    })


if __name__ == "__main__":
    main()
