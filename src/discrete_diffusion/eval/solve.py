"""Solve rate of a logic-task checkpoint under one sampler and grid:
``python -m discrete_diffusion.eval.solve checkpoint_dir=<run>/checkpoints/<step> nfe=64``.

Each held-out instance is sampled once with its problem as clean anchors, and the
task's verifier checks the completed solution.
"""
from __future__ import annotations

import math

import hydra
import jax.numpy as jnp
import numpy as np
from jax import random
from omegaconf import DictConfig

from ..checkpoints import load_checkpoint
from ..data import eval_batches, solved
from ..process import survival_grid
from ..sample import sample
from . import write_result


@hydra.main(config_path="../../../configs/eval", config_name="solve", version_base="1.3")
def main(cfg: DictConfig) -> None:
    model, run_cfg, _, ctx, _ = load_checkpoint(cfg.checkpoint_dir, cfg.read_as,
                                                cfg.compute_dtype)
    if run_cfg.data.source == "text":
        raise ValueError("text has no verifier; sample it with discrete_diffusion.sample")
    grid = survival_grid(cfg.nfe, ctx.t_min, ctx.t_max, cfg.spacing)
    keys = random.split(random.PRNGKey(cfg.seed), cfg.n)
    hits = []
    for i, batch in enumerate(eval_batches(run_cfg, cfg.split, cfg.batch_size, cfg.n, cfg.seed)):
        cond = jnp.asarray(batch["cond_mask"])
        seq = jnp.where(cond, jnp.asarray(batch["input_ids"]), 0)   # hide the solution
        pred = sample(cfg.sampler, model, ctx, seq, cond, batch.get("attention_mask"),
                      keys[i * cfg.batch_size:(i + 1) * cfg.batch_size], grid,
                      cfg.temperature)
        hits.append(solved(run_cfg, batch, pred))
    rate = float(np.concatenate(hits).mean())
    write_result({"solve_rate": rate, "se": math.sqrt(rate * (1.0 - rate) / cfg.n)})


if __name__ == "__main__":
    main()
