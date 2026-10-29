"""Train one model: ``python -m discrete_diffusion.train --config-name paper/owt/masked_cavity``.

The run directory (``hydra.run.dir``) holds the composed config, ``metrics.jsonl`` and
``checkpoints/<step>/{state,ema}``; rerunning its command resumes from the latest step.
"""
from __future__ import annotations

import json
import os
import time
from functools import partial

import flax.nnx as nn
import hydra
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import wandb
from hydra.core.hydra_config import HydraConfig
from jax import random
from jax.sharding import AxisType, NamedSharding, PartitionSpec
from omegaconf import DictConfig, OmegaConf

from .checkpoints import restore
from .data import create_tokenizer, eval_batches, train_batches
from .loss import path_term, prior_kl, reconstruction
from .models import build_model, config_from_hydra
from .optim import build_optimizer
from .process import Diffusion


class MetricsLogger:
    """Append metrics to ``metrics.jsonl`` in the run directory, print them, and
    send them to W&B unless ``wandb.mode`` is ``disabled``."""

    def __init__(self, cfg: DictConfig, run_dir: str):
        self.path = os.path.join(run_dir, "metrics.jsonl")
        self.wandb = cfg.wandb.mode != "disabled"
        if self.wandb:
            # The id kept in the run directory reattaches a resumed run.
            id_path = os.path.join(run_dir, "wandb_id.txt")
            if not os.path.exists(id_path):
                with open(id_path, "w") as f:
                    f.write(wandb.util.generate_id())
            with open(id_path) as f:
                run_id = f.read().strip()
            wandb.init(project=cfg.wandb.project, name=cfg.wandb.run_name, id=run_id,
                       resume="allow", mode=cfg.wandb.mode,
                       config=OmegaConf.to_container(cfg, resolve=True))

    def log(self, metrics: dict, step: int, split: str) -> None:
        metrics = {k: float(v) for k, v in metrics.items()}
        logged = {f"{split}/{k}": v for k, v in metrics.items()}
        if self.wandb:
            wandb.log(logged, step=step)
        with open(self.path, "a") as f:
            f.write(json.dumps({"step": step, **logged}) + "\n")
        print(f"[{split} step={step}] "
              + " ".join(f"{k}={v:.4g}" for k, v in metrics.items()), flush=True)

    def finish(self) -> None:
        if self.wandb:
            wandb.finish()


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_dir = HydraConfig.get().runtime.output_dir
    t = cfg.training
    batch_size, num_steps, save_every = t.global_batch_size, t.num_steps, cfg.checkpoint.save_every
    # Data parallel: every device holds the whole model and a slice of each batch (JAX
    # rejects a batch that does not split). Auto axes let XLA propagate the batch
    # sharding; explicit axes reject the embedding gather.
    mesh = jax.make_mesh((jax.device_count(),), ("data",), axis_types=(AxisType.Auto,))
    replicated = NamedSharding(mesh, PartitionSpec())
    batch_sharding = NamedSharding(mesh, PartitionSpec("data"))

    tok = create_tokenizer(cfg)
    ctx = Diffusion.from_config(cfg, tok)
    init_rng, base_rng = random.split(random.PRNGKey(t.seed))
    model = build_model(config_from_hydra(cfg, tok), nn.Rngs(init_rng))
    lr_schedule = optax.linear_schedule(0.0, float(t.learning_rate), int(t.warmup_steps))
    optimizer = nn.Optimizer(model, build_optimizer(t, lr_schedule), wrt=nn.Param)
    for module in (model, optimizer):
        nn.update(module, jax.device_put(nn.state(module), replicated))
    ema = jax.tree.map(jnp.copy, nn.state(model, nn.Param))
    ema_rate = 1.0 - float(t.ema_decay)

    @partial(nn.jit, donate_argnums=(0, 1))
    def train_step(model, optimizer, base_rng, step, batch):
        rng = random.fold_in(base_rng, step)
        loss, grads = nn.value_and_grad(lambda m: path_term(m, ctx, batch, rng))(model)
        optimizer.update(model, grads)
        return {"loss": loss, "grad_norm": optax.global_norm(grads), "lr": lr_schedule(step)}

    @partial(nn.jit, donate_argnums=(0,))
    def ema_update(ema, params):
        return optax.incremental_update(params, ema, ema_rate)

    path_fn = nn.jit(lambda m, batch, rng: path_term(m, ctx, batch, rng))
    rec_fn = nn.jit(lambda m, batch, rng: reconstruction(m, ctx, batch, rng))
    prior = prior_kl(ctx)
    val_set = [jax.device_put(batch, batch_sharding) for batch in eval_batches(
        cfg, "val", batch_size, t.val_batches * batch_size, t.seed)]
    eval_rng = random.PRNGKey(t.seed + 1)

    def validate(step: int) -> dict:
        """NELBO terms in nats per scored token: the path term over the validation
        batches, the reconstruction on the first, and the prior KL."""
        val_rng, rec_rng = random.split(random.fold_in(eval_rng, step))
        total = sum(path_fn(model, batch, random.fold_in(val_rng, i))
                    for i, batch in enumerate(val_set))
        path = float(total) / len(val_set)
        rec = float(rec_fn(model, val_set[0], rec_rng))
        return {"path": path, "rec": rec, "prior_kl": prior, "nelbo": path + rec + prior}

    manager = ocp.CheckpointManager(
        os.path.join(run_dir, "checkpoints"),
        options=ocp.CheckpointManagerOptions(max_to_keep=cfg.checkpoint.max_to_keep,
                                             create=True))
    start = manager.latest_step() or 0
    if start:
        restored = restore(manager, start, {
            "state": {"params": nn.state(model, nn.Param), "opt_state": nn.state(optimizer)},
            "ema": {"params": ema}})
        nn.update(model, restored.state["params"])
        nn.update(optimizer, restored.state["opt_state"])
        ema = restored.ema["params"]
        print(f"Resuming from step {start}", flush=True)

    logger = MetricsLogger(cfg, run_dir)
    if start == 0:
        logger.log(validate(0), 0, "val")
    sums, count = {}, 0
    batches = train_batches(cfg, start)
    # The training rate excludes validation and checkpoint writes.
    mark_step, mark_time = start, time.time()
    for step in range(start, num_steps):
        batch = jax.device_put(next(batches), batch_sharding)
        metrics = train_step(model, optimizer, base_rng, jnp.asarray(step, jnp.int32), batch)
        lr = metrics.pop("lr")
        sums, count = {k: sums.get(k, 0.0) + v for k, v in metrics.items()}, count + 1
        ema = ema_update(ema, nn.state(model, nn.Param))

        done = step + 1
        log_train = done % t.log_steps == 0 or done == num_steps
        log_val = done % t.val_steps == 0
        save = done % save_every == 0 or done == num_steps
        if log_train:
            metrics = jax.device_get({k: v / count for k, v in sums.items()} | {"lr": lr})
            sums, count = {}, 0
            metrics["steps_per_sec"] = (done - mark_step) / (time.time() - mark_time)
            logger.log(metrics, done, "train")
        if log_val:
            logger.log(validate(done), done, "val")
        if save:
            manager.save(done, args=ocp.args.Composite(
                state=ocp.args.PyTreeSave({"params": nn.state(model, nn.Param),
                                           "opt_state": nn.state(optimizer)}),
                ema=ocp.args.PyTreeSave({"params": ema})))
        if log_train or log_val or save:
            mark_step, mark_time = done, time.time()

    manager.wait_until_finished()
    logger.finish()


if __name__ == "__main__":
    main()
