"""Checkpoint restore, shared by training resume and evaluation.

A run directory holds ``.hydra/config.yaml`` and ``checkpoints/<step>/{state,ema}``:
``state`` is ``{params, opt_state}`` for resuming, ``ema`` is ``{params}``. Orbax
compares tree structure, not shapes (a wider head would load silently), so the
restore checks the shapes itself.
"""
from __future__ import annotations

import os
from typing import Any

import flax.nnx as nn
import jax
import orbax.checkpoint as ocp
from omegaconf import DictConfig, OmegaConf

from .data import create_tokenizer
from .models import build_model, config_from_hydra
from .process import Diffusion


def _shapes(tree) -> dict[str, tuple]:
    """``a/b/c -> shape`` per array leaf, without the trailing ``value`` level that
    the on-disk metadata spells out for an nnx Variable."""
    leaves = jax.tree_util.tree_flatten_with_path(
        tree, is_leaf=lambda x: getattr(x, "shape", None) is not None)[0]
    out = {}
    for path, leaf in leaves:
        if getattr(leaf, "shape", None) is None:
            continue          # optimizer states store no array for masked-out params
        keys = [str(getattr(k, "key", getattr(k, "idx", k))) for k in path]
        out["/".join(keys[:-1] if keys[-1] == "value" else keys)] = tuple(leaf.shape)
    return out


def restore(manager: ocp.CheckpointManager, step: int,
            targets: dict[str, dict]) -> ocp.args.Composite:
    """Restore the items ``targets = {item: pytree}`` of one step, each checked against
    its on-disk metadata first; a target may name a subset of its item's entries."""
    for item, target in targets.items():
        item_dir = os.path.join(str(manager.directory), str(step), item)
        on_disk = _shapes(ocp.PyTreeCheckpointer().metadata(item_dir).item_metadata)
        on_disk = {k: v for k, v in on_disk.items() if k.split("/")[0] in target}
        wanted = _shapes(target)
        bad = {
            k: {"model": wanted.get(k, "absent"), "checkpoint": on_disk.get(k, "absent")}
            for k in wanted.keys() | on_disk.keys()
            if wanted.get(k) != on_disk.get(k)
        }
        if bad:
            raise ValueError(f"checkpoint {item_dir} does not match this model: {bad}")

    return manager.restore(step, args=ocp.args.Composite(**{
        item: ocp.args.PyTreeRestore(
            target, restore_args=ocp.checkpoint_utils.construct_restore_args(target),
            partial_restore=True)
        for item, target in targets.items()}))


def load_config(checkpoint_dir: str) -> DictConfig:
    """The training config of the run that wrote ``checkpoint_dir``
    (``<run>/checkpoints/<step>``)."""
    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint_dir)))
    return OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))


def load_checkpoint(
    checkpoint_dir: str, read_as: str | None, compute_dtype: str | None
) -> tuple[nn.Module, DictConfig, Any, Diffusion, int]:
    """``(model, cfg, tok, ctx, step)`` of one checkpoint step: its EMA weights and
    training config. ``read_as`` reads the head in another coordinate without
    converting it (the head width depends on the kernel alone); ``compute_dtype``
    runs another dtype than training did, without flash attention unless bfloat16."""
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    step = int(os.path.basename(checkpoint_dir))
    cfg = load_config(checkpoint_dir)
    if read_as is not None:
        cfg.diffusion.parametrization = read_as
    if compute_dtype is not None:
        cfg.model.dtype = compute_dtype
        cfg.model.flash_attention = bool(cfg.model.flash_attention) and compute_dtype == "bfloat16"
    tok = create_tokenizer(cfg)
    ctx = Diffusion.from_config(cfg, tok)
    model = build_model(config_from_hydra(cfg, tok), nn.Rngs(0))
    manager = ocp.CheckpointManager(os.path.dirname(checkpoint_dir))
    restored = restore(manager, step, {"ema": {"params": nn.state(model, nn.Param)}})
    nn.update(model, restored.ema["params"])
    print(f"Loaded checkpoint step {step} from {checkpoint_dir}")
    return model, cfg, tok, ctx, step
