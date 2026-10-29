"""Denoiser construction from a Hydra config."""
from __future__ import annotations

import flax.nnx as nn
from omegaconf import DictConfig

from .dit import DIT, DITConfig
from .structured import GraphDIT, MazeDIT

ARCHITECTURES = {"dit": DIT, "graph_dit": GraphDIT, "maze_dit": MazeDIT}


def config_from_hydra(cfg: DictConfig, tok) -> DITConfig:
    """The ``cfg.model`` architecture over the task tokenizer's vocabulary.

    The embedding covers the whole vocabulary, mask included; the head predicts
    content only, plus the mask logit for gidd below lambda = 1. Head widths are
    8-aligned, which the head GEMM runs markedly faster on.
    """
    m, d = cfg.model, cfg.data
    mask_logit = cfg.diffusion.noise_type == "gidd" and cfg.diffusion.gidd_lambda < 1.0
    graph = m.architecture == "graph_dit"
    maze = m.architecture == "maze_dit"
    return DITConfig(
        vocab_size=len(tok),
        head_width=-(-(tok.num_classes + mask_logit) // 8) * 8,
        hidden_size=m.hidden_dim,
        n_heads=m.num_heads,
        n_blocks=m.num_layers,
        cond_dim=m.cond_dim,
        mlp_dim=m.mlp_dim,
        max_seq_len=d.max_seq_len,
        clue_embedding=m.clue_embedding,
        dtype=m.dtype,
        flash_attention=m.flash_attention,
        architecture=m.architecture,
        graph_n=d.graphcolor.n if graph else None,
        grid_height=d.maze.grid_height if maze else None,
        grid_width=d.maze.grid_width if maze else None,
        grid_encoder_dim=m.grid_encoder_dim if maze else None,
        grid_encoder_dilations=(tuple((int(a), int(b)) for a, b in m.grid_encoder_dilations)
                                if maze else None),
        grid_pool=tuple(int(p) for p in m.grid_pool) if maze else None,
    )


def build_model(config: DITConfig, rngs: nn.Rngs) -> nn.Module:
    """Instantiate the configured denoiser."""
    return ARCHITECTURES[config.architecture](config, rngs)
