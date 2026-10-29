"""Graph- and grid-aware DiTs with the sequence denoiser's interface."""
from __future__ import annotations

import flax.nnx as nn
import jax
import jax.numpy as jnp

from .dit import (
    DDiTBlock,
    DDitFinalLayer,
    DiffusionConditioner,
    DITConfig,
    EmbeddingLayer,
    LayerNorm,
    lecun_normal,
    modulate,
    precompute_freqs_cis,
)


def graph_locality_mask(adjacency: jnp.ndarray, n_heads: int) -> jnp.ndarray:
    """Mixed local/global attention mask with the first half of heads local."""
    batch, n, _ = adjacency.shape
    n_local = n_heads // 2
    local = adjacency | jnp.eye(n, dtype=jnp.bool_)[None]
    local = jnp.broadcast_to(local[:, None], (batch, n_local, n, n))
    global_ = jnp.ones((batch, n_heads - n_local, n, n), dtype=jnp.bool_)
    return jnp.concatenate([local, global_], axis=1)


class GraphDDiTBlock(DDiTBlock):
    """DiT block with mixed local/global graph attention and relation biases."""

    def __init__(self, config: DITConfig, rngs: nn.Rngs):
        super().__init__(config, rngs)
        # 1D parameters, so optim.py classifies them as biases rather than Muon matrices.
        zeros = nn.initializers.zeros
        self.self_bias = nn.Param(zeros(rngs.params(), (self.n_heads,)))
        self.edge_bias = nn.Param(zeros(rngs.params(), (self.n_heads,)))
        self.nonedge_bias = nn.Param(zeros(rngs.params(), (self.n_heads,)))

    def __call__(self, x: jnp.ndarray, c: jnp.ndarray, graph_mask: jnp.ndarray,
                 edge_sel: jnp.ndarray, eye_sel: jnp.ndarray) -> jnp.ndarray:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = self._adaln(c)
        batch, n = edge_sel.shape[0], edge_sel.shape[-1]
        bias = jnp.broadcast_to(self.nonedge_bias.value[None, :, None, None],
                                (batch, self.n_heads, n, n))
        bias = jnp.where(edge_sel, self.edge_bias.value[None, :, None, None], bias)
        bias = jnp.where(eye_sel, self.self_bias.value[None, :, None, None], bias)
        bias = bias.astype(self.compute_dtype)
        x = self._attention(x, shift_attn, scale_attn, gate_attn, bias=bias, mask=graph_mask)
        return self._mlp(x, shift_mlp, scale_mlp, gate_mlp)


class GraphDIT(nn.Module):
    """Graphormer-style denoiser over vertex colors.

    The ``[B, n, n]`` adjacency arrives as ``attention_mask``: it masks the local
    half of the heads and selects per-head edge, non-edge and self biases.
    """

    def __init__(self, config: DITConfig, rngs: nn.Rngs):
        self.config = config
        self.compute_dtype = config.compute_dtype
        self.conditioner = DiffusionConditioner(config, rngs)
        self.degree_embed = EmbeddingLayer(config.hidden_size, config.graph_n, rngs)
        self.blocks = nn.List([GraphDDiTBlock(config, rngs) for _ in range(config.n_blocks)])
        self.output_layer = DDitFinalLayer(config, rngs)

    def __call__(self, input_ids: jnp.ndarray, sigma: jnp.ndarray,
                 attention_mask: jnp.ndarray | None = None,
                 cond_mask: jnp.ndarray | None = None) -> jnp.ndarray:
        adjacency = attention_mask.astype(jnp.bool_)
        x, c = self.conditioner(input_ids, sigma)
        x = x + self.degree_embed(jnp.sum(adjacency, axis=-1).astype(jnp.int32))
        graph_mask = graph_locality_mask(adjacency, self.config.n_heads)
        edge_sel = adjacency[:, None]
        eye_sel = jnp.eye(adjacency.shape[-1], dtype=jnp.bool_)[None, None]
        x = x.astype(self.compute_dtype)
        for block in self.blocks:
            x = block(x, c, graph_mask, edge_sel, eye_sel)
        return self.output_layer(x.astype(jnp.float32), c)


class GridResidualBlock(nn.Module):
    """Two pre-normalized dilated convolutions with an identity residual."""

    def __init__(self, width: int, dilations: tuple[int, int], compute_dtype: jnp.dtype,
                 rngs: nn.Rngs):
        self.compute_dtype = compute_dtype
        self.norm1 = LayerNorm(width, rngs)
        self.norm2 = LayerNorm(width, rngs)
        self.conv1, self.conv2 = (
            nn.Conv(width, width, kernel_size=(3, 3), padding="SAME", kernel_dilation=(d, d),
                    dtype=compute_dtype, kernel_init=nn.initializers.lecun_normal(), rngs=rngs)
            for d in dilations)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = self.conv1(nn.silu(self.norm1(x)).astype(self.compute_dtype))
        x = self.conv2(nn.silu(self.norm2(x)).astype(self.compute_dtype))
        return residual + x


class GridEncoder(nn.Module):
    """Residual convolutional encoder for a fixed board grid, with space-to-depth pooling."""

    def __init__(self, config: DITConfig, rngs: nn.Rngs):
        width = config.grid_encoder_dim
        self.height = config.grid_height
        self.width = config.grid_width
        self.pool = tuple(config.grid_pool)
        self.compute_dtype = config.compute_dtype
        self.board_embed = EmbeddingLayer(width, config.vocab_size, rngs)
        kernel_init = nn.initializers.lecun_normal()
        bias_init = nn.initializers.zeros
        # ResNet staging: each pool folds a pxp neighbourhood into the channels
        # and divides the resolution by p, doubling the width.
        blocks, folds = [], []
        for pair, p in zip(config.grid_encoder_dilations, self.pool, strict=True):
            blocks.append(GridResidualBlock(width, pair, self.compute_dtype, rngs))
            # A None entry keeps the fold indices aligned with the blocks.
            folds.append(None if p == 1 else nn.Linear(
                width * p * p, 2 * width, dtype=self.compute_dtype,
                kernel_init=kernel_init, bias_init=bias_init, rngs=rngs))
            if p > 1:
                width *= 2
        self.blocks = nn.List(blocks)
        self.folds = nn.List(folds)
        self.norm = LayerNorm(width, rngs)
        self.projection = nn.Linear(width, config.hidden_size, dtype=self.compute_dtype,
                                    kernel_init=kernel_init, bias_init=bias_init, rngs=rngs)

    def __call__(self, board: jnp.ndarray) -> jnp.ndarray:
        B = board.shape[0]
        gh, gw = self.height, self.width
        x = self.board_embed(board)
        x = x.reshape(B, gh, gw, x.shape[-1]).astype(self.compute_dtype)
        for block, fold, p in zip(self.blocks, self.folds, self.pool, strict=True):
            x = block(x)
            if p > 1:
                x = x.reshape(B, gh // p, p, gw // p, p, x.shape[-1])
                x = fold(x.transpose(0, 1, 3, 2, 4, 5).reshape(
                    B, gh // p, gw // p, -1))
                gh, gw = gh // p, gw // p
        x = self.projection(self.norm(x).astype(self.compute_dtype))
        return x.reshape(B, gh * gw, -1)


class MazeDDiTBlock(DDiTBlock):
    """Move self-attention, move-to-board cross-attention, then MLP."""

    def __init__(self, config: DITConfig, rngs: nn.Rngs):
        super().__init__(config, rngs, n_adaln=9)
        d, h, k = self.dim, self.n_heads, self.head_dim
        self.norm_cross = LayerNorm(d, rngs)
        self.norm_context = LayerNorm(d, rngs)
        self.cross_q = nn.Param(lecun_normal(rngs.params(), (d, h, k)))
        self.cross_kv = nn.Param(lecun_normal(rngs.params(), (d, h, 2, k)))
        self.cross_out = nn.Param(lecun_normal(rngs.params(), (h, k, d)))

    def cross_kv_for(self, context: jnp.ndarray) -> jnp.ndarray:
        """Cross-attention K/V ``[B, T, H, 2, Dh]``. A function of ``context``
        alone -- no adaLN, no timestep -- so it is reusable for as long as the
        board is."""
        cdt = self.compute_dtype
        memory = self.norm_context(context).astype(cdt)
        return jnp.einsum("btd,dhqk->bthqk", memory, self.cross_kv.value.astype(cdt))

    def __call__(self, x: jnp.ndarray, cross_kv: jnp.ndarray, cos: jnp.ndarray,
                 sin: jnp.ndarray, c: jnp.ndarray) -> jnp.ndarray:
        cdt = self.compute_dtype
        (shift_self, scale_self, gate_self, shift_cross, scale_cross, gate_cross,
         shift_mlp, scale_mlp, gate_mlp) = self._adaln(c)
        x = self._attention(x, shift_self, scale_self, gate_self, cos=cos, sin=sin)
        queries = modulate(self.norm_cross(x), shift_cross, scale_cross).astype(cdt)
        q = jnp.einsum("bsd,dhk->bshk", queries, self.cross_q.value.astype(cdt))
        k, v = cross_kv[:, :, :, 0], cross_kv[:, :, :, 1]
        even = k.shape[1] % 2 == 0 and x.shape[1] % 2 == 0
        cross_out = jax.nn.dot_product_attention(
            q, k, v, implementation="cudnn" if self.flash_attention and even else None)
        x = x + gate_cross * jnp.einsum("bshd,hdo->bso", cross_out,
                                        self.cross_out.value.astype(cdt))
        return self._mlp(x, shift_mlp, scale_mlp, gate_mlp)


class MazeDIT(nn.Module):
    """Grid encoder plus a move transformer, for ``[board | moves]`` sequences.

    The board prefix is encoded convolutionally and read through cross-attention;
    the move suffix is denoised. Geometry comes from ``grid_height``/``grid_width``.
    """

    def __init__(self, config: DITConfig, rngs: nn.Rngs):
        self.config = config
        self.board_len = config.grid_height * config.grid_width
        self.compute_dtype = config.compute_dtype
        self.conditioner = DiffusionConditioner(config, rngs)
        self.grid_encoder = GridEncoder(config, rngs)
        self.rotary_cos, self.rotary_sin = precompute_freqs_cis(
            config.hidden_size // config.n_heads, config.max_seq_len - self.board_len)
        self.blocks = nn.List([MazeDDiTBlock(config, rngs) for _ in range(config.n_blocks)])
        self.output_layer = DDitFinalLayer(config, rngs)

    def encode_board(self, input_ids: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
        """Per-block cross-attention K/V of the board prefix of ``input_ids``."""
        context = self.grid_encoder(input_ids[:, : self.board_len])
        return tuple(block.cross_kv_for(context) for block in self.blocks)

    def __call__(self, input_ids: jnp.ndarray, sigma: jnp.ndarray,
                 attention_mask: jnp.ndarray | None = None,
                 cond_mask: jnp.ndarray | None = None,
                 board: tuple[jnp.ndarray, ...] | None = None) -> jnp.ndarray:
        # The board stays anchored while sampling, so a sampler passes its
        # encoding once (``board``) instead of recomputing it at every level.
        if board is None:
            board = self.encode_board(input_ids)
        x, c = self.conditioner(input_ids[:, self.board_len:], sigma)
        x = x.astype(self.compute_dtype)
        for block, cross_kv in zip(self.blocks, board, strict=True):
            x = block(x, cross_kv, self.rotary_cos, self.rotary_sin, c)
        logits = self.output_layer(x.astype(jnp.float32), c)
        prefix = jnp.zeros((input_ids.shape[0], self.board_len, logits.shape[-1]), logits.dtype)
        return jnp.concatenate([prefix, logits], axis=1)
