"""The diffusion transformer (DiT) in Flax NNX: adaLN time conditioning, rotary
self-attention and a zero-initialized output head.

Adapted from the DiT of MDLM (https://github.com/kuleshov-group/mdlm), licensed under Apache-2.0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import flax.nnx as nn
import jax
import jax.numpy as jnp

FREQUENCY_EMBEDDING_SIZE = 256  # sinusoidal features of the time embedding
MAX_PERIOD = 10_000             # longest period of the time embedding
ROPE_THETA = 10_000.0
LN_EPS = 1e-5


@dataclass(frozen=True)
class DITConfig:
    vocab_size: int             # embedding rows: content tokens, task specials and the mask
    head_width: int             # output logits, 8-aligned
    hidden_size: int
    n_heads: int
    n_blocks: int
    cond_dim: int               # adaLN time-conditioning width
    mlp_dim: int
    max_seq_len: int
    clue_embedding: bool        # learned vector added to the embeddings of the anchors
    dtype: str                  # "bfloat16" = mixed precision (params stay fp32)
    flash_attention: bool       # cuDNN fused attention on a GPU; requires dtype="bfloat16"
    architecture: str           # dit | graph_dit | maze_dit
    graph_n: int | None         # graph_dit: vertices per graph
    grid_height: int | None     # maze_dit: board geometry and its encoder
    grid_width: int | None
    grid_encoder_dim: int | None
    grid_encoder_dilations: tuple[tuple[int, int], ...] | None
    grid_pool: tuple[int, ...] | None  # per-block board space-to-depth factors

    @property
    def compute_dtype(self) -> jnp.dtype:
        return jnp.bfloat16 if self.dtype == "bfloat16" else jnp.float32


def precompute_freqs_cis(dim: int, max_seq_len: int):
    inv_freq = 1.0 / (ROPE_THETA ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    freqs = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(freqs), jnp.sin(freqs)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)


def apply_rotary_emb(q, k, cos, sin):
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def modulate(x, shift, scale):
    return x * (1.0 + scale) + shift


# Both initializers read the fan-in off the leading axis.
def kaiming_uniform(key, shape, dtype=jnp.float32):
    fan_in = shape[0]
    bound = math.sqrt(5.0) * math.sqrt(1.0 / fan_in)
    return jax.random.uniform(key, shape, dtype, -bound, bound)


def lecun_normal(key, shape, dtype=jnp.float32):
    fan_in = shape[0]
    std = 1.0 / math.sqrt(fan_in)
    return jax.random.normal(key, shape, dtype) * std


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = jnp.exp(
        -math.log(MAX_PERIOD) * jnp.arange(0, half, dtype=jnp.float32) / half
    )
    args = t[:, None].astype(jnp.float32) * freqs[None]
    return jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)


class LayerNorm(nn.Module):
    """Scale-only LayerNorm, computed in float32."""

    def __init__(self, dim: int, rngs: nn.Rngs):
        self.scale = nn.Param(nn.initializers.ones(rngs.params(), (dim,)))

    def __call__(self, x):
        x = x.astype(jnp.float32)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + LN_EPS)
        return x * self.scale.value


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, rngs: nn.Rngs):
        kernel_init = nn.initializers.lecun_normal()
        bias_init = nn.initializers.zeros
        self.mlp_0 = nn.Linear(FREQUENCY_EMBEDDING_SIZE, hidden_size,
                               kernel_init=kernel_init, bias_init=bias_init, rngs=rngs)
        self.mlp_2 = nn.Linear(hidden_size, hidden_size,
                               kernel_init=kernel_init, bias_init=bias_init, rngs=rngs)

    def __call__(self, t):
        x = self.mlp_0(timestep_embedding(t, FREQUENCY_EMBEDDING_SIZE))
        x = nn.silu(x)
        return self.mlp_2(x)


class EmbeddingLayer(nn.Module):
    def __init__(self, dim: int, vocab_size: int, rngs: nn.Rngs):
        self.embedding = nn.Param(kaiming_uniform(rngs.params(), (vocab_size, dim)))

    def __call__(self, x):
        return self.embedding.value[x]


class DiffusionConditioner(nn.Module):
    """Token and diffusion-time embeddings."""

    def __init__(self, config: DITConfig, rngs: nn.Rngs):
        self.config = config
        self.vocab_embed = EmbeddingLayer(config.hidden_size, config.vocab_size, rngs)
        if config.clue_embedding:
            self.clue_embed = nn.Param(jnp.zeros((config.hidden_size,)))
        self.sigma_map = TimestepEmbedder(config.cond_dim, rngs)

    def __call__(self, tokens: jnp.ndarray, sigma: jnp.ndarray,
                 cond_mask: jnp.ndarray | None = None) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = self.vocab_embed(tokens)
        if self.config.clue_embedding:
            x = x + cond_mask[..., None] * self.clue_embed.value
        return x, nn.silu(self.sigma_map(sigma))


class DDiTBlock(nn.Module):
    """DiT block with adaLN modulation and RoPE; the QKV kernel is one 4D tensor.

    Base for the structured variants: owns the adaLN conditioning, the
    self-attention branch, and the MLP branch."""

    def __init__(self, config: DITConfig, rngs: nn.Rngs, n_adaln: int = 6):
        dim = config.hidden_size
        n_heads = config.n_heads

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.n_adaln = n_adaln
        # bf16 mixed precision: activations and matmuls in compute_dtype, params
        # (and LayerNorm/adaLN/logits) in float32.
        self.compute_dtype = config.compute_dtype
        self.flash_attention = config.flash_attention and jax.default_backend() == "gpu"

        self.adaLN_modulation = nn.Linear(
            config.cond_dim, n_adaln * dim,
            kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros, rngs=rngs,
        )

        self.norm1 = LayerNorm(dim, rngs)
        self.attn_qkv = nn.Param(lecun_normal(rngs.params(), (dim, n_heads, 3, self.head_dim)))
        self.attn_out = nn.Param(lecun_normal(rngs.params(), (n_heads, self.head_dim, dim)))

        self.norm2 = LayerNorm(dim, rngs)
        self.mlp_fc1 = nn.Linear(
            dim, config.mlp_dim,
            dtype=self.compute_dtype,
            kernel_init=nn.initializers.lecun_normal(), bias_init=nn.initializers.zeros,
            rngs=rngs,
        )
        self.mlp_fc2 = nn.Linear(
            config.mlp_dim, dim,
            dtype=self.compute_dtype,
            kernel_init=nn.initializers.lecun_normal(), bias_init=nn.initializers.zeros,
            rngs=rngs,
        )

    def _adaln(self, c):
        out = self.adaLN_modulation(c).astype(self.compute_dtype)
        return jnp.split(out[:, None, :], self.n_adaln, axis=-1)

    def _attention(self, x, shift, scale, gate, cos=None, sin=None, bias=None, mask=None):
        cdt = self.compute_dtype
        x_mod = modulate(self.norm1(x), shift, scale).astype(cdt)
        # cuDNN flash attention needs a GPU, half-precision inputs, no dense mask, and
        # an even sequence length for the backward pass (odd lengths raise); the rest
        # runs XLA attention.
        impl = "cudnn" if (self.flash_attention and mask is None and x.shape[1] % 2 == 0) else None
        qkv = jnp.einsum("bsd,dhqk->bshqk", x_mod, self.attn_qkv.value.astype(cdt))
        q, k, v = qkv[:, :, :, 0], qkv[:, :, :, 1], qkv[:, :, :, 2]
        if cos is not None:
            seq_len = x_mod.shape[1]
            q, k = apply_rotary_emb(q, k, cos[:seq_len].astype(cdt), sin[:seq_len].astype(cdt))
        out = jax.nn.dot_product_attention(q, k, v, bias=bias, mask=mask,
                                           implementation=impl)
        out = jnp.einsum("bshd,hdo->bso", out, self.attn_out.value.astype(cdt))
        return x + gate * out

    def _mlp(self, x, shift, scale, gate):
        h = modulate(self.norm2(x), shift, scale).astype(self.compute_dtype)
        h = self.mlp_fc1(h)
        h = nn.gelu(h, approximate=True)
        h = self.mlp_fc2(h)
        return x + gate * h

    def __call__(self, x, cos, sin, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._adaln(c)
        x = self._attention(x, shift_msa, scale_msa, gate_msa, cos=cos, sin=sin)
        return self._mlp(x, shift_mlp, scale_mlp, gate_mlp)


class DDitFinalLayer(nn.Module):
    def __init__(self, config: DITConfig, rngs: nn.Rngs):
        hidden_size = config.hidden_size
        self.compute_dtype = config.compute_dtype
        self.adaLN_modulation = nn.Linear(
            config.cond_dim, 2 * hidden_size,
            kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros, rngs=rngs,
        )
        self.norm_final = LayerNorm(hidden_size, rngs)
        self.linear = nn.Linear(
            hidden_size, config.head_width,
            kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros, rngs=rngs,
        )

    def __call__(self, x, c):
        shift, scale = jnp.split(self.adaLN_modulation(c)[:, None, :], 2, axis=-1)
        x = modulate(self.norm_final(x), shift, scale).astype(self.compute_dtype)
        # The head matmul runs in the compute dtype like every block matmul,
        # accumulating and emitting fp32 logits; softmax and the losses stay fp32.
        return jnp.einsum("bsd,dv->bsv", x, self.linear.kernel.value.astype(self.compute_dtype),
                          preferred_element_type=jnp.float32) + self.linear.bias.value


class DIT(nn.Module):
    """Sequence denoiser: a stack of DiT blocks over the token embeddings."""

    def __init__(self, config: DITConfig, rngs: nn.Rngs):
        self.config = config
        self.compute_dtype = config.compute_dtype
        self.conditioner = DiffusionConditioner(config, rngs)
        self.rotary_cos, self.rotary_sin = precompute_freqs_cis(
            config.hidden_size // config.n_heads, config.max_seq_len)
        self.blocks = nn.List([DDiTBlock(config, rngs) for _ in range(config.n_blocks)])
        self.output_layer = DDitFinalLayer(config, rngs)

    def __call__(self, input_ids: jnp.ndarray, sigma: jnp.ndarray,
                 attention_mask: jnp.ndarray | None = None,
                 cond_mask: jnp.ndarray | None = None) -> jnp.ndarray:
        x, c = self.conditioner(input_ids, sigma, cond_mask)
        x = x.astype(self.compute_dtype)
        for block in self.blocks:
            x = block(x, self.rotary_cos, self.rotary_sin, c)
        return self.output_layer(x.astype(jnp.float32), c)
