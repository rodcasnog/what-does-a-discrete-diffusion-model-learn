"""AdamW, or Muon on the hidden matrices."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from omegaconf import DictConfig

_NS_COEFFS = (3.4445, -4.7750, 2.0315)  # Muon's quintic Newton-Schulz iteration


def _newton_schulz(g: jnp.ndarray) -> jnp.ndarray:
    """Approximate orthogonalization of a 2D matrix via the quintic NS iteration.
    Runs in bfloat16 (the iteration is robust to low precision)."""
    a, b, c = _NS_COEFFS
    steps, eps = 5, 1e-7
    x = g.astype(jnp.bfloat16)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (jnp.linalg.norm(x) + eps)
    for _ in range(steps):
        xxt = x @ x.T
        x = a * x + (b * xxt + c * (xxt @ xxt)) @ x
    if transposed:
        x = x.T
    return x.astype(g.dtype)


def _orth(path, u):
    """NS orthogonalization of the logical input/output matrix, RMS-matched by
    ``0.2 * sqrt(max(m, n))``; Q/K/V are input-first, the other weights output-last."""
    path_s = _path_str(path)
    input_projection = any(name in path_s for name in ("attn_qkv", "cross_q", "cross_kv"))
    arranged = jnp.moveaxis(u, 0, -1) if input_projection else u
    mat = arranged.reshape(-1, arranged.shape[-1])
    o = _newton_schulz(mat)
    scale = 0.2 * jnp.sqrt(jnp.maximum(mat.shape[0], mat.shape[1])).astype(u.dtype)
    out = (o * scale).reshape(arranged.shape)
    return jnp.moveaxis(out, -1, 0) if input_projection else out


def scale_by_muon(momentum: float) -> optax.GradientTransformation:
    """Nesterov momentum, then ``_orth`` on every parameter it is given."""
    return optax.chain(optax.trace(decay=momentum, nesterov=True),
                       optax.stateless(lambda u, _: jax.tree_util.tree_map_with_path(_orth, u)))


def _path_str(path) -> str:
    return "/".join(str(getattr(k, "key", getattr(k, "name", k))) for k in path).lower()


def param_labels(params) -> optax.Params:
    """Route non-embedding matrix weights to Muon and everything else to AdamW.
    Routing reads module names, so renaming a module re-routes its weights."""
    def label(path, p):
        hidden = not any(name in _path_str(path) for name in ("embed", "output_layer"))
        return "muon" if getattr(p, "ndim", 0) >= 2 and hidden else "adamw"
    return jax.tree_util.tree_map_with_path(label, params)


def decay_mask(params) -> optax.Params:
    """Weight-decay mask: decay >=2D non-embedding weights only."""
    def mask(path, p):
        return getattr(p, "ndim", 0) >= 2 and "embed" not in _path_str(path)
    return jax.tree_util.tree_map_with_path(mask, params)


def build_optimizer(t: DictConfig, lr_schedule: optax.Schedule) -> optax.GradientTransformation:
    """Global-norm clipping, then AdamW, or Muon on the hidden matrices and AdamW on the
    rest, with decoupled weight decay under the decay mask and the shared lr schedule."""
    b1, b2, wd = t.adam_b1, t.adam_b2, t.weight_decay
    if t.optimizer == "adamw":
        inner = optax.adamw(learning_rate=lr_schedule, b1=b1, b2=b2, weight_decay=wd,
                            mask=decay_mask)
    elif t.optimizer == "muon":
        inner = optax.chain(
            optax.multi_transform({"muon": scale_by_muon(t.muon_momentum),
                                   "adamw": optax.scale_by_adam(b1=b1, b2=b2)}, param_labels),
            optax.add_decayed_weights(wd, mask=decay_mask),
            optax.scale_by_learning_rate(lr_schedule))
    else:
        raise ValueError(f"unknown optimizer {t.optimizer!r}; use adamw or muon")
    return optax.chain(optax.clip_by_global_norm(t.max_grad_norm), inner)
