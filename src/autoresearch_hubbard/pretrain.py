"""Supervised orbital pretraining utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax


def orbital_mse_loss(params, model, configs: jax.Array, targets: jax.Array) -> jax.Array:
    orbitals = model.apply(params, configs, method=model.backflow_orbitals)
    if orbitals.ndim == targets.ndim + 1:
        targets = targets[:, None, :, :]
    return jnp.mean(jnp.square(orbitals - targets))


def supervised_pretrain_step(
    params,
    model,
    configs: jax.Array,
    targets: jax.Array,
    *,
    optimizer: optax.GradientTransformation,
    opt_state,
):
    loss, grads = jax.value_and_grad(
        lambda p: orbital_mse_loss(p, model, configs, targets)
    )(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss
