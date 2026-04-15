"""NNB initialization and supervised orbital pretraining utilities."""

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
    learning_rate: float = 3e-4,
    optimizer: optax.GradientTransformation | None = None,
    opt_state=None,
):
    tx = optax.adam(learning_rate) if optimizer is None else optimizer
    state = tx.init(params) if opt_state is None else opt_state

    loss_fn = lambda p: orbital_mse_loss(p, model, configs, targets)
    _, grads = jax.value_and_grad(loss_fn)(params)
    updates, state = tx.update(grads, state, params)
    params = optax.apply_updates(params, updates)
    loss_after = loss_fn(params)
    return params, state, loss_after


def run_supervised_pretraining(
    params,
    model,
    configs: jax.Array,
    targets: jax.Array,
    *,
    n_steps: int = 5000,
    learning_rate: float = 3e-4,
):
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    losses = []

    for _ in range(n_steps):
        params, opt_state, loss = supervised_pretrain_step(
            params,
            model,
            configs,
            targets,
            optimizer=optimizer,
            opt_state=opt_state,
        )
        losses.append(float(loss))

    return params, losses
