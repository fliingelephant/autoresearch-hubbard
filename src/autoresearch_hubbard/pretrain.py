"""Supervised orbital pretraining utilities.

Paper reference: Gu et al. 2025, Eq. S28. The NNB targets are computed
from the supplied configs at every call, so callers can drive the loop
with fresh MCMC samples (paper §S.2, online pretraining).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import optax


def orbital_mse_loss(params, sit_model, nnb_variables, nnb_model, configs):
    """MSE between SiT backflow orbitals and NNB targets on the same configs."""
    predictions = sit_model.apply(params, configs, method=sit_model.backflow_orbitals)
    targets = nnb_model.apply(nnb_variables, configs, method=nnb_model.backflow_orbitals)
    if predictions.ndim == targets.ndim + 1:
        targets = targets[:, None, :, :]
    return jnp.mean(jnp.square(predictions - targets))


@partial(jax.jit, static_argnames=("sit_model", "nnb_model", "optimizer"))
def supervised_pretrain_step(
    params,
    sit_model,
    nnb_variables,
    nnb_model,
    configs: jax.Array,
    *,
    optimizer: optax.GradientTransformation,
    opt_state,
):
    loss, grads = jax.value_and_grad(orbital_mse_loss)(
        params, sit_model, nnb_variables, nnb_model, configs,
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss
