"""Neural network backflow initializer from Gu et al. 2025 supplementary §2."""

import flax.linen as nn
import jax
import jax.numpy as jnp

from netket import jax as nkjax
from netket.hilbert import SpinOrbitalFermions
from netket.utils.types import DType


def _flatten_configs(configs: jax.Array) -> tuple[jax.Array, tuple[int, ...]]:
    return configs.reshape((-1, configs.shape[-1])), configs.shape[:-1]


def _occupied_rows(configs: jax.Array, n_electrons: int) -> jax.Array:
    return jax.vmap(lambda x: jnp.where(x > 0, size=n_electrons)[0])(configs)


class NNB(nn.Module):
    """Two-hidden-layer MLP producing a block-diagonal backflow matrix."""

    hilbert: SpinOrbitalFermions
    hidden_dim: int = 256
    param_dtype: DType = jnp.float32

    @nn.compact
    def backflow_orbitals(self, configs: jax.Array) -> jax.Array:
        flat_configs, batch_shape = _flatten_configs(jnp.asarray(configs, self.param_dtype))
        n_sites = self.hilbert.n_orbitals
        n_down, n_up = self.hilbert.n_fermions_per_spin

        x = nn.Dense(self.hidden_dim, param_dtype=self.param_dtype)(flat_configs)
        x = nn.silu(x)
        x = nn.Dense(self.hidden_dim, param_dtype=self.param_dtype)(x)
        x = nn.silu(x)
        x = nn.Dense(n_sites * (n_down + n_up), param_dtype=self.param_dtype)(x)

        down_size = n_sites * n_down
        down = x[:, :down_size].reshape((-1, n_sites, n_down))
        up = x[:, down_size:].reshape((-1, n_sites, n_up))

        zeros_du = jnp.zeros((down.shape[0], n_sites, n_up), dtype=down.dtype)
        zeros_ud = jnp.zeros((up.shape[0], n_sites, n_down), dtype=up.dtype)
        top = jnp.concatenate([down, zeros_du], axis=-1)
        bottom = jnp.concatenate([zeros_ud, up], axis=-1)
        orbitals = jnp.concatenate([top, bottom], axis=1)

        return orbitals.reshape(batch_shape + orbitals.shape[1:])

    def __call__(self, configs: jax.Array) -> jax.Array:
        orbitals = self.backflow_orbitals(configs)
        flat_configs, batch_shape = _flatten_configs(jnp.asarray(configs, self.param_dtype))
        flat_orbitals = orbitals.reshape((-1,) + orbitals.shape[-2:])
        rows = _occupied_rows(flat_configs, self.hilbert.n_fermions)
        slater = jax.vmap(lambda m, idx: m[idx, :])(flat_orbitals, rows)
        logpsi = jax.vmap(nkjax.logdet_cmplx)(slater)
        return logpsi.reshape(batch_shape)
