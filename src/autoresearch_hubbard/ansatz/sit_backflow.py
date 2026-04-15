"""Transformer backflow ansatz from Gu et al. 2025 supplementary §1.1."""

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


class TransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    param_dtype: DType = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            param_dtype=self.param_dtype,
        )(x)
        x = x + y
        y = nn.Dense(self.d_model, param_dtype=self.param_dtype)(x)
        y = nn.silu(y)
        return x + y


class SiTBackflow(nn.Module):
    """Site-token transformer producing a sum of Slater determinants."""

    hilbert: SpinOrbitalFermions
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    n_determinants: int = 4
    param_dtype: DType = jnp.float32

    def site_tokens(self, configs: jax.Array) -> jax.Array:
        flat_configs, batch_shape = _flatten_configs(jnp.asarray(configs, self.param_dtype))
        n_sites = self.hilbert.n_orbitals
        down = flat_configs[:, :n_sites].astype(jnp.int32)
        up = flat_configs[:, n_sites:].astype(jnp.int32)
        tokens = up + 2 * down
        return tokens.reshape(batch_shape + (n_sites,))

    @nn.compact
    def backflow_orbitals(self, configs: jax.Array) -> jax.Array:
        tokens = self.site_tokens(configs)
        flat_tokens = tokens.reshape((-1, tokens.shape[-1]))
        n_sites = self.hilbert.n_orbitals
        n_electrons = self.hilbert.n_fermions

        embed = nn.Embed(4, self.d_model, param_dtype=self.param_dtype)
        x = embed(flat_tokens)
        positional = self.param(
            "positional_encoding",
            nn.initializers.normal(stddev=1.0),
            (n_sites, self.d_model),
            self.param_dtype,
        )
        x = x + positional[None, :, :]

        for _ in range(self.n_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                param_dtype=self.param_dtype,
            )(x)

        x = nn.Dense(
            self.n_determinants * 2 * n_electrons, param_dtype=self.param_dtype
        )(x)
        x = x.reshape((-1, n_sites, self.n_determinants, 2, n_electrons))
        x = x.transpose(0, 2, 3, 1, 4)
        down = x[:, :, 0, :, :]
        up = x[:, :, 1, :, :]
        orbitals = jnp.concatenate([down, up], axis=2)

        return orbitals.reshape(tokens.shape[:-1] + orbitals.shape[1:])

    def __call__(self, configs: jax.Array) -> jax.Array:
        orbitals = self.backflow_orbitals(configs)
        flat_configs, batch_shape = _flatten_configs(jnp.asarray(configs, self.param_dtype))
        flat_orbitals = orbitals.reshape((-1,) + orbitals.shape[-3:])
        rows = _occupied_rows(flat_configs, self.hilbert.n_fermions)
        slater = jax.vmap(
            lambda m_sample, idx: jax.vmap(lambda m_k: m_k[idx, :])(m_sample)
        )(flat_orbitals, rows)
        logdets = jax.vmap(jax.vmap(nkjax.logdet_cmplx))(slater)
        logpsi = nkjax.logsumexp_cplx(logdets, axis=1)
        return logpsi.reshape(batch_shape)
