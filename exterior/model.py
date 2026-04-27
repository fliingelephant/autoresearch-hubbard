"""Backflow-like exterior amplitude evaluator."""

from __future__ import annotations

import math

import flax.linen as nn
import jax
import jax.numpy as jnp
from netket.hilbert import SpinOrbitalFermions
from netket.utils.types import DType

from exterior.algebra import basis_blades, interior, scalar_pair, wedge


def _flatten_configs(configs: jax.Array) -> tuple[jax.Array, tuple[int, ...]]:
    return configs.reshape((-1, configs.shape[-1])), configs.shape[:-1]


def _occupied_modes(configs: jax.Array, n_electrons: int) -> jax.Array:
    return jax.vmap(lambda x: jnp.where(x > 0, size=n_electrons)[0])(configs)


def _gather_rows(features: jax.Array, rows: jax.Array) -> jax.Array:
    return jax.vmap(lambda sample, idx: sample[idx])(features, rows)


def _vacuum_init(_key: jax.Array, shape: tuple[int, ...], dtype: DType) -> jax.Array:
    values = jnp.zeros(shape, dtype)
    return values.at[:, 0].set(1.0)


def _log_scalar_cmplx(amplitude: jax.Array) -> jax.Array:
    complex_dtype = jnp.promote_types(amplitude.dtype, jnp.complex64)
    return jnp.log(amplitude.astype(complex_dtype))


class ExteriorLinear(nn.Module):
    """Channel mixer applied independently to each stored blade."""

    features: int
    use_bias: bool = False
    param_dtype: DType = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        blades_first = jnp.moveaxis(x, -2, -1)
        y = nn.Dense(
            self.features,
            use_bias=self.use_bias,
            param_dtype=self.param_dtype,
        )(blades_first)
        return jnp.moveaxis(y, -1, -2)


class ExteriorAttention(nn.Module):
    ext_dim: int
    ext_channels: int
    param_dtype: DType = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        q = ExteriorLinear(self.ext_channels, param_dtype=self.param_dtype)(x)
        k = ExteriorLinear(self.ext_channels, param_dtype=self.param_dtype)(x)
        v = ExteriorLinear(self.ext_channels, param_dtype=self.param_dtype)(x)
        logits = jnp.einsum("btcn,bscn->bts", q, k)
        logits = logits / math.sqrt(self.ext_channels * len(basis_blades(self.ext_dim)))
        weights = nn.softmax(logits, axis=-1)
        return jnp.einsum("bts,bscn->btcn", weights, v)


class ExteriorBilinearFFN(nn.Module):
    ext_dim: int
    ext_channels: int
    param_dtype: DType = jnp.float32

    @nn.compact
    def __call__(self, scalar_tokens: jax.Array, ext_tokens: jax.Array) -> jax.Array:
        u = ExteriorLinear(self.ext_channels, param_dtype=self.param_dtype)(ext_tokens)
        v = ExteriorLinear(self.ext_channels, param_dtype=self.param_dtype)(ext_tokens)
        wedge_update = wedge(u, v, ext_dim=self.ext_dim)
        contraction_update = interior(u, v, ext_dim=self.ext_dim)
        mixed = jnp.concatenate(
            [ext_tokens, wedge_update, contraction_update],
            axis=-2,
        )
        update = ExteriorLinear(self.ext_channels, param_dtype=self.param_dtype)(mixed)
        gate = nn.Dense(self.ext_channels, param_dtype=self.param_dtype)(scalar_tokens)
        return nn.sigmoid(gate)[..., None] * update


class ExteriorReducer(nn.Module):
    ext_dim: int
    ext_channels: int
    param_dtype: DType = jnp.float32

    @nn.compact
    def __call__(self, accumulator: jax.Array, token: jax.Array) -> jax.Array:
        u = ExteriorLinear(self.ext_channels, param_dtype=self.param_dtype)(accumulator)
        v = ExteriorLinear(self.ext_channels, param_dtype=self.param_dtype)(token)
        wedge_update = wedge(u, v, ext_dim=self.ext_dim)
        contraction_update = interior(u, v, ext_dim=self.ext_dim)
        mixed = jnp.concatenate(
            [accumulator, token, wedge_update, contraction_update],
            axis=-2,
        )
        update = ExteriorLinear(self.ext_channels, param_dtype=self.param_dtype)(mixed)
        gate_input = jnp.concatenate([accumulator[..., 0], token[..., 0]], axis=-1)
        gate = nn.Dense(self.ext_channels, param_dtype=self.param_dtype)(gate_input)
        return accumulator + nn.sigmoid(gate)[..., None] * update


class ExteriorBlock(nn.Module):
    d_model: int
    n_heads: int
    ext_dim: int
    ext_channels: int
    param_dtype: DType = jnp.float32

    @nn.compact
    def __call__(
        self,
        scalar_tokens: jax.Array,
        ext_tokens: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        scalar_update = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            out_features=self.d_model,
            param_dtype=self.param_dtype,
        )(scalar_tokens)
        scalar_tokens = scalar_tokens + scalar_update
        scalar_tokens = scalar_tokens + nn.silu(
            nn.Dense(self.d_model, param_dtype=self.param_dtype)(scalar_tokens)
        )

        ext_tokens = ext_tokens + ExteriorAttention(
            ext_dim=self.ext_dim,
            ext_channels=self.ext_channels,
            param_dtype=self.param_dtype,
        )(ext_tokens)
        ext_tokens = ext_tokens + ExteriorBilinearFFN(
            ext_dim=self.ext_dim,
            ext_channels=self.ext_channels,
            param_dtype=self.param_dtype,
        )(scalar_tokens, ext_tokens)
        return scalar_tokens, ext_tokens


class ExteriorAmplitude(nn.Module):
    """Configuration-conditioned exterior circuit returning complex log amplitudes."""

    hilbert: SpinOrbitalFermions
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    ext_dim: int = 4
    ext_channels: int = 8
    param_dtype: DType = jnp.float32

    @property
    def n_sites(self) -> int:
        return self.hilbert.n_orbitals

    @property
    def n_blades(self) -> int:
        return len(basis_blades(self.ext_dim))

    def site_tokens(self, configs: jax.Array) -> jax.Array:
        flat_configs, batch_shape = _flatten_configs(jnp.asarray(configs, self.param_dtype))
        down = flat_configs[:, : self.n_sites].astype(jnp.int32)
        up = flat_configs[:, self.n_sites :].astype(jnp.int32)
        tokens = up + 2 * down
        return tokens.reshape(batch_shape + (self.n_sites,))

    @nn.compact
    def __call__(self, configs: jax.Array) -> jax.Array:
        flat_configs, batch_shape = _flatten_configs(jnp.asarray(configs, self.param_dtype))
        site_tokens = self.site_tokens(flat_configs)
        flat_site_tokens = site_tokens.reshape((-1, self.n_sites))

        scalar_tokens = nn.Embed(4, self.d_model, param_dtype=self.param_dtype)(flat_site_tokens)
        positional = self.param(
            "mode_encoding",
            nn.initializers.normal(stddev=0.02),
            (self.n_sites, self.d_model),
            self.param_dtype,
        )
        scalar_tokens = scalar_tokens + positional[None, :, :]

        ext_tokens = nn.Dense(
            self.ext_channels * self.n_blades,
            param_dtype=self.param_dtype,
        )(scalar_tokens)
        ext_tokens = ext_tokens.reshape(
            flat_configs.shape[0], self.n_sites, self.ext_channels, self.n_blades
        )

        for _ in range(self.n_layers):
            scalar_tokens, ext_tokens = ExteriorBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                ext_dim=self.ext_dim,
                ext_channels=self.ext_channels,
                param_dtype=self.param_dtype,
            )(scalar_tokens, ext_tokens)

        n_down, n_up = self.hilbert.n_fermions_per_spin
        down = flat_configs[:, : self.n_sites]
        up = flat_configs[:, self.n_sites :]
        down_rows = _occupied_modes(down, n_down)
        up_rows = _occupied_modes(up, n_up)
        spin_modes = ExteriorLinear(
            2 * self.ext_channels,
            param_dtype=self.param_dtype,
        )(ext_tokens)
        spin_modes = spin_modes.reshape(
            flat_configs.shape[0],
            self.n_sites,
            2,
            self.ext_channels,
            self.n_blades,
        )
        down_occupied = _gather_rows(spin_modes[:, :, 0, :, :], down_rows)
        up_occupied = _gather_rows(spin_modes[:, :, 1, :, :], up_rows)

        vacuum = self.param(
            "vacuum",
            _vacuum_init,
            (self.ext_channels, self.n_blades),
            self.param_dtype,
        )
        accumulator = jnp.broadcast_to(
            vacuum[None, :, :],
            (flat_configs.shape[0], self.ext_channels, self.n_blades),
        )
        reducer = ExteriorReducer(
            ext_dim=self.ext_dim,
            ext_channels=self.ext_channels,
            param_dtype=self.param_dtype,
        )
        for mode_id in range(n_down):
            accumulator = reducer(accumulator, down_occupied[:, mode_id, :, :])
        for mode_id in range(n_up):
            accumulator = reducer(accumulator, up_occupied[:, mode_id, :, :])

        readout_form = self.param(
            "readout_form",
            _vacuum_init,
            (self.ext_channels, self.n_blades),
            self.param_dtype,
        )
        amplitude = jnp.sum(scalar_pair(accumulator, readout_form), axis=-1)
        amplitude = amplitude / self.ext_channels
        logpsi = _log_scalar_cmplx(amplitude)
        return logpsi.reshape(batch_shape)
