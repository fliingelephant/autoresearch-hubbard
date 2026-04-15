import jax
import jax.numpy as jnp
import netket as nk
import numpy as np

from autoresearch_hubbard.ansatz.nnb import NNB


def occupied_rows(configs: jax.Array, n_electrons: int) -> jax.Array:
    return jax.vmap(lambda x: jnp.where(x > 0, size=n_electrons)[0])(configs)


def test_nnb_backflow_orbitals_are_block_diagonal():
    hilbert = nk.hilbert.SpinOrbitalFermions(
        n_orbitals=2, s=1 / 2, n_fermions_per_spin=(1, 1)
    )
    configs = jnp.asarray(hilbert.all_states(), dtype=jnp.float32)
    model = NNB(hilbert, hidden_dim=8)

    params = model.init(jax.random.PRNGKey(0), configs)
    orbitals = model.apply(params, configs, method=NNB.backflow_orbitals)

    assert orbitals.shape == (configs.shape[0], 4, 2)
    np.testing.assert_allclose(orbitals[:, :2, 1], 0.0, atol=1e-6)
    np.testing.assert_allclose(orbitals[:, 2:, 0], 0.0, atol=1e-6)


def test_nnb_logpsi_matches_manual_determinant():
    hilbert = nk.hilbert.SpinOrbitalFermions(
        n_orbitals=2, s=1 / 2, n_fermions_per_spin=(1, 1)
    )
    configs = jnp.asarray(hilbert.all_states(), dtype=jnp.float32)
    model = NNB(hilbert, hidden_dim=8)

    params = model.init(jax.random.PRNGKey(1), configs)
    orbitals = model.apply(params, configs, method=NNB.backflow_orbitals)
    logpsi = model.apply(params, configs)

    row_ids = occupied_rows(configs, hilbert.n_fermions)
    slater = jax.vmap(lambda m, rows: m[rows, :])(orbitals, row_ids)
    manual = jax.vmap(jnp.linalg.det)(slater)

    np.testing.assert_allclose(jnp.exp(logpsi), manual, atol=1e-6, rtol=1e-6)
