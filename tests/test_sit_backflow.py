import jax
import jax.numpy as jnp
import netket as nk
import numpy as np

from autoresearch_hubbard.ansatz.sit_backflow import SiTBackflow


def occupied_rows(configs: jax.Array, n_electrons: int) -> jax.Array:
    return jax.vmap(lambda x: jnp.where(x > 0, size=n_electrons)[0])(configs)


def test_site_tokens_follow_paper_local_basis():
    hilbert = nk.hilbert.SpinOrbitalFermions(
        n_orbitals=2, s=1 / 2, n_fermions_per_spin=(1, 1)
    )
    model = SiTBackflow(
        hilbert, d_model=8, n_heads=2, n_layers=1, n_determinants=2
    )
    configs = jnp.asarray([[0, 1, 1, 0]], dtype=jnp.float32)

    params = model.init(jax.random.PRNGKey(0), configs)
    tokens = model.apply(params, configs, method=SiTBackflow.site_tokens)

    np.testing.assert_array_equal(tokens, jnp.asarray([[1, 2]], dtype=jnp.int32))


def test_sit_backflow_orbitals_have_paper_shape():
    hilbert = nk.hilbert.SpinOrbitalFermions(
        n_orbitals=2, s=1 / 2, n_fermions_per_spin=(1, 1)
    )
    configs = jnp.asarray(hilbert.all_states(), dtype=jnp.float32)
    model = SiTBackflow(
        hilbert, d_model=8, n_heads=2, n_layers=1, n_determinants=3
    )

    params = model.init(jax.random.PRNGKey(1), configs)
    orbitals = model.apply(params, configs, method=SiTBackflow.backflow_orbitals)

    assert orbitals.shape == (configs.shape[0], 3, 4, 2)


def test_sit_backflow_logpsi_matches_manual_sum_of_determinants():
    hilbert = nk.hilbert.SpinOrbitalFermions(
        n_orbitals=2, s=1 / 2, n_fermions_per_spin=(1, 1)
    )
    configs = jnp.asarray(hilbert.all_states(), dtype=jnp.float32)
    model = SiTBackflow(
        hilbert, d_model=8, n_heads=2, n_layers=1, n_determinants=2
    )

    params = model.init(jax.random.PRNGKey(2), configs)
    orbitals = model.apply(params, configs, method=SiTBackflow.backflow_orbitals)
    logpsi = model.apply(params, configs)

    row_ids = occupied_rows(configs, hilbert.n_fermions)
    slater = jax.vmap(
        lambda m_sample, rows: jax.vmap(lambda m_k: m_k[rows, :])(m_sample)
    )(orbitals, row_ids)
    manual = jax.vmap(lambda phi: jnp.linalg.det(phi).sum())(slater)

    np.testing.assert_allclose(jnp.exp(logpsi), manual, atol=1e-6, rtol=1e-6)
