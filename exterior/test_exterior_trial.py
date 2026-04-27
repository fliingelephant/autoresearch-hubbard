import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import optax

from exterior import model as exterior_model
from exterior.algebra import blade_index, basis_blades, interior, wedge
from exterior.model import ExteriorAmplitude


def test_wedge_has_fermionic_signs_for_basis_vectors():
    blades = basis_blades(3)
    e0 = jnp.zeros((len(blades),)).at[blade_index(blades, (0,))].set(1.0)
    e1 = jnp.zeros((len(blades),)).at[blade_index(blades, (1,))].set(1.0)

    e0_wedge_e1 = wedge(e0, e1, ext_dim=3)
    e1_wedge_e0 = wedge(e1, e0, ext_dim=3)
    e0_wedge_e0 = wedge(e0, e0, ext_dim=3)

    bivector = blade_index(blades, (0, 1))
    assert e0_wedge_e1[bivector] == 1.0
    assert e1_wedge_e0[bivector] == -1.0
    np.testing.assert_allclose(e0_wedge_e0, jnp.zeros_like(e0_wedge_e0))


def test_interior_contracts_vectors_against_bivectors():
    blades = basis_blades(3)
    e0 = jnp.zeros((len(blades),)).at[blade_index(blades, (0,))].set(1.0)
    e1 = jnp.zeros((len(blades),)).at[blade_index(blades, (1,))].set(1.0)
    e01 = jnp.zeros((len(blades),)).at[blade_index(blades, (0, 1))].set(1.0)

    np.testing.assert_allclose(interior(e0, e01, ext_dim=3), e1)
    np.testing.assert_allclose(interior(e1, e01, ext_dim=3), -e0)


def test_log_scalar_cmplx_encodes_negative_sign_as_phase():
    log_values = exterior_model._log_scalar_cmplx(jnp.asarray([-2.0, 3.0]))

    np.testing.assert_allclose(jnp.real(log_values), jnp.log(jnp.asarray([2.0, 3.0])))
    np.testing.assert_allclose(jnp.imag(log_values), jnp.asarray([jnp.pi, 0.0]), rtol=1e-6)


def test_exterior_model_uses_sit_style_site_tokens():
    hilbert = nk.hilbert.SpinOrbitalFermions(
        n_orbitals=2, s=1 / 2, n_fermions_per_spin=(1, 1)
    )
    model = ExteriorAmplitude(
        hilbert, d_model=8, n_heads=2, n_layers=1, ext_dim=3, ext_channels=4
    )
    configs = jnp.asarray([[0, 1, 1, 0]], dtype=jnp.float32)
    variables = model.init(jax.random.PRNGKey(0), configs)
    tokens = model.apply(variables, configs, method=ExteriorAmplitude.site_tokens)

    np.testing.assert_array_equal(tokens, jnp.asarray([[1, 2]], dtype=jnp.int32))


def test_exterior_model_readout_is_a_learned_multivector_pairing():
    hilbert = nk.hilbert.SpinOrbitalFermions(
        n_orbitals=2, s=1 / 2, n_fermions_per_spin=(1, 1)
    )
    configs = jnp.asarray(hilbert.all_states(), dtype=jnp.float32)
    model = ExteriorAmplitude(
        hilbert, d_model=8, n_heads=2, n_layers=1, ext_dim=3, ext_channels=4
    )

    variables = model.init(jax.random.PRNGKey(0), configs)

    assert variables["params"]["readout_form"].shape == (model.ext_channels, model.n_blades)
    assert "spin_tags" not in variables["params"]


def test_exterior_model_returns_finite_batch_log_amplitudes():
    hilbert = nk.hilbert.SpinOrbitalFermions(
        n_orbitals=2, s=1 / 2, n_fermions_per_spin=(1, 1)
    )
    configs = jnp.asarray(hilbert.all_states(), dtype=jnp.float32)
    model = ExteriorAmplitude(
        hilbert, d_model=8, n_heads=2, n_layers=1, ext_dim=3, ext_channels=4
    )

    variables = model.init(jax.random.PRNGKey(0), configs)
    logpsi = model.apply(variables, configs)

    assert logpsi.shape == (configs.shape[0],)
    assert jnp.all(jnp.isfinite(logpsi))


def test_exterior_model_gradients_are_finite():
    hilbert = nk.hilbert.SpinOrbitalFermions(
        n_orbitals=2, s=1 / 2, n_fermions_per_spin=(1, 1)
    )
    configs = jnp.asarray(hilbert.all_states(), dtype=jnp.float32)
    model = ExteriorAmplitude(
        hilbert, d_model=8, n_heads=2, n_layers=1, ext_dim=3, ext_channels=4
    )
    variables = model.init(jax.random.PRNGKey(1), configs)

    def loss_fn(params):
        logpsi = model.apply({"params": params}, configs)
        return jnp.mean(jnp.square(jnp.real(logpsi)) + jnp.square(jnp.imag(logpsi)))

    loss, grads = jax.value_and_grad(loss_fn)(variables["params"])
    leaves = jax.tree_util.tree_leaves(grads)

    assert jnp.isfinite(loss)
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)


def test_exterior_model_can_be_optimized_on_tiny_supervised_target():
    hilbert = nk.hilbert.SpinOrbitalFermions(
        n_orbitals=2, s=1 / 2, n_fermions_per_spin=(1, 1)
    )
    configs = jnp.asarray(hilbert.all_states(), dtype=jnp.float32)
    target = configs[:, 0] - configs[:, 1] + 0.5 * configs[:, 2]
    model = ExteriorAmplitude(
        hilbert, d_model=12, n_heads=2, n_layers=1, ext_dim=3, ext_channels=4
    )
    variables = model.init(jax.random.PRNGKey(2), configs)
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(variables["params"])

    def loss_fn(params):
        prediction = jnp.real(model.apply({"params": params}, configs))
        return jnp.mean(jnp.square(prediction - target))

    params = variables["params"]
    initial_loss = loss_fn(params)
    for _ in range(25):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    assert loss_fn(params) < initial_loss
