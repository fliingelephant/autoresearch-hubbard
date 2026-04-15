import jax
import jax.numpy as jnp
import netket as nk
import optax

from autoresearch_hubbard.ansatz.nnb import NNB
from autoresearch_hubbard.ansatz.sit_backflow import SiTBackflow
from autoresearch_hubbard.pretrain import orbital_mse_loss, supervised_pretrain_step


def test_supervised_pretrain_step_reduces_orbital_mse():
    hilbert = nk.hilbert.SpinOrbitalFermions(
        n_orbitals=2, s=1 / 2, n_fermions_per_spin=(1, 1)
    )
    configs = jnp.asarray(hilbert.all_states(), dtype=jnp.float32)

    nnb = NNB(hilbert, hidden_dim=8)
    nnb_params = nnb.init(jax.random.PRNGKey(0), configs)
    targets = nnb.apply(nnb_params, configs, method=NNB.backflow_orbitals)

    model = SiTBackflow(hilbert, d_model=8, n_heads=2, n_layers=1, n_determinants=2)
    params = model.init(jax.random.PRNGKey(1), configs)
    optimizer = optax.adam(3e-4)
    opt_state = optimizer.init(params)

    loss_before = orbital_mse_loss(params, model, configs, targets)
    new_params, _, loss = supervised_pretrain_step(
        params, model, configs, targets, optimizer=optimizer, opt_state=opt_state,
    )
    loss_after = orbital_mse_loss(new_params, model, configs, targets)

    assert loss <= loss_before
    assert loss_after <= loss_before
    assert jax.tree_util.tree_structure(new_params) == jax.tree_util.tree_structure(params)
