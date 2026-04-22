import netket as nk
import numpy as np

from autoresearch_hubbard.hamiltonian import (
    build_hamiltonian,
    free_fermion_ground_state_energy,
)


def test_build_hamiltonian_returns_operator_and_hilbert_and_graph():
    H, hi, g = build_hamiltonian(Lx=2, Ly=2, U=1.0, t=1.0, pbc=False)

    assert isinstance(hi, nk.hilbert.SpinOrbitalFermions)
    assert hi.n_orbitals == 4
    assert tuple(hi.n_fermions_per_spin) == (2, 2)
    assert g.n_nodes == 4
    assert H.hilbert is hi
    assert H.is_hermitian


def test_rectangular_lattice():
    """16x4 OBC gives 64 sites, 32 per spin, 108 NN edges under Grid."""
    _, hi, g = build_hamiltonian(Lx=16, Ly=4, U=8.0, t=1.0, pbc=False)
    assert hi.n_orbitals == 64
    assert tuple(hi.n_fermions_per_spin) == (32, 32)
    assert g.n_nodes == 64
    # Edge count: (Lx-1)*Ly + Lx*(Ly-1) = 15*4 + 16*3 = 108
    assert g.n_edges == 108


def test_per_direction_pbc():
    """Cylinder: PBC in y (short axis), OBC in x (long axis)."""
    _, _, g = build_hamiltonian(Lx=4, Ly=4, U=1.0, t=1.0, pbc=[False, True])
    # OBC x: 3*4 = 12 NN edges. PBC y wraps: 4*4 = 16 edges. Total = 28.
    assert g.n_edges == 28


def test_u0_ed_matches_free_fermion_filling():
    """For U=0 the Hubbard model is free fermions; ED must match direct hopping-matrix diagonalization."""
    H, hi, g = build_hamiltonian(Lx=2, Ly=2, U=0.0, t=1.0, pbc=False)
    e_ff = free_fermion_ground_state_energy(g, hi, t=1.0)
    e_ed = nk.exact.lanczos_ed(H, k=1)[0]
    np.testing.assert_allclose(e_ed, e_ff, atol=1e-10)
