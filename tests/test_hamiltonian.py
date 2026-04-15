import netket as nk
import pytest
from autoresearch_hubbard.hamiltonian import build_hamiltonian


def test_build_hamiltonian_returns_operator_and_hilbert_and_graph():
    H, hi, g = build_hamiltonian(L=2, U=1.0, t=1.0, pbc=False)

    assert isinstance(hi, nk.hilbert.SpinOrbitalFermions)
    assert hi.n_orbitals == 4
    assert tuple(hi.n_fermions_per_spin) == (2, 2)
    assert g.n_nodes == 4
    assert H.hilbert is hi
    assert H.is_hermitian


import numpy as np
from autoresearch_hubbard.hamiltonian import free_fermion_ground_state_energy


def test_u0_ed_matches_free_fermion_filling():
    """For U=0 the Hubbard model is free fermions; ED must match direct hopping-matrix diagonalization."""
    H, hi, g = build_hamiltonian(L=2, U=0.0, t=1.0, pbc=False)
    e_ff = free_fermion_ground_state_energy(g, hi, t=1.0)
    e_ed = nk.exact.lanczos_ed(H, k=1)[0]
    np.testing.assert_allclose(e_ed, e_ff, atol=1e-10)
