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
