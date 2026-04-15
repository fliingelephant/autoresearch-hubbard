"""Hubbard Hamiltonian builder for the Phase 1 instance (square, OBC, half-filled)."""

import netket as nk
from netket.graph import AbstractGraph
from netket.hilbert import SpinOrbitalFermions


def build_hamiltonian(L: int = 4, U: float = 8.0, t: float = 1.0, pbc: bool = False):
    """Build the 2D square Hubbard Hamiltonian at half filling.

    Returns (H, hilbert, graph). The Hilbert space is the fixed-particle-number
    sector with N_up = N_down = L*L/2 (L must be even).
    """
    n_sites = L * L
    n_per_spin = n_sites // 2
    g: AbstractGraph = nk.graph.Hypercube(length=L, n_dim=2, pbc=pbc)
    hi = SpinOrbitalFermions(
        n_orbitals=n_sites, s=1 / 2, n_fermions_per_spin=(n_per_spin, n_per_spin)
    )
    H = nk.operator.FermiHubbardJax(hi, graph=g, t=t, U=U)
    return H, hi, g
