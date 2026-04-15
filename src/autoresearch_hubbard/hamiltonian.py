"""Hubbard Hamiltonian builder for the Phase 1 instance (square, OBC, half-filled)."""

import netket as nk
import numpy as np
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


def free_fermion_ground_state_energy(graph, hilbert, t: float = 1.0) -> float:
    """Ground-state energy of non-interacting fermions (U=0) on the given graph.

    Computed as twice (two spins) the sum of the n_per_spin lowest eigenvalues of
    the single-particle hopping matrix -t·A, where A is the adjacency matrix.
    """
    n_sites = graph.n_nodes
    A = np.zeros((n_sites, n_sites))
    for i, j in graph.edges():
        A[i, j] = 1.0
        A[j, i] = 1.0
    H1 = -t * A
    eps = np.linalg.eigvalsh(H1)
    n_up, n_dn = hilbert.n_fermions_per_spin
    return float(eps[:n_up].sum() + eps[:n_dn].sum())
