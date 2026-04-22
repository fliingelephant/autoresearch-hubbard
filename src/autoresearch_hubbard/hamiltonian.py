"""Hubbard Hamiltonian builder for Phase 1 instances (rectangular lattices, mixed BC, half filling)."""

import netket as nk
import numpy as np
from netket.graph import AbstractGraph
from netket.hilbert import SpinOrbitalFermions


def build_hamiltonian(
    Lx: int,
    Ly: int,
    U: float = 8.0,
    t: float = 1.0,
    pbc=False,
):
    """Build the 2D square-lattice Hubbard Hamiltonian at half filling.

    Parameters
    ----------
    Lx, Ly:
        Lattice extents. Total sites = Lx * Ly. Must give an even site count
        (half filling requires N_up = N_down = Lx*Ly/2).
    U, t:
        On-site repulsion and nearest-neighbor hopping amplitudes.
    pbc:
        ``bool`` for uniform BC in both directions, or a length-2 sequence
        ``[pbc_x, pbc_y]`` for per-direction control (cylinders).

    Returns
    -------
    (H, hilbert, graph) — all via NetKet.
    """
    pbc_list = [bool(pbc), bool(pbc)] if isinstance(pbc, bool) else [bool(pbc[0]), bool(pbc[1])]

    n_sites = Lx * Ly
    n_per_spin = n_sites // 2
    g: AbstractGraph = nk.graph.Grid([Lx, Ly], pbc=pbc_list)
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
