import json
import os

import netket as nk
import numpy as np

from autoresearch_hubbard.hamiltonian import build_hamiltonian
from autoresearch_hubbard.hamiltonian import (
    compute_ed_reference,
    free_fermion_ground_state_energy,
)


def test_build_hamiltonian_returns_operator_and_hilbert_and_graph():
    H, hi, g = build_hamiltonian(L=2, U=1.0, t=1.0, pbc=False)

    assert isinstance(hi, nk.hilbert.SpinOrbitalFermions)
    assert hi.n_orbitals == 4
    assert tuple(hi.n_fermions_per_spin) == (2, 2)
    assert g.n_nodes == 4
    assert H.hilbert is hi
    assert H.is_hermitian


def test_u0_ed_matches_free_fermion_filling():
    """For U=0 the Hubbard model is free fermions; ED must match direct hopping-matrix diagonalization."""
    H, hi, g = build_hamiltonian(L=2, U=0.0, t=1.0, pbc=False)
    e_ff = free_fermion_ground_state_energy(g, hi, t=1.0)
    e_ed = nk.exact.lanczos_ed(H, k=1)[0]
    np.testing.assert_allclose(e_ed, e_ff, atol=1e-10)


def test_compute_ed_reference_caches_to_disk(tmp_path):
    cache = tmp_path / "ed.json"
    H, _, _ = build_hamiltonian(L=2, U=8.0, t=1.0, pbc=False)

    e1 = compute_ed_reference(H, cache_path=str(cache))
    assert cache.exists()
    payload = json.loads(cache.read_text())
    assert abs(payload["energy"] - e1) < 1e-12

    mtime_before = os.path.getmtime(cache)
    e2 = compute_ed_reference(H, cache_path=str(cache))
    assert abs(e1 - e2) < 1e-12
    # Second call should read cache, not recompute.
    assert os.path.getmtime(cache) == mtime_before


def test_compute_ed_reference_uses_matrix_free_lanczos_by_default(tmp_path, monkeypatch):
    cache = tmp_path / "ed_matrix_free.json"
    H, _, _ = build_hamiltonian(L=2, U=8.0, t=1.0, pbc=False)
    calls = []

    def fake_lanczos(operator, k=1, *, matrix_free=False, **kwargs):
        calls.append(
            {
                "operator": operator,
                "k": k,
                "matrix_free": matrix_free,
            }
        )
        return np.asarray([-1.234]), None

    monkeypatch.setattr(nk.exact, "lanczos_ed", fake_lanczos)

    energy = compute_ed_reference(H, cache_path=str(cache))

    assert energy == -1.234
    assert calls == [{"operator": H, "k": 1, "matrix_free": True}]
