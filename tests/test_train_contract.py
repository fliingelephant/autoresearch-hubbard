import hashlib

import pytest


# ---------------------------------------------------------------------------
# Frozen-surface tripwire
# ---------------------------------------------------------------------------


def test_verify_frozen_surface_passes_on_pristine_tree():
    import prepare

    prepare.verify_frozen_surface()  # must not raise


def test_manifest_lists_every_frozen_file_with_correct_sha():
    import prepare

    for rel_path, expected_sha in prepare._manifest["hashes"].items():
        full = prepare._REPO_ROOT / rel_path
        observed = hashlib.sha256(full.read_bytes()).hexdigest()
        assert observed == expected_sha, rel_path


def test_verify_frozen_surface_detects_frozen_file_drift(tmp_path, monkeypatch):
    import prepare

    # Swap the manifest's expected hash to something wrong.
    bad_manifest = {
        "hashes": {list(prepare._manifest["hashes"].keys())[0]: "0" * 64},
        "invariants": prepare._manifest["invariants"],
    }
    monkeypatch.setattr(prepare, "_manifest", bad_manifest)
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        prepare.verify_frozen_surface()


def test_verify_frozen_surface_detects_probe_drift(monkeypatch):
    import prepare

    monkeypatch.setattr(prepare, "FREE_FERMION_PROBE", 0.0)
    with pytest.raises(RuntimeError, match="free-fermion"):
        prepare.verify_frozen_surface()


def test_manifest_sha256_is_reproducible():
    import prepare

    h1 = prepare.manifest_sha256()
    h2 = hashlib.sha256(prepare._MANIFEST_PATH.read_bytes()).hexdigest()
    assert h1 == h2
    assert len(h1) == 64


# ---------------------------------------------------------------------------
# Invariant exposure
# ---------------------------------------------------------------------------


def test_prepare_exports_fixed_constants():
    import prepare

    assert prepare.TRIAL_SECONDS == 600
    assert prepare.LATTICE_LX == 16
    assert prepare.LATTICE_LY == 4
    assert prepare.LATTICE_PBC is False
    assert prepare.U == 8.0
    assert prepare.T == 1.0
    assert prepare.T_PRIME == 0.0


def test_prepare_build_system_returns_hamiltonian_triple():
    import netket as nk

    import prepare

    H, hi, g = prepare.build_system()
    n_sites = prepare.LATTICE_LX * prepare.LATTICE_LY
    assert isinstance(hi, nk.hilbert.SpinOrbitalFermions)
    assert hi.n_orbitals == n_sites
    assert tuple(hi.n_fermions_per_spin) == (n_sites // 2, n_sites // 2)
    assert g.n_nodes == n_sites
    assert H.hilbert is hi


