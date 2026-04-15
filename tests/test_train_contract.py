import hashlib
from pathlib import Path

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

    assert prepare.TRIAL_SECONDS == 300
    assert prepare.LATTICE_L == 4
    assert prepare.LATTICE_PBC is False
    assert prepare.U == 8.0
    assert prepare.T == 1.0
    assert prepare.T_PRIME == 0.0


def test_prepare_build_system_returns_hamiltonian_triple():
    import netket as nk

    import prepare

    H, hi, g = prepare.build_system()
    assert isinstance(hi, nk.hilbert.SpinOrbitalFermions)
    assert hi.n_orbitals == prepare.LATTICE_L ** 2
    assert tuple(hi.n_fermions_per_spin) == (
        prepare.LATTICE_L ** 2 // 2,
        prepare.LATTICE_L ** 2 // 2,
    )
    assert g.n_nodes == prepare.LATTICE_L ** 2
    assert H.hilbert is hi


# ---------------------------------------------------------------------------
# train.py contract helpers
# ---------------------------------------------------------------------------


def test_allocate_phase_seconds_splits_budget_cleanly():
    import train

    budgets = train.allocate_phase_seconds(
        total_seconds=100.0,
        nnb_fraction=0.25,
        pretrain_fraction=0.15,
    )

    assert budgets == {"nnb": 25.0, "pretrain": 15.0, "spring": 60.0}


def test_allocate_phase_seconds_rejects_overfull_budgets():
    import train

    with pytest.raises(ValueError):
        train.allocate_phase_seconds(
            total_seconds=10.0,
            nnb_fraction=0.8,
            pretrain_fraction=0.3,
        )


def test_format_summary_lines_matches_grep_contract():
    import train

    lines = train.format_summary_lines(
        {
            "final_energy": -7.25,
            "min_energy": -7.5,
            "elapsed_seconds": 12.5,
            "nnb_steps": 3,
            "pretrain_steps": 2,
            "spring_steps": 4,
        }
    )

    assert lines[0] == "---"
    assert "final_energy:    -7.250000" in lines
    assert "min_energy:      -7.500000" in lines
    assert "elapsed_seconds: 12.5" in lines
    assert "nnb_steps:       3" in lines
    assert "pretrain_steps:  2" in lines
    assert "spring_steps:    4" in lines
