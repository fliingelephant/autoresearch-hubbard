"""
Frozen runtime utilities for autoresearch-hubbard experiments.

Constants, the Hamiltonian builder, and the frozen-surface tripwire live here.
The single source of truth for all pinned values is ``frozen-manifest.toml``.
Do not modify this file directly; modify the manifest under human review.
"""

import os

# Shared perf plumbing pinned here — does not affect numerical results,
# only wall-clock. Numerical precision (x64 vs x32) is a trial knob and
# lives in train.py.
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "jax_cache")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "-1")
os.environ.setdefault("JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES", "all")
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "xla_gpu_force_compilation_parallelism" not in _xla_flags:
    os.environ["XLA_FLAGS"] = f"{_xla_flags} --xla_gpu_force_compilation_parallelism=16".strip()

import hashlib
import tomllib
from pathlib import Path

from autoresearch_hubbard.hamiltonian import (
    build_hamiltonian,
    free_fermion_ground_state_energy,
)

# ---------------------------------------------------------------------------
# Load the frozen manifest (authoritative source for constants + hashes)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent
_MANIFEST_PATH = _REPO_ROOT / "frozen-manifest.toml"


def _load_manifest() -> dict:
    with open(_MANIFEST_PATH, "rb") as f:
        return tomllib.load(f)


_manifest = _load_manifest()
_invariants = _manifest["invariants"]

# Expose invariants as module attributes for train.py / callers.
TRIAL_SECONDS: int = _invariants["TRIAL_SECONDS"]
LATTICE_LX: int = _invariants["LATTICE_LX"]
LATTICE_LY: int = _invariants["LATTICE_LY"]
LATTICE_PBC: bool = _invariants["LATTICE_PBC"]
U: float = _invariants["U"]
T: float = _invariants["T"]
T_PRIME: float = _invariants["T_PRIME"]

FREE_FERMION_PROBE: float = _invariants["FREE_FERMION_PROBE"]
PROBE_TOL: float = _invariants["PROBE_TOL"]


# ---------------------------------------------------------------------------
# System builder
# ---------------------------------------------------------------------------


def build_system():
    """Return (H, hilbert, graph) for the frozen Hubbard instance."""
    return build_hamiltonian(Lx=LATTICE_LX, Ly=LATTICE_LY, U=U, t=T, pbc=LATTICE_PBC)


# ---------------------------------------------------------------------------
# Frozen-surface tripwire
# ---------------------------------------------------------------------------


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_sha256() -> str:
    """SHA-256 of the manifest file itself. This is the value CI checks
    against a repo secret — the trust root outside the editable tree."""
    return _file_sha256(_MANIFEST_PATH)


def verify_frozen_surface() -> None:
    """Raise if any frozen file has drifted or the physics probe disagrees.

    Does NOT verify the manifest itself against an external secret — that
    check lives in CI. Local enforcement stops silent edits of listed files.
    """
    for rel_path, expected_sha in _manifest["hashes"].items():
        full = _REPO_ROOT / rel_path
        observed = _file_sha256(full)
        if observed != expected_sha:
            raise RuntimeError(
                f"Frozen-surface drift: {rel_path} sha256 mismatch.\n"
                f"  expected: {expected_sha}\n"
                f"  observed: {observed}"
            )

    _, hi, g = build_system()
    e_free = free_fermion_ground_state_energy(g, hi, t=T)
    if abs(e_free - FREE_FERMION_PROBE) > PROBE_TOL:
        raise RuntimeError(
            "Frozen-surface drift: free-fermion physics probe has changed.\n"
            f"  expected: {FREE_FERMION_PROBE}\n"
            f"  observed: {e_free}"
        )


__all__ = [
    "TRIAL_SECONDS",
    "LATTICE_LX",
    "LATTICE_LY",
    "LATTICE_PBC",
    "U",
    "T",
    "T_PRIME",
    "FREE_FERMION_PROBE",
    "PROBE_TOL",
    "build_system",
    "build_hamiltonian",
    "free_fermion_ground_state_energy",
    "manifest_sha256",
    "verify_frozen_surface",
]


if __name__ == "__main__":
    verify_frozen_surface()
    H, hi, g = build_system()
    bc_desc = "PBC" if LATTICE_PBC else "OBC"
    print(
        f"System: {LATTICE_LX}x{LATTICE_LY} {bc_desc} "
        f"U={U} t={T} t'={T_PRIME} half-filled"
    )
    print(f"Hilbert: {hi.n_orbitals} orbitals, n_fermions_per_spin={hi.n_fermions_per_spin}")
    print(f"Graph: {g.n_nodes} nodes, {g.n_edges} edges")
    print(f"Trial budget: {TRIAL_SECONDS}s")
    print(f"Frozen-manifest SHA: {manifest_sha256()}")
    print("Frozen-surface tripwire: OK")
