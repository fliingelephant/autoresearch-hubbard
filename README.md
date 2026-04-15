# Autoresearch-Hubbard

Paper-faithful transformer-backflow neural quantum state for the 2D Fermi-Hubbard
model (Gu et al. 2025, arXiv 2507.02644), wired into a Karpathy-style
[autoresearch](https://github.com/karpathy/autoresearch) loop. A frozen physics
surface + GitHub CI prevent the autoresearch agent from silently changing the
target while it tunes the model.

## Files

**Frozen (agent read-only, CI-enforced):**

- `frozen-manifest.toml` — authoritative pin for the physics surface: sha256s of the frozen files, the six lattice constants, the `FREE_FERMION_PROBE` value.
- `prepare.py` — loads the manifest, exposes pinned constants, builds the Hamiltonian, runs the three-layer tripwire (`verify_frozen_surface`).
- `src/autoresearch_hubbard/hamiltonian.py` — the physics target: `build_hamiltonian`, `free_fermion_ground_state_energy`, `compute_ed_reference`.
- `tests/test_hamiltonian.py` — physics regression tests.

**Editable (agent mutates freely):**

- `train.py` — NNB warm-start → supervised pretraining → SPRING VMC pipeline. All hyperparameters as module-level constants.
- `src/autoresearch_hubbard/ansatz/nnb.py`, `sit_backflow.py` — warm-start MLP and transformer-backflow ansatz.
- `src/autoresearch_hubbard/pretrain.py` — supervised orbital pretraining utility.

**Operational:**

- `program.md` — runbook the autoresearch agent reads (setup, experiment loop, logging, keep/discard).
- `references/autoresearch/` — upstream Karpathy repo pinned as a submodule for reference.

## Quick start

```bash
uv sync
git submodule update --init --recursive
uv run train.py > run.log 2>&1
grep "^final_energy:" run.log
```

See `program.md` for the full autoresearch experimentation protocol.

## Security model

The agent can only lower energy on the exact pinned physics. Three complementary checks, enforced both locally and in CI:

| Check | What it catches | Local | GitHub CI |
|---|---|---|---|
| **File hashes** — sha256 of every entry in `frozen-manifest.toml` (`prepare.py`, `hamiltonian.py`, `test_hamiltonian.py`) | silent source edits to any frozen file, including the verifier itself | `verify_frozen_surface()` at `train.main()` startup | same check, run by the `physics-freeze` workflow |
| **Manifest self-hash** — sha256 of `frozen-manifest.toml` compared against the `FROZEN_MANIFEST_SHA` repo secret | edits to the manifest (constants, file hashes, probe) — local can't self-verify, so CI is the trust root | — | `physics-freeze` workflow |
| **Physics probe** — free-fermion (U=0) ground-state energy of the pinned lattice | semantic drift the hash-based checks can't see (e.g. upstream NetKet semantics change) | `verify_frozen_surface()` | same |

The manifest SHA secret lives outside the repo, so an agent that edits both a frozen file **and** its manifest entry still fails CI (manifest content changed → SHA changed → mismatch with secret).

Branch protection on `main` requires:
- PR with code-owner review
- `physics-freeze / verify` status check green
- No force-push, no branch deletion

## Development

```bash
uv run pytest tests/ -q        # 20 tests including tripwire regressions
uv run python prepare.py       # print system info and run tripwire locally
```

Changing the physics target (e.g. to a 6×6 lattice for Phase 4) requires
updating `frozen-manifest.toml`, recomputing the `FROZEN_MANIFEST_SHA` secret,
and landing both through a human-reviewed PR.
