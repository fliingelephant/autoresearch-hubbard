# autoresearch-hubbard

A toy for studying autonomous ML research loops. An LLM agent edits a
variational neural quantum state for the 2D Hubbard model on a tiny 4×4
instance. A frozen physics surface, pinned in `frozen-manifest.toml` and
enforced by GitHub Actions, keeps the agent from quietly moving the goalposts
while it mutates hyperparameters, the ansatz, or the training loop.

Not a physics contribution. The baseline is a direct reproduction of
Gu et al. 2025 (arXiv 2507.02644) at the smallest instance the paper covers.

## What the experiment does

- **Fixed instance**: 4×4 square lattice, open boundary conditions, half-filled
  (N↑ = N↓ = 8), U = 8, t = 1, t' = 0.
- **Per-trial budget**: `TRIAL_SECONDS = 300` of wall-clock training.
- **Metric**: `final_energy` — the variational expectation ⟨ψ|H|ψ⟩ at end of
  trial. Lower is better. Extracted from stdout via `grep "^final_energy:"`.
- **Baseline pipeline** (`train.py`): NNB warm-start → supervised orbital
  pretraining → SPRING stochastic reconfiguration with a transformer-backflow
  ansatz (SiT, K-determinant head).
- **Loop** (`program.md`): agent edits editable files, commits, runs, keeps
  commits that lower `final_energy`, reverts otherwise. Logs every attempt
  (including crashes) to `results.tsv`.

## File map

Frozen (agent cannot modify; CI rejects drift):

- `frozen-manifest.toml` — single source of truth. File hashes, lattice
  constants, physics probe value.
- `prepare.py` — loads the manifest, exposes the fixed constants,
  runs `verify_frozen_surface()` on `train.main()` startup.
- `src/autoresearch_hubbard/hamiltonian.py` — Hamiltonian builder, ED helper,
  free-fermion reference.
- `tests/test_hamiltonian.py` — physics regression tests.

Editable (agent mutates freely):

- `train.py` — pipeline, optimizer, schedules, all hyperparameters as
  module-level constants.
- `src/autoresearch_hubbard/ansatz/nnb.py` — MLP warm-start ansatz.
- `src/autoresearch_hubbard/ansatz/sit_backflow.py` — transformer-backflow
  ansatz with determinant head.
- `src/autoresearch_hubbard/pretrain.py` — supervised orbital pretraining.

Operational:

- `program.md` — the agent's runbook (setup, loop, logging, keep/discard).
- `references/autoresearch/` — upstream Karpathy repo pinned as a submodule.

## Quick start

```bash
uv sync
git submodule update --init --recursive
uv run train.py > run.log 2>&1
grep "^final_energy:" run.log
```

## Enforcement model

Three independent checks, each run in two places. The manifest SHA secret
lives outside the repo, so an agent who edits both a frozen file and its
manifest entry still fails CI.

| Check | What it catches | Local (`verify_frozen_surface`) | GitHub Actions (`physics-freeze`) |
|---|---|---|---|
| `sha256(<frozen file>)` for every entry in `frozen-manifest.toml` — currently `prepare.py`, `hamiltonian.py`, `test_hamiltonian.py` | byte-level edits to any frozen file, including the verifier itself | yes | yes |
| `sha256(frozen-manifest.toml)` compared to the `FROZEN_MANIFEST_SHA` repo secret | edits to the manifest (constants, file hashes, probe) | no — manifest is the local source of truth | yes |
| `FREE_FERMION_PROBE` — pre-computed U=0 ground-state energy of the pinned lattice | lattice / filling / hopping drift that leaves source bytes unchanged (e.g. upstream NetKet semantic changes) | yes | yes |

Branch rules on `main`:
- PR required, `physics-freeze / verify` must be green.
- Code-owner review required (CODEOWNERS → @fliingelephant).
- No force-push, no branch deletion.
- Admin (solo owner) has bypass for direct pushes during setup; the rule
  still fires on any non-admin contributor.

## Changing the physics surface

Phases beyond the current one (6×6, PBC, doped, t' ≠ 0) require a deliberate
update:

1. Edit `frozen-manifest.toml` (file hashes, constants, probe value).
2. Recompute `sha256(frozen-manifest.toml)`.
3. Rotate the `FROZEN_MANIFEST_SHA` repo secret.
4. Land all of the above through a reviewed PR.

## Development

```bash
uv run pytest tests/ -q        # 20 tests, includes tripwire regressions
uv run python prepare.py       # print system info and run tripwire locally
```
