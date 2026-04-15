# Autoresearch-Hubbard

Paper-faithful transformer-backflow neural quantum state for the 2D Fermi-Hubbard model,
wired into a Karpathy-style autoresearch loop.

## Files

- `prepare.py` — **frozen**: loads `frozen-manifest.toml` (constants, file hashes, physics probe), exposes the Hamiltonian builder, runs the frozen-surface tripwire.
- `frozen-manifest.toml` — **frozen**: authoritative source for pinned physics (mirrored in CI via `FROZEN_MANIFEST_SHA` secret).
- `train.py` — **editable**: agent-mutable training pipeline (NNB → supervised pretraining → SPRING VMC).
- `program.md` — operational runbook for the autoresearch agent.
- `src/autoresearch_hubbard/hamiltonian.py` — **frozen**: physics target.
- `src/autoresearch_hubbard/ansatz/{nnb,sit_backflow}.py` — **editable**: ansatz definitions.
- `src/autoresearch_hubbard/pretrain.py` — **editable**: supervised orbital pretraining.

## Quick start

```bash
uv sync
uv run train.py > run.log 2>&1
grep "^final_energy:" run.log
```

See `program.md` for the full experimentation protocol.
