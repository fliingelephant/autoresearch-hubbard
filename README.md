# autoresearch-hubbard

A toy for studying autonomous ML research loops. An LLM agent edits a
variational neural quantum state for the 2D Hubbard model on a tiny 4×4
instance. A frozen physics surface, pinned in `frozen-manifest.toml` and
enforced by GitHub Actions, keeps the agent from quietly moving the goalposts
while it mutates hyperparameters, the ansatz, or the training loop.

Not a physics contribution. The baseline is a direct reproduction of
Gu et al. 2025 (arXiv 2507.02644) at the smallest instance the paper covers.

## Quick start

From a cloned repo, open Claude Code:

```bash
cd Autoresearch-Hubbard
claude   # or your agent of choice
```

For an unattended run on a server, launch with
`claude --permission-mode bypassPermissions` so the agent doesn't stop on
edit/bash prompts. Claude refuses bypass mode under `root` — on a fresh
cloud GPU box, create a non-root user first (and reinstall `uv` / `claude`
for them if they aren't on that user's `PATH`):

```bash
useradd -m -s /bin/bash agent
chown -R agent:agent /workspace/autoresearch-hubbard
su - agent
cd /workspace/autoresearch-hubbard
claude --permission-mode bypassPermissions
```

In the session:

```
/setup
/config set autoCompactThreshold 20
```

Then prompt:

> Read program.md and start a new autoresearch experiment.

`/setup` installs `uv` (if missing), detects CUDA via `nvidia-smi`, and runs
`uv sync --extra cuda` or `uv sync --extra cpu` accordingly. The auto-compact
threshold keeps a long iteration history before compaction kicks in.

The agent takes over from `program.md`: proposes a branch tag
(`autoresearch/<tag>`), initializes `results.tsv`, runs the baseline, then
loops edit → commit → train → keep/discard until you Ctrl+C. Each iteration
is ~5–10 min wall-clock. Per `program.md`'s `NEVER STOP` rule, the agent
will not pause to ask — you are the kill switch.

After (or during) a run, `/analyze` renders `progress.png` — a scatter of
all experiments with discard/keep colors, a running-min step line, and
annotations on kept commits.

## What the experiment does

- **System**: 4×4 square lattice, OBC, half-filled (N↑=N↓=8), U=8, t=1, t'=0.
- **Budget**: `TRIAL_SECONDS = 300` wall-clock training per trial (JIT excluded).
- **Metric**: `final_energy` — `⟨ψ|H|ψ⟩/⟨ψ|ψ⟩` at end of trial. Lower is better.
- **Baseline** (`train.py`): NNB warm-start → supervised orbital pretraining →
  SPRING SR with a transformer-backflow ansatz (SiT, K-determinant head).
- **Loop** (`program.md`): edit → commit → run → keep if `final_energy` dropped
  else `git reset --hard HEAD~1`. All attempts logged to `results.tsv`.

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
- `references/autoresearch/` — upstream Karpathy repo as a submodule
  (reference reading; populate with `git submodule update --init` if needed).

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

1. Edit the relevant frozen file(s) — `prepare.py`, `hamiltonian.py`,
   `test_hamiltonian.py`, or the physics constants inside `frozen-manifest.toml`.
2. Run `/rehash` — updates the `[hashes]` block in the manifest and prints the
   new `FROZEN_MANIFEST_SHA` along with the exact `gh secret set` command.
3. Rotate the `FROZEN_MANIFEST_SHA` repo secret as instructed.
4. Land all of the above through a reviewed PR.
