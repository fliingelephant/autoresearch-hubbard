# autoresearch-hubbard

An LLM-driven research loop on the 2D Fermi-Hubbard model. Each trial trains a
neural quantum state on a fixed 4×4 half-filled instance and scores the final
variational energy. Lower is better.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr16`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current `main`.
3. **Read the background material** (required — the paper postdates most
   training cutoffs, and the baseline code is a direct replication):
   - `notes/Gu et al. - 2025 - Solving the Hubbard model with Neural Quantum States.pdf` —
     the paper this pipeline replicates. Key: the SiT transformer-backflow
     architecture (supp. §1.1), the NNB warm-start + supervised orbital
     pretraining procedure (supp. §2), the MARCH optimizer (§3, SPRING +
     Adam-style second-moment normalization — not yet implemented here,
     good candidate for an experiment), and the reference energies for
     4×4 OBC at U=8 half-filling (use these as convergence sanity).
   - `docs/superpowers/specs/2026-04-15-autoresearch-hubbard-design.md` —
     this repo's Phase 1 scope and rationale for each architectural choice.
   - `references/autoresearch/program.md` (submodule) — the upstream
     autoresearch protocol this runbook mirrors.
4. **Read the in-scope files**:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, Hamiltonian builder, tripwire. Do not modify.
   - `frozen-manifest.toml` — authoritative physics pin. Do not modify.
   - `train.py` — the top-level file you modify. Pipeline, optimizer, schedules, hyperparameters.
   - `src/autoresearch_hubbard/ansatz/nnb.py` — NNB warm-start ansatz. Editable.
   - `src/autoresearch_hubbard/ansatz/sit_backflow.py` — transformer backflow ansatz. Editable.
   - `src/autoresearch_hubbard/pretrain.py` — supervised orbital pretraining. Editable.
5. **Initialize `results.tsv`**: create with only the header row (see below). The baseline is recorded after the first run.
6. **Confirm and go**.

## Experimentation

Each experiment runs for a fixed wall-clock training budget of **30 minutes**
defined by `TRIAL_SECONDS` in `prepare.py`. Launch it as:

```bash
uv run train.py > run.log 2>&1
```

**What you CAN do:**
- Modify `train.py` — full pipeline: architecture wiring, optimizer, schedules, phase splits, logging cadence.
- Modify `src/autoresearch_hubbard/ansatz/nnb.py` — NNB warm-start ansatz.
- Modify `src/autoresearch_hubbard/ansatz/sit_backflow.py` — transformer backflow ansatz.
- Modify `src/autoresearch_hubbard/pretrain.py` — supervised pretraining procedure.

**What you CANNOT do:**
- Modify `prepare.py` — fixed constants, Hamiltonian builder, metric rules.
- Modify `src/autoresearch_hubbard/hamiltonian.py` — the physics target.
- Change the system: 4×4 square, OBC, U=8, t=1, t'=0, half-filling (N↑=N↓=8).
- Install new dependencies.
- Change `TRIAL_SECONDS` (hard-locked at 30 minutes).

**Goal: lowest `final_energy`.** Time budget is fixed, so bigger is not always
better — faster convergence often wins.

**Simplicity criterion**: all else equal, simpler is better. A tiny improvement
that adds ugly complexity is not worth it; removing code and getting equal or
better results is a simplification win and always kept.

**Paper-faithfulness is the starting point, not a constraint.** The default
ansatz mirrors Gu et al. 2025 (no LayerNorm, SiLU FFN, K-determinant sum,
real params feeding a complex head). If you deviate, justify it in the
`results.tsv` description and keep only if `final_energy` drops. Good
directions to explore (grounded in the paper or its references):
MARCH optimizer (paper §3), alternative pretraining targets, different
phase-budget schedules, wider/deeper transformer, more determinants.

**The first run**: establish the baseline by running the training script as is.

## Output format

The script prints a summary block at the end:

```
---
final_energy:    -9.123456
min_energy:      -9.234567
elapsed_seconds: 1801.3
nnb_steps:       120
pretrain_steps:  450
spring_steps:    960
```

Extract the key metric with:

```bash
grep "^final_energy:" run.log
```

## Logging results

Log each experiment to `results.tsv` (tab-separated, NOT comma — commas break
in descriptions). The file is gitignored; do not commit it.

Columns:

```
commit	final_energy	status	description
```

1. git commit hash (short, 7 chars)
2. `final_energy` achieved (e.g. `-7.123456`) — use `99.999999` for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of the change

Example:

```
commit	final_energy	status	description
a1b2c3d	-7.123456	keep	baseline
b2c3d4e	-7.520000	keep	n_layers 4->6, d_model 128->192
c3d4e5f	-6.800000	discard	drop positional encoding
d4e5f6g	99.999999	crash	d_model=1024 (OOM)
```

## The experiment loop

Runs on a dedicated branch (e.g. `autoresearch/apr16`).

LOOP FOREVER:

1. Look at the git state: current branch/commit.
2. Tune the editable files with an experimental idea.
3. `git commit -am "<short description>"`
4. Run: `uv run train.py > run.log 2>&1`
5. Read out the results: `grep "^final_energy:\|^min_energy:\|^elapsed_seconds:" run.log`
6. If grep is empty, the run crashed. `tail -n 50 run.log` to read the stack trace. If trivially fixable, fix and rerun; otherwise log `crash` and move on.
7. Record the results in `results.tsv`.
8. If `final_energy` improved (more negative), keep the commit and advance the branch.
9. Otherwise, `git reset --hard HEAD~1` and try a different idea.

**Timeout**: each experiment should take ~30 min training + a few seconds of
startup / compilation. If a run exceeds 40 min wall-clock, kill it and treat
as a failure.

**Crashes**: if a run crashes (NaN, OOM, bug), use judgment. Dumb typos get
fixed and rerun. Fundamentally broken ideas are logged as `crash` and skipped.

**NEVER STOP**: once the loop has begun, do not pause to ask whether to
continue. Do not ask "is this a good stopping point?". The human might be
asleep and expects you to keep iterating. You are autonomous. If you run out
of ideas, re-read the paper in `notes/`, try combining previous near-misses,
or make a more radical architectural change. The loop runs until manually
stopped.
