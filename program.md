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

Each experiment runs for a fixed wall-clock training budget of **10 minutes**
(`TRIAL_SECONDS = 600` in `frozen-manifest.toml`). JIT compile time is
excluded — see the warmup pattern in `train.py`. Launch it as:

```bash
uv run train.py > run.log 2>&1
```

**What you CAN do:**
- Modify any non-frozen file. The fixed surface is the Hamiltonian and the
  metric (`prepare.py`, `src/autoresearch_hubbard/hamiltonian.py`,
  `frozen-manifest.toml`). Everything else — ansatz family, pretraining
  target (or absence of one), optimizer, sampler, loop structure — is
  mutable. You may add new files (e.g. an alternative ansatz module) as
  long as `train.py` remains the entry point and prints the summary block.

**What you CANNOT do:**
- Modify `prepare.py` — fixed constants, Hamiltonian builder, tripwire.
- Modify `src/autoresearch_hubbard/hamiltonian.py` — the physics target.
- Change the system: 4×4 square, OBC, U=8, t=1, t'=0, half-filling (N↑=N↓=8).
- Install new dependencies.
- Change `TRIAL_SECONDS` (hard-locked at 10 minutes).

**Goal: lowest `final_energy`** — the variational expectation
⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ at the end of the trial. Lower is better. Time budget is
fixed, so bigger isn't always better — faster convergence often wins.

**Simplicity criterion**: all else equal, simpler is better. A tiny improvement
that adds ugly complexity is not worth it; removing code and getting equal or
better results is a simplification win and always kept.

**The wavefunction class itself is fair game, not just hyperparameters.**
Treat the task as "minimize the Rayleigh quotient over any parametric ψ
that fits the budget", not "tune this transformer". The default mirrors
Gu et al. 2025 (transformer-backflow, NNB warm-start, SPRING SR), but at
4×4 with 16 sites, expressivity rarely beats step count — simpler families
often win. If you deviate, justify in `results.tsv`; keep only if
`final_energy` drops or the code shrinks materially. Directions worth
trying:

- **Other ansatz families** (often best at this scale): pure Slater
  determinant (`nk.models.Slater2nd` — Hartree-Fock baseline, ~256 params),
  Slater × Jastrow correlator, MLP-only backflow (drop attention), RBM
  (`nk.models.RBM` — removes the orbital pretraining step entirely).
- **Optimizer family**: plain SR (`momentum=None`), Min-SR (`use_ntk=True,
  on_the_fly=True`), MARCH (paper §3 — implement locally), Adam-only.
- **Within the transformer**: wider/deeper attention, more determinants,
  alternative pretraining targets, different phase-budget schedules.

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
commit	timestamp	final_energy	model	spring_steps	elapsed_sec	status	description
```

1. git commit hash (short, 7 chars)
2. ISO 8601 timestamp from the run (e.g. `2026-04-16T10:30:45`)
3. `final_energy` achieved (e.g. `-7.123456`) — use `99.999999` for crashes
4. model identifier (`MODEL_ID` from `train.py`, e.g. `SiT(d=128,L=4,K=4)`,
   `Slater`, `Slater+Jastrow`, `MLP-backflow`, `RBM`)
5. number of timed SPRING steps (the actual variational training count;
   excludes the warmup step) — `0` for crashes
6. wall-clock elapsed in seconds (training + compile + startup) — `0` for crashes
7. status: `keep`, `discard`, or `crash`
8. short text description of the change

`train.py` prints a pre-filled `tsv_entry:` line at the end of every run.
Copy it, prepend the commit hash, and replace `[STATUS]` / `[DESC]`.

Example:

```
commit	timestamp	final_energy	model	spring_steps	elapsed_sec	status	description
a1b2c3d	2026-04-16T10:30:45	-7.123456	SiT(d=128,L=4,K=4)	16	332.7	keep	baseline
b2c3d4e	2026-04-16T10:38:12	-7.520000	SiT(d=192,L=6,K=4)	8	410.4	keep	wider+deeper transformer
c3d4e5f	2026-04-16T10:46:33	-6.800000	SiT(d=128,L=4,K=4)	18	298.1	discard	drop positional encoding
d4e5f6g	2026-04-16T10:47:00	99.999999	SiT(d=1024,L=4,K=4)	0	0	crash	d_model=1024 (OOM)
e5f6g7h	2026-04-16T10:55:00	-7.890000	Slater+Jastrow	—	120.5	keep	pivot to HF + Jastrow (no NN)
```

Run `uv run analyze.py` (or the `/analyze` skill) at any time to render
`progress.png` from the current `results.tsv`.

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

**Timeout**: each experiment should take ~10 min training + JIT compile
(typically 30 s – 3 min depending on model size). If a run exceeds 20 min
wall-clock, kill it and treat as a failure.

**Crashes**: if a run crashes (NaN, OOM, bug), use judgment. Dumb typos get
fixed and rerun. Fundamentally broken ideas are logged as `crash` and skipped.

**NEVER STOP**: once the loop has begun, do not pause to ask whether to
continue. Do not ask "is this a good stopping point?". The human might be
asleep and expects you to keep iterating. You are autonomous. If you run out
of ideas, re-read the paper in `notes/`, try combining previous near-misses,
or make a more radical architectural change. The loop runs until manually
stopped.
