# Autoresearch-Hubbard — Design Spec

**Date:** 2026-04-15
**Status:** Design, Phase 1 scope.

## 1. Goal

Replicate the methodology of Gu et al. 2025, *"Solving the Hubbard model with Neural Quantum States"* (arXiv 2507.02644) on a small 2D Hubbard instance using NetKet, then wrap the pipeline in a Karpathy-style [autoresearch](https://github.com/karpathy/autoresearch) harness that lets an AI agent iteratively mutate the training code and keep edits that lower the variational energy.

**Phase 1 target (this spec):** 4×4 OBC, half-filled (N↑ = N↓ = 8), U = 8, t = 1, t' = 0. Paper-faithful transformer-backflow ansatz. NetKet's built-in SPRING optimizer. Benchmarked against exact diagonalization.

**Later phases (design must not foreclose):**
- Phase 2 — MARCH optimizer (implement locally, API-compatible with a future NetKet PR).
- Phase 3 — PBC, doped δ = 1/8, t' = −0.2.
- Phase 4 — scaling to 6×6 / 8×8.
- Phase 5 — optional upstream MARCH PR.

## 2. Physics setup

- **Hamiltonian:** `H = −t Σ_{⟨ij⟩,σ} c†_{iσ} c_{jσ} + U Σ_i n_{i↑} n_{i↓}`, with t = 1, U = 8, t' = 0.
- **Lattice:** `nk.graph.Hypercube(length=4, n_dim=2, pbc=False)` — 16 sites, 24 NN edges.
- **Hilbert space:** `nk.hilbert.SpinOrbitalFermions(n_orbitals=16, s=1/2, n_fermions_per_spin=(8, 8))` — fixed particle-number sector.
- **Hamiltonian constructor:** `nk.experimental.operator.FermiHubbardJax(hi, t=1.0, U=8.0, graph=g)`. This implementation is particle-and-spin-conserving and JAX-friendly. It is NN-only; Phase 3 (t' ≠ 0) will extend the graph with NNN edges + a per-edge `t` sequence, or fall back to `ParticleNumberAndSpinConservingFermioperator2nd` constructed by hand.
- **Sampler:** `nk.sampler.MetropolisFermionHop(hi, graph=g, n_chains=…, sweep_size=…, spin_symmetric=True)` — hops a fermion along a graph edge; `spin_symmetric=True` duplicates the graph for each spin sector so both sectors can move independently and the particle count per spin is conserved.
- **ED reference:** `nk.exact.lanczos_ed(H, k=1)` — sparse Lanczos; for 4×4 half-filled the Slater space has `C(16, 8)² ≈ 1.65e8` configurations, which the sparse operator can handle. Result `E_ED_ref` is cached to disk once and reused.

## 3. Ansatz — paper-faithful SiT-backflow

A single Flax module `SiTBackflow`.

```
Input:      n ∈ {0,1}^{2·N_sites}         occupation, (N_sites·2,) flattened as up-concat-down
              ↓ per-site linear embedding  (each site gets a unique embedding vector of dim d_model)
Tokens:     x ∈ (N_sites, d_model)
              ↓ N_layers × Transformer block:
                  y = x + MHA(LN(x))       attention: softmax(QK^T / √d_H) V,
                                           d_H = d_model / n_heads, n_heads heads
                  x = y + FFN(LN(y))       FFN: d_model → mlp_ratio · d_model → d_model, GELU
Features:   f ∈ (N_sites, d_model)
              ↓ Linear head (per site):    d_model → N_electrons
Orbitals:   M(n) ∈ (N_sites, N_electrons)  (complex-valued)
              ↓ for each spin σ, take rows of M corresponding to the N_σ occupied sites of that spin
Submatrix:  M_σ(n) ∈ (N_σ, N_σ)
              ↓ determinants
ψ(n) = det M_↑(n) · det M_↓(n)
log ψ(n) as returned by the module (using sign-log-det to avoid overflow)
```

**Design notes:**
- Tokens are **per-site**, not patch-based. At 4×4 the sequence length is 16 — patching gains nothing.
- Orbital head outputs an `(N_sites, N_electrons)` matrix per spin. The determinant structure mirrors `netket.models.Slater2nd` with `generalized=False, restricted=False` (unrestricted), except the orbitals depend on `n` through the transformer — the "backflow" relation.
- Output is complex; NetKet's SR pipeline supports `mode="complex"` on `VMC_SR`.
- Hyperparameters `d_model`, `n_heads`, `n_layers`, `mlp_ratio` to be read from the paper's supplementary. If the supplementary is ambiguous, use the smallest setting reported and let the autoresearch harness tune.
- No defensive parameter checks inside the module; failures crash loudly (per the `Let it crash` rule).

## 4. Optimizer

**Phase 1 — SPRING** (no new code).
```python
opt = nk.optimizer.Sgd(learning_rate=...)
gs  = nk.driver.VMC_SR(
        H, opt, variational_state=vs,
        diag_shift=..., momentum=0.8,     # momentum != None → SPRING branch
        mode="complex",
      )
```
Min-SR variant available via `use_ntk=True` if needed for larger parameter counts. Plain SR via `momentum=None`.

**Phase 2 — MARCH** (separate spec, separate branch).
MARCH = SPRING + Adam-style second-moment normalization. Paper, p.7: *"MARCH enhances SPRING by also incorporating an estimate of the second moment of the gradients... adapting the learning rate for each parameter individually."*

Implementation plan for Phase 2 (sketch only):
- New branch in `netket._src.ngd.sr_srt_common._sr_srt_common`, gated by a `second_moment` kwarg.
- Extra pytree buffer `old_second_moment` alongside `old_updates`.
- Extra hyperparameter β₂.
- Per-parameter update rescale by `1 / (√v̂ + ε)`.
- API-compatible with `VMC_SR` so a future upstream PR is a file move.

## 5. Autoresearch harness (Phase 1, step 1.4)

Three files, mirroring `karpathy/autoresearch`:

- **`prepare.py`** (frozen across trials):
  - `build_hamiltonian()` → `(H, graph, hilbert)`.
  - `build_sampler(hilbert, graph, **config)` factory.
  - `compute_ed_reference(H)` (cached on disk).
  - `score(log_file) → float` — returns `min(E_trace) − E_ED_ref`.
  - `TRIAL_SECONDS` wall-clock budget constant (calibrated once; expected 120–300 s on CPU for 4×4).
  - Log schema (iteration, mean energy, variance, timestamp).

- **`train.py`** (agent-editable):
  - Imports `prepare`.
  - Defines the `SiTBackflow` module with all its hyperparameters inline.
  - Constructs `MCState`, optimizer, `VMC_SR` driver.
  - Runs VMC until `TRIAL_SECONDS` elapsed.
  - Writes the log.

- **`program.md`** (agent instructions):
  - Task: minimize the final value of `score`.
  - Allowed to modify: everything in `train.py` — architecture, optimizer config, sampler config, schedules, chunking.
  - Forbidden to modify: `prepare.py`, the metric definition, the wall-clock budget, the Hamiltonian, the (N↑, N↓) sector.
  - Trial workflow: edit `train.py` → run → read `score` → commit or revert.

## 6. Directory layout

```
Autoresearch-Hubbard/
├── notes/                              # existing (paper PDF)
├── docs/superpowers/specs/             # this spec
├── pyproject.toml                      # uv-managed
├── uv.lock
├── src/autoresearch_hubbard/
│   ├── __init__.py
│   ├── hamiltonian.py                  # build_hamiltonian(), ED reference
│   ├── ansatz/
│   │   └── sit_backflow.py             # SiTBackflow Flax module
│   ├── prepare.py                      # harness: frozen pieces (step 1.4)
│   ├── train.py                        # harness: agent-editable (step 1.4)
│   └── program.md                      # harness: agent instructions (step 1.4)
├── scripts/
│   └── train_oneoff.py                 # step 1.3 one-shot training script
├── runs/                               # per-trial logs (gitignored)
└── tests/
    ├── test_hamiltonian.py             # step 1.1
    └── test_sit_backflow.py            # step 1.2
```

Phase 2 adds `src/autoresearch_hubbard/optimizer/march.py` and `tests/test_march.py`.

## 7. Phase 1 stepwise plan

### Step 1.1 — Hamiltonian correctness

**Deliverable:** `src/autoresearch_hubbard/hamiltonian.py` exposing `build_hamiltonian()` + `compute_ed_reference()`, with `tests/test_hamiltonian.py` green.

**Checks:**
1. **U = 0 check.** Build `H|_{U=0}`; diagonalize the 16×16 single-particle hopping matrix directly; fill the 8 lowest single-particle eigenvalues per spin to get the free-fermion ground-state energy. Compare to `lanczos_ed(H|_{U=0})`. Must agree to numerical precision.
2. **Small-U perturbative check.** For small ε, `E_ED(U=ε) − E_free ≈ ε · ⟨Σ_i n_{i↑} n_{i↓}⟩_free`. The ⟨·⟩_free expectation is computable from the free-fermion Slater determinant. Linear-in-ε slope must match.
3. **Full U = 8 ED.** Run `lanczos_ed(H, k=1)`; store `E_ED_ref` to disk (`runs/ed_reference.json`).

### Step 1.2 — SiT-backflow model correctness

**Deliverable:** `src/autoresearch_hubbard/ansatz/sit_backflow.py` + `tests/test_sit_backflow.py` green.

**Checks:**
1. **Shape.** `SiTBackflow.apply(params, n_batch)` with `n_batch: (B, 2·N_sites)` returns `(B,)` complex scalar (log-amplitude).
2. **Antisymmetry.** For a random sampled `n` in the (8, 8) sector, swapping two occupied sites of the same spin flips the sign of `ψ(n) = exp(log ψ(n))`. This is the fermionic-sign gate — if this fails, the orbital-determinant wiring is wrong.
3. **Gradient flow.** `jax.grad(log|ψ|²)` w.r.t. params runs without NaN at random init on a minibatch of 32 configs.
4. **Dtype.** Output dtype is complex when `param_dtype` is complex.

Hyperparameters in tests: minimal setting (e.g. `d_model=16, n_heads=2, n_layers=1`) — tests verify structural correctness, not physics accuracy.

### Step 1.3 — Plain VMC end-to-end

**Deliverable:** `scripts/train_oneoff.py` — one reproducible run that achieves `E_var − E_ED_ref < 1e-3` on Phase 1's 4×4 instance.

- Wires `build_hamiltonian()` + `MetropolisFermionHop` + `SiTBackflow` + `VMC_SR(momentum=0.8, diag_shift=…)`.
- Loop is simple and manually tuned — no harness, no agent, no time budget.
- This step's purpose is proving the pipeline converges. No unit tests (experiment script, per the `No trivial TDD` rule).
- If 1e-3 is not attained, the blocker lives in 1.1 or 1.2 — debug by swapping `SiTBackflow` for `nk.models.Slater2nd` (plain Hartree-Fock, no NN). HF on 4×4 half-filled U=8 converges to a known energy and provides a debugging anchor.

### Step 1.4 — Autoresearch harness

**Deliverable:** `src/autoresearch_hubbard/{prepare,train,program}.*` per Section 5; an agent-driven run that produces at least one trial whose metric beats the seed configuration.

- Re-factor `scripts/train_oneoff.py` into `prepare.py` (frozen) + `train.py` (mutable).
- Calibrate `TRIAL_SECONDS` so one trial on CPU fits cleanly into a budget that still allows meaningful convergence (expected 120–300 s; set by measurement, not guessed).
- Trial keep/discard mechanism: git-based (each trial on a temporary branch; keep = merge, discard = reset).
- Metric logged as JSON after each trial for reproducibility.

## 8. Testing stance

- **Library code (H builder, SiTBackflow, later MARCH):** unit tests required. TDD applies.
- **Experiment scripts (`train_oneoff.py`, `train.py`):** no tests. These are research runs; regression surface is the ED benchmark, not pytest.
- **Harness plumbing (`prepare.py`):** small smoke test confirming `score` extracts the right field from a sample log. Not full TDD.

## 9. Dependencies

- Python ≥ 3.11.
- `netket` ≥ 3.22 (for `FermiHubbardJax` + `VMC_SR` with momentum).
- `jax` + `flax` (pulled by NetKet).
- `pytest` for tests.
- macOS / CPU execution assumed for Phase 1; GPU path is orthogonal and can be added later.
- Managed via `uv`: `uv init` then `uv add netket pytest`.

## 10. Out of scope

- Multi-GPU / distributed.
- Larger lattices (Phases 3+).
- t' ≠ 0 and doping (Phases 3+).
- MARCH optimizer implementation (Phase 2, separate spec).
- Any refactor of NetKet upstream (Phase 5, separate spec).
