# Autoresearch-Hubbard Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a correctness-verified pipeline that solves the 4×4 OBC half-filled Hubbard model (U=8, t=1, t'=0) with a paper-faithful transformer-backflow NQS trained by NetKet's built-in SPRING optimizer, then wrap it in a Karpathy-style autoresearch harness.

**Architecture:** Four stepwise deliverables. (1.1) `hamiltonian.py` builds the Hamiltonian with layered correctness tests against free-fermion and ED references. (1.2) `ansatz/sit_backflow.py` implements the per-site-token transformer + linear orbital head + Slater-determinant amplitude, with wiring tests against a manual `logdet` reference. (1.3) `scripts/train_oneoff.py` trains end-to-end and must reach `E_var − E_ED < 1e-3`. (1.4) the training script is refactored into `prepare.py` / `train.py` / `program.md` for the autoresearch loop.

**Tech Stack:** Python ≥ 3.11, `netket ≥ 3.22` (for `FermiHubbardJax` + `VMC_SR(momentum=...)` SPRING), `jax[cpu]`, `flax.linen`, `pytest`. Managed via `uv`.

**A note on commits:** Steps below include `git commit` as the last step of each TDD cycle. Per user preference, feel free to skip them if you prefer to batch; they're included for per-task traceability and to keep working state green.

**Note on the spec vs this plan:** The paper-grounded architecture in Task 6 was refined from the supplementary materials (§1.1, equations S1–S8) after the design spec was written. The plan supersedes the spec's §3 ansatz description on these points: (a) no LayerNorm inside transformer blocks; (b) FFN is a single `SiLU(Dense d→d)`, not a two-layer expansion; (c) generalized (not unrestricted) Slater with real params; (d) amplitude is a sum of `K` determinants per equation S8. Read supplementary pages S2–S3 of the paper PDF (`notes/Gu et al. - 2025 - Solving the Hubbard model with Neural Quantum States.pdf`, pages 26–27 of the PDF) before implementing Task 6.

---

## File map

| Path | Purpose | Testing |
|---|---|---|
| `pyproject.toml` | uv-managed project + deps | — |
| `.gitignore` | ignore `runs/`, `.venv/`, caches | — |
| `src/autoresearch_hubbard/__init__.py` | package marker | — |
| `src/autoresearch_hubbard/hamiltonian.py` | `build_hamiltonian`, `compute_ed_reference`, `free_fermion_reference` | `tests/test_hamiltonian.py` |
| `src/autoresearch_hubbard/ansatz/__init__.py` | subpackage marker | — |
| `src/autoresearch_hubbard/ansatz/sit_backflow.py` | `SiTBackflow` Flax module | `tests/test_sit_backflow.py` |
| `scripts/train_oneoff.py` | step 1.3 milestone run | none (experiment) |
| `src/autoresearch_hubbard/prepare.py` | frozen harness pieces | smoke test in `tests/test_harness.py` |
| `src/autoresearch_hubbard/train.py` | agent-editable training | none |
| `src/autoresearch_hubbard/program.md` | agent instructions | none |
| `tests/test_hamiltonian.py` | Hamiltonian correctness | — |
| `tests/test_sit_backflow.py` | module correctness | — |
| `tests/test_harness.py` | harness smoke | — |

---

## Task 1: Scaffold the project

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/autoresearch_hubbard/__init__.py`
- Create: `src/autoresearch_hubbard/ansatz/__init__.py`
- Create: `tests/__init__.py`
- Create: `README.md` (one line — just a title, so `uv sync` doesn't complain about missing README)

- [ ] **Step 1: Initialize git repository**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && git init && git branch -m main
```

Expected: `Initialized empty Git repository ...`

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "autoresearch-hubbard"
version = "0.1.0"
description = "Paper-faithful NQS for the 2D Hubbard model with an autoresearch-style harness"
requires-python = ">=3.11"
readme = "README.md"
dependencies = [
    "netket>=3.22",
    "jax[cpu]",
    "flax",
    "numpy",
    "scipy",
]

[dependency-groups]
dev = [
    "pytest>=8",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/autoresearch_hubbard"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
filterwarnings = [
    "ignore::DeprecationWarning",
]
```

- [ ] **Step 3: Create `.gitignore`**

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
runs/
*.log
.DS_Store
dist/
build/
*.egg-info/
.ipynb_checkpoints/
```

- [ ] **Step 4: Create package markers and README**

`src/autoresearch_hubbard/__init__.py`:
```python
"""Paper-faithful NQS for the 2D Hubbard model with an autoresearch-style harness."""
```

`src/autoresearch_hubbard/ansatz/__init__.py`:
```python
```

`tests/__init__.py`:
```python
```

`README.md`:
```markdown
# Autoresearch-Hubbard

Paper-faithful transformer-backflow NQS for the 2D Hubbard model, wrapped in a Karpathy-style autoresearch harness. See `docs/superpowers/specs/` and `docs/superpowers/plans/` for design and implementation.
```

- [ ] **Step 5: Install and verify environment**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && uv sync && uv run python -c "import netket, jax, flax; print('netket', netket.__version__); print('jax', jax.__version__); print('flax', flax.__version__)"
```

Expected: non-error output showing netket ≥ 3.22. If netket <3.22, bump the constraint and rerun `uv sync`.

- [ ] **Step 6: Commit scaffold**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && git add pyproject.toml uv.lock .gitignore README.md src/ tests/ docs/ notes/ && git commit -m "feat: scaffold autoresearch-hubbard project with uv"
```

---

## Task 2: Hamiltonian builder — `build_hamiltonian`

**Files:**
- Create: `src/autoresearch_hubbard/hamiltonian.py`
- Test: `tests/test_hamiltonian.py`

- [ ] **Step 1: Write the failing test**

`tests/test_hamiltonian.py`:
```python
import netket as nk
import pytest
from autoresearch_hubbard.hamiltonian import build_hamiltonian


def test_build_hamiltonian_returns_operator_and_hilbert_and_graph():
    H, hi, g = build_hamiltonian(L=2, U=1.0, t=1.0, pbc=False)

    assert isinstance(hi, nk.hilbert.SpinOrbitalFermions)
    assert hi.n_orbitals == 4
    assert tuple(hi.n_fermions_per_spin) == (2, 2)
    assert g.n_nodes == 4
    assert H.hilbert is hi
    assert H.is_hermitian
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && uv run pytest tests/test_hamiltonian.py::test_build_hamiltonian_returns_operator_and_hilbert_and_graph -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'autoresearch_hubbard.hamiltonian'`.

- [ ] **Step 3: Implement `build_hamiltonian`**

`src/autoresearch_hubbard/hamiltonian.py`:
```python
"""Hubbard Hamiltonian builder for the Phase 1 instance (square, OBC, half-filled)."""

import netket as nk
import netket.experimental as nkx
from netket.graph import AbstractGraph
from netket.hilbert import SpinOrbitalFermions


def build_hamiltonian(L: int = 4, U: float = 8.0, t: float = 1.0, pbc: bool = False):
    """Build the 2D square Hubbard Hamiltonian at half filling.

    Returns (H, hilbert, graph). The Hilbert space is the fixed-particle-number
    sector with N_up = N_down = L*L/2 (L must be even).
    """
    n_sites = L * L
    n_per_spin = n_sites // 2
    g: AbstractGraph = nk.graph.Hypercube(length=L, n_dim=2, pbc=pbc)
    hi = SpinOrbitalFermions(
        n_orbitals=n_sites, s=1 / 2, n_fermions_per_spin=(n_per_spin, n_per_spin)
    )
    H = nkx.operator.FermiHubbardJax(hi, graph=g, t=t, U=U)
    return H, hi, g
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && uv run pytest tests/test_hamiltonian.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && git add src/autoresearch_hubbard/hamiltonian.py tests/test_hamiltonian.py && git commit -m "feat(hamiltonian): add build_hamiltonian via FermiHubbardJax"
```

---

## Task 3: Hamiltonian correctness — U = 0 matches free fermions

**Files:**
- Modify: `src/autoresearch_hubbard/hamiltonian.py` (add `free_fermion_ground_state_energy` helper)
- Modify: `tests/test_hamiltonian.py` (add U=0 test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hamiltonian.py`:
```python
import numpy as np
from autoresearch_hubbard.hamiltonian import free_fermion_ground_state_energy


def test_u0_ed_matches_free_fermion_filling():
    """For U=0 the Hubbard model is free fermions; ED must match direct hopping-matrix diagonalization."""
    H, hi, g = build_hamiltonian(L=2, U=0.0, t=1.0, pbc=False)
    e_ff = free_fermion_ground_state_energy(g, hi, t=1.0)
    e_ed = nk.exact.lanczos_ed(H, k=1)[0]
    np.testing.assert_allclose(e_ed, e_ff, atol=1e-10)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && uv run pytest tests/test_hamiltonian.py::test_u0_ed_matches_free_fermion_filling -v
```

Expected: FAIL with `ImportError: cannot import name 'free_fermion_ground_state_energy'`.

- [ ] **Step 3: Implement `free_fermion_ground_state_energy`**

Append to `src/autoresearch_hubbard/hamiltonian.py`:
```python
import numpy as np


def free_fermion_ground_state_energy(graph, hilbert, t: float = 1.0) -> float:
    """Ground-state energy of non-interacting fermions (U=0) on the given graph.

    Computed as twice (two spins) the sum of the n_per_spin lowest eigenvalues of
    the single-particle hopping matrix -t·A, where A is the adjacency matrix.
    """
    n_sites = graph.n_nodes
    A = np.zeros((n_sites, n_sites))
    for i, j in graph.edges():
        A[i, j] = 1.0
        A[j, i] = 1.0
    H1 = -t * A
    eps = np.linalg.eigvalsh(H1)
    n_up, n_dn = hilbert.n_fermions_per_spin
    return float(eps[:n_up].sum() + eps[:n_dn].sum())
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && uv run pytest tests/test_hamiltonian.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && git add src/autoresearch_hubbard/hamiltonian.py tests/test_hamiltonian.py && git commit -m "test(hamiltonian): verify U=0 ED matches free-fermion filling"
```

---

## Task 4: Hamiltonian correctness — small-U perturbative slope

**Files:**
- Modify: `tests/test_hamiltonian.py`

Rationale: At small U, first-order perturbation theory gives `E(U) ≈ E(0) + U · ⟨Σ_i n_{i↑} n_{i↓}⟩_{U=0}`. At half filling with free fermions, `⟨n_{i↑} n_{i↓}⟩ = ⟨n_{i↑}⟩⟨n_{i↓}⟩ = (1/2)(1/2) = 1/4` by Wick's theorem on uncorrelated spin sectors. So `Σ_i ⟨n_{i↑} n_{i↓}⟩_{U=0} = N_sites / 4`. The test checks the linear slope with a finite-difference against this analytical value.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hamiltonian.py`:
```python
def test_small_u_perturbative_slope():
    """E(U) - E(0) ~ U * N_sites/4 at half filling for small U (Wick's theorem on free fermions)."""
    L = 2
    n_sites = L * L
    expected_slope = n_sites / 4.0  # <sum_i n_up n_dn>_{U=0} at half filling

    H0, _, _ = build_hamiltonian(L=L, U=0.0, t=1.0, pbc=False)
    e0 = nk.exact.lanczos_ed(H0, k=1)[0]

    eps = 1e-3
    He, _, _ = build_hamiltonian(L=L, U=eps, t=1.0, pbc=False)
    e_eps = nk.exact.lanczos_ed(He, k=1)[0]

    slope = (e_eps - e0) / eps
    np.testing.assert_allclose(slope, expected_slope, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it passes**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && uv run pytest tests/test_hamiltonian.py::test_small_u_perturbative_slope -v
```

Expected: PASS (no new implementation needed — builder already accepts U).

- [ ] **Step 3: Commit**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && git add tests/test_hamiltonian.py && git commit -m "test(hamiltonian): verify small-U perturbative slope"
```

---

## Task 5: Cached ED reference for the 4×4 instance

**Files:**
- Modify: `src/autoresearch_hubbard/hamiltonian.py` (add `compute_ed_reference`)
- Modify: `tests/test_hamiltonian.py` (add cache test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_hamiltonian.py`:
```python
import json
import os
from autoresearch_hubbard.hamiltonian import compute_ed_reference


def test_compute_ed_reference_caches_to_disk(tmp_path):
    cache = tmp_path / "ed.json"
    H, _, _ = build_hamiltonian(L=2, U=8.0, t=1.0, pbc=False)

    e1 = compute_ed_reference(H, cache_path=str(cache))
    assert cache.exists()
    payload = json.loads(cache.read_text())
    assert abs(payload["energy"] - e1) < 1e-12

    mtime_before = os.path.getmtime(cache)
    e2 = compute_ed_reference(H, cache_path=str(cache))
    assert abs(e1 - e2) < 1e-12
    # Second call should read cache, not recompute.
    assert os.path.getmtime(cache) == mtime_before
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && uv run pytest tests/test_hamiltonian.py::test_compute_ed_reference_caches_to_disk -v
```

Expected: FAIL with `ImportError: cannot import name 'compute_ed_reference'`.

- [ ] **Step 3: Implement `compute_ed_reference`**

Append to `src/autoresearch_hubbard/hamiltonian.py`:
```python
import json
import os


def compute_ed_reference(H, cache_path: str) -> float:
    """Return the ground-state energy of H, caching the result to `cache_path`.

    Uses `nk.exact.lanczos_ed`. Subsequent calls read from cache.
    """
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return float(json.load(f)["energy"])

    energy = float(nk.exact.lanczos_ed(H, k=1)[0])
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"energy": energy}, f)
    return energy
```

- [ ] **Step 4: Run all Hamiltonian tests**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && uv run pytest tests/test_hamiltonian.py -v
```

Expected: all four tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && git add src/autoresearch_hubbard/hamiltonian.py tests/test_hamiltonian.py && git commit -m "feat(hamiltonian): add cached ED reference computation"
```

---

## Task 6: SiTBackflow module — paper-faithful architecture

**Files:**
- Create: `src/autoresearch_hubbard/ansatz/sit_backflow.py`
- Create: `tests/test_sit_backflow.py`

**Paper reference:** Gu et al. 2025, supplementary §1.1, equations S1–S8. Before coding, re-read the equations:
- Token alphabet: 4 physical states per site `{|0⟩, |↑⟩, |↓⟩, |↑↓⟩}` with embedding `E ∈ ℝ^{4×d}` (S1–S3 prose).
- Positional encoding `P ∈ ℝ^{N×d}` learnable (S3 prose, above S4).
- Transformer block (S4, S5): `Y = X + Attn(X); X' = Y + FFN(Y)` — **no LayerNorm**.
- Attention (S6): standard multi-head `softmax(Q K^T / √d_H) V` with per-head `W_Q, W_K, W_V, W_O ∈ ℝ^{d×d_H}`.
- FFN (S7): single linear with SiLU — `FFN(X) = σ(X W_F)` with `W_F ∈ ℝ^{d×d}`, `σ = SiLU` (ref 58).
- Orbital head & amplitude (S8): linear head → `K` orbital matrices `M^k ∈ ℝ^{2N×N_e}`, then `ψ(n) = Σ_{k=1}^K det[Φ^k]` where `Φ^k = M^k[R, :]` and `R = {occupied positions in n}`.
- Real-valued params throughout.

**Shape conventions (NetKet):** for `hilbert = SpinOrbitalFermions(n_orbitals=N, n_fermions_per_spin=(N_d, N_u))`, a config `n ∈ {0,1}^{2N}` has `n[:N]` = spin sector 0 (we call down), `n[N:]` = sector 1 (up). The orbital matrix rows match the same order: `M^k[:N, :]` = down spin-orbitals, `M^k[N:, :]` = up.

- [ ] **Step 1: Write failing shape + dtype + amplitude-reality tests**

`tests/test_sit_backflow.py`:
```python
import jax
import jax.numpy as jnp
import netket as nk
import pytest
from autoresearch_hubbard.ansatz.sit_backflow import SiTBackflow


@pytest.fixture
def small_hilbert():
    return nk.hilbert.SpinOrbitalFermions(
        n_orbitals=4, s=1 / 2, n_fermions_per_spin=(2, 2)
    )


@pytest.fixture
def random_configs(small_hilbert):
    return jnp.asarray(small_hilbert.all_states()[:5], dtype=jnp.int32)


def test_shape_is_batch_scalar(small_hilbert, random_configs):
    model = SiTBackflow(small_hilbert, d_model=8, n_heads=2, n_layers=1)
    params = model.init(jax.random.PRNGKey(0), random_configs)
    out = model.apply(params, random_configs)
    assert out.shape == (random_configs.shape[0],)


def test_log_amp_is_complex_with_real_params(small_hilbert, random_configs):
    """Log-amplitude is complex via logdet_cmplx (sign encoded in imag part),
    but params are real (paper S8 uses M^k ∈ ℝ)."""
    model = SiTBackflow(
        small_hilbert, d_model=8, n_heads=2, n_layers=1, param_dtype=jnp.float32
    )
    params = model.init(jax.random.PRNGKey(0), random_configs)
    flat_params = jax.tree_util.tree_leaves(params)
    for p in flat_params:
        assert jnp.issubdtype(p.dtype, jnp.floating), f"param dtype {p.dtype}"
    out = model.apply(params, random_configs)
    assert jnp.iscomplexobj(out)


def test_k_determinants_hparam(small_hilbert, random_configs):
    """K>1 should produce orbitals of shape (..., K, 2N, N_e)."""
    model = SiTBackflow(
        small_hilbert, d_model=8, n_heads=2, n_layers=1, n_determinants=3
    )
    params = model.init(jax.random.PRNGKey(0), random_configs)
    M = model.apply(params, random_configs, method=SiTBackflow.backflow_orbitals)
    B = random_configs.shape[0]
    N = small_hilbert.n_orbitals
    Ne = small_hilbert.n_fermions
    assert M.shape == (B, 3, 2 * N, Ne)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && uv run pytest tests/test_sit_backflow.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'autoresearch_hubbard.ansatz.sit_backflow'`.

- [ ] **Step 3: Implement `SiTBackflow`**

`src/autoresearch_hubbard/ansatz/sit_backflow.py`:
```python
"""Paper-faithful transformer backflow ansatz (Gu et al. 2025, supplementary §1.1).

Architecture (equations S1–S8):
    n ∈ {0,1}^{2N}, NetKet convention [down_sites (N), up_sites (N)]
      → token_i = n_up[i] + 2 * n_down[i] ∈ {0, 1, 2, 3}  (|0⟩, |↑⟩, |↓⟩, |↑↓⟩)
      → X^(0) = E[token] + P        # (N, d), E ∈ ℝ^{4×d}, P ∈ ℝ^{N×d}, both learnable
    L× Transformer block (NO LayerNorm, S4 & S5):
      Y = X + Attn(X)               # standard multi-head with W_Q, W_K, W_V, W_O
      X = Y + FFN(Y)                # FFN(X) = SiLU(X W_F), W_F ∈ ℝ^{d×d}  (S7)
    Linear orbital head:
      per-site Dense → (N, 2·K·N_e) → reshape/permute → (K, 2N, N_e)
    Slater amplitude (S8):
      Φ^k = M^k[R, :] where R = nonzero(n)   # (K, N_e, N_e) per sample
      ψ(n) = Σ_{k=1}^K det[Φ^k]
      log ψ = logsumexp_c over K of nkjax.logdet_cmplx(Φ^k)
"""

import flax.linen as nn
import jax
import jax.numpy as jnp

from netket import jax as nkjax
from netket.hilbert import SpinOrbitalFermions
from netket.utils.types import DType


class TransformerBlock(nn.Module):
    d_model: int
    n_heads: int
    param_dtype: DType

    @nn.compact
    def __call__(self, x):
        # Residual MHA — no LayerNorm (paper eq. S4)
        y = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.d_model,
            param_dtype=self.param_dtype,
        )(x)
        x = x + y
        # Residual single-layer FFN with SiLU (paper eq. S5 + S7)
        y = nn.Dense(self.d_model, param_dtype=self.param_dtype)(x)
        y = nn.silu(y)
        return x + y


class SiTBackflow(nn.Module):
    hilbert: SpinOrbitalFermions
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    n_determinants: int = 1
    param_dtype: DType = jnp.float32

    @nn.compact
    def __call__(self, n):
        M = self.backflow_orbitals(n)
        return self._logdet_sum(n, M)

    def backflow_orbitals(self, n):
        """Return M of shape (..., K, 2N, N_e)."""
        N = self.hilbert.n_orbitals
        Ne = self.hilbert.n_fermions
        K = self.n_determinants

        if not jnp.issubdtype(n.dtype, jnp.integer):
            n = jnp.asarray(jnp.isclose(n, 1), dtype=jnp.int32)

        n_down = n[..., :N]
        n_up = n[..., N:]
        token = n_up + 2 * n_down  # ∈ {0, 1, 2, 3}

        E = self.param(
            "embed_E",
            nn.initializers.normal(0.02),
            (4, self.d_model),
            self.param_dtype,
        )
        P = self.param(
            "embed_P",
            nn.initializers.normal(0.02),
            (N, self.d_model),
            self.param_dtype,
        )
        x = E[token] + P  # (..., N, d)

        for layer in range(self.n_layers):
            x = TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                param_dtype=self.param_dtype,
                name=f"block_{layer}",
            )(x)

        head = nn.Dense(
            2 * K * Ne, param_dtype=self.param_dtype, name="orbital_head"
        )(x)  # (..., N, 2*K*Ne)
        head = head.reshape(*head.shape[:-1], 2, K, Ne)  # (..., N, 2, K, Ne)
        head = jnp.swapaxes(head, -4, -2)               # (..., K, 2, N, Ne)
        head = head.reshape(*head.shape[:-3], 2 * N, Ne)  # (..., K, 2N, Ne)
        return head

    def _logdet_sum(self, n, M):
        """log ψ(n) = log(Σ_k det(M^k[R, :])) where R = nonzero(n)."""
        Ne = self.hilbert.n_fermions
        if not jnp.issubdtype(n.dtype, jnp.integer):
            n = jnp.asarray(jnp.isclose(n, 1), dtype=jnp.int32)

        def per_sample(n_b, M_b):
            R = jnp.nonzero(n_b, size=Ne)[0]       # (Ne,)
            Phi = M_b[:, R, :]                     # (K, Ne, Ne)
            log_dets = jax.vmap(nkjax.logdet_cmplx)(Phi)  # (K,) complex
            # logsumexp over K with complex values: shift by real max for stability
            shift = jnp.max(log_dets.real)
            return shift + jnp.log(jnp.sum(jnp.exp(log_dets - shift)))

        return jax.vmap(per_sample)(n, M)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && uv run pytest tests/test_sit_backflow.py -v
```

Expected: all three tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && git add src/autoresearch_hubbard/ansatz/sit_backflow.py tests/test_sit_backflow.py && git commit -m "feat(ansatz): add paper-faithful SiTBackflow (no LN, SiLU-FFN, K-det sum)"
```

---

## Task 7: SiTBackflow — amplitude matches manual det computation

**Files:**
- Modify: `tests/test_sit_backflow.py`

Rationale: Test that the module's log-amplitude agrees with a manual determinant computation using the orbital matrices extracted from `backflow_orbitals`. Catches row-selection and logsumexp bugs. We test both `K=1` (single det) and `K=2` (sum of dets) for coverage.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sit_backflow.py`:
```python
import numpy as np


@pytest.mark.parametrize("K", [1, 2])
def test_amplitude_matches_manual_det_sum(small_hilbert, random_configs, K):
    model = SiTBackflow(
        small_hilbert, d_model=8, n_heads=2, n_layers=1, n_determinants=K
    )
    params = model.init(jax.random.PRNGKey(0), random_configs)

    log_amp = np.asarray(model.apply(params, random_configs))
    M = np.asarray(
        model.apply(params, random_configs, method=SiTBackflow.backflow_orbitals)
    )  # (B, K, 2N, Ne)

    Ne = small_hilbert.n_fermions
    configs_np = np.asarray(random_configs)
    manual = np.zeros(configs_np.shape[0], dtype=np.complex128)
    for b in range(configs_np.shape[0]):
        R = np.nonzero(configs_np[b])[0]
        assert R.shape == (Ne,)
        psi_b = 0.0 + 0.0j
        for k in range(K):
            Phi = M[b, k, R, :]           # (Ne, Ne)
            psi_b = psi_b + np.linalg.det(Phi)
        manual[b] = np.log(psi_b + 0j)

    np.testing.assert_allclose(log_amp, manual, atol=1e-4, rtol=1e-4)
```

- [ ] **Step 2: Run test to verify it passes**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && uv run pytest tests/test_sit_backflow.py -k manual_det_sum -v
```

Expected: both parametrized variants PASS. If either fails, the bug is in orbital-head reshape/permute (producing wrong `M[b,k,i,:]` layout) or in `_logdet_sum`'s row selection / logsumexp.

- [ ] **Step 3: Commit**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && git add tests/test_sit_backflow.py && git commit -m "test(ansatz): verify K-determinant sum matches manual computation"
```

---

## Task 8: SiTBackflow — gradient flow

**Files:**
- Modify: `tests/test_sit_backflow.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_sit_backflow.py`:
```python
def test_gradient_is_finite(small_hilbert, random_configs):
    model = SiTBackflow(small_hilbert, d_model=8, n_heads=2, n_layers=1)
    params = model.init(jax.random.PRNGKey(0), random_configs)

    def loss(p):
        log_amp = model.apply(p, random_configs)
        return jnp.mean(jnp.abs(log_amp) ** 2).real

    grads = jax.grad(loss)(params)
    leaves = jax.tree_util.tree_leaves(grads)
    for g in leaves:
        assert jnp.all(jnp.isfinite(g)), f"non-finite grad: {g}"
```

- [ ] **Step 2: Run test to verify it passes**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && uv run pytest tests/test_sit_backflow.py::test_gradient_is_finite -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && git add tests/test_sit_backflow.py && git commit -m "test(ansatz): verify gradients flow without NaN"
```

---

## Task 9: Full test suite green

- [ ] **Step 1: Run all tests**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && uv run pytest -v
```

Expected: all tests PASS (4 Hamiltonian + 4 ansatz = 8 tests green).

No commit — this is a gate.

---

## Task 10: One-off end-to-end training

**Files:**
- Create: `scripts/train_oneoff.py`
- Create: `runs/` (directory, gitignored)

Rationale: Non-TDD experiment script per the user's `No trivial TDD` rule. Success criterion is `E_var − E_ED_ref < 1e-3` at 4×4 half-filled U=8.

- [ ] **Step 1: Write the training script**

`scripts/train_oneoff.py`:
```python
"""Phase 1 step 1.3 — one-off end-to-end training on 4×4 OBC half-filled U=8.

Success gate: (min variational energy over the run) - E_ED_ref < 5e-3.
(Paper's accuracy benchmark at 4x4, Table S4, reports 99.9% = 1e-3 relative error —
we aim for a looser absolute gap as a first milestone on this machine.)
"""

from pathlib import Path

import netket as nk

from autoresearch_hubbard.ansatz.sit_backflow import SiTBackflow
from autoresearch_hubbard.hamiltonian import build_hamiltonian, compute_ed_reference


def main():
    # Phase 1 instance
    L = 4
    H, hi, g = build_hamiltonian(L=L, U=8.0, t=1.0, pbc=False)

    cache = Path("runs/ed_reference_4x4_obc_half_U8.json")
    cache.parent.mkdir(exist_ok=True)
    e_ed = compute_ed_reference(H, cache_path=str(cache))
    print(f"E_ED = {e_ed:.6f}")

    sampler = nk.sampler.MetropolisFermionHop(
        hi, graph=g, n_chains=16, sweep_size=64, spin_symmetric=True
    )
    model = SiTBackflow(hi, d_model=32, n_heads=4, n_layers=2, n_determinants=1)
    vs = nk.vqs.MCState(sampler, model, n_samples=1024, n_discard_per_chain=16)

    opt = nk.optimizer.Sgd(learning_rate=0.03)
    driver = nk.driver.VMC_SR(
        H,
        opt,
        variational_state=vs,
        diag_shift=1e-3,
        momentum=0.8,
        mode="real",  # real params; logψ is complex via logdet_cmplx (sign encoded)
    )

    log_path = Path("runs/oneoff")
    log_path.parent.mkdir(exist_ok=True)
    driver.run(1000, out=str(log_path))

    import json

    with open(f"{log_path}.log") as f:
        data = json.load(f)
    energies = data["Energy"]["Mean"]["real"]
    e_best = min(energies)
    gap = e_best - e_ed
    print(f"E_best = {e_best:.6f}   E_best - E_ED = {gap:.3e}")
    if gap < 5e-3:
        print("Phase 1 step 1.3 MILESTONE: met (gap < 5e-3).")
    else:
        print("Phase 1 step 1.3 MILESTONE: NOT met yet; tune (or add pretraining, Task 10.5).")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the script**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && uv run python scripts/train_oneoff.py
```

Expected: the run prints the ED reference (first call caches it; subsequent runs reuse), runs 1000 VMC steps, and reports the gap. JIT compilation dominates the first ~60 s; then each step is a few seconds on CPU.

If the gap is `< 5e-3`: milestone met, proceed to Task 11.

If not, tune (in priority order):
1. Increase `n_samples` (e.g. 2048 or 4096).
2. Increase `n_layers` / `d_model` (e.g. `d_model=64, n_layers=4, n_heads=4`).
3. Try `n_determinants > 1` (paper's K sum-of-determinants improves expressivity).
4. Adjust `diag_shift` (1e-4 to 1e-2) and `learning_rate` (1e-2 to 1e-1).
5. Run more iterations (e.g. 3000).
6. Turn off SPRING (`momentum=None`) to isolate whether momentum is destabilizing.
7. **If still stuck, switch to Task 10.5** (pretraining). The paper notes (supp §2) that "compared to random initialization, a pretrained transformer is much more stable and converges significantly faster". At 4×4 with K=1, random init may still be workable; beyond that, pretraining becomes essential.
8. Debug anchor: swap `SiTBackflow` for `nk.models.Slater2nd(hi)` — pure HF. HF on 4×4 OBC U=8 should reach an energy near the Hartree-Fock value (paper Table S2 reports HF per-site energies for 16×4, 8×8, etc.; we don't have 4×4 directly but HF converges in seconds). If HF-Slater fails to converge, the bug is in H or sampler.

- [ ] **Step 3: Commit script**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && git add scripts/train_oneoff.py && git commit -m "feat(scripts): add one-off 4x4 Hubbard training milestone"
```

- [ ] **Step 4: Only after milestone is met, commit the ED cache**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && git add -f runs/ed_reference_4x4_obc_half_U8.json && git commit -m "chore(runs): cache 4x4 OBC half-filled U=8 ED reference"
```

(Force-add because `runs/` is gitignored; the reference is a physical invariant worth committing for reproducibility.)

---

## Task 11: Autoresearch harness refactor

**Files:**
- Create: `src/autoresearch_hubbard/prepare.py`
- Create: `src/autoresearch_hubbard/train.py`
- Create: `src/autoresearch_hubbard/program.md`
- Create: `tests/test_harness.py`

Rationale: Refactor `scripts/train_oneoff.py` into three files matching the autoresearch pattern. `prepare.py` is frozen across trials; `train.py` is the agent's edit target; `program.md` tells the agent what it may and may not change.

- [ ] **Step 1: Write the failing harness smoke test**

`tests/test_harness.py`:
```python
import json
from autoresearch_hubbard.prepare import score_from_log


def test_score_from_log_returns_gap(tmp_path):
    log = tmp_path / "trial.log"
    log.write_text(json.dumps({
        "Energy": {
            "iters": [0, 1, 2],
            "Mean": {"real": [-5.0, -6.0, -7.0]},
        }
    }))
    gap = score_from_log(str(log), e_ed_ref=-8.0)
    assert gap == -7.0 - (-8.0)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && uv run pytest tests/test_harness.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'autoresearch_hubbard.prepare'`.

- [ ] **Step 3: Implement `prepare.py`**

`src/autoresearch_hubbard/prepare.py`:
```python
"""Frozen harness pieces for the autoresearch-style trial loop.

This file is NOT agent-editable. Agents may only edit `train.py`.
"""

import json
from pathlib import Path

import netket as nk

from autoresearch_hubbard.hamiltonian import build_hamiltonian, compute_ed_reference

TRIAL_SECONDS = 180  # wall-clock budget per trial; calibrate after first run
ED_CACHE = "runs/ed_reference_4x4_obc_half_U8.json"


def build_problem():
    """Build the Phase 1 problem: H, hilbert, graph, sampler, E_ED."""
    H, hi, g = build_hamiltonian(L=4, U=8.0, t=1.0, pbc=False)
    sampler = nk.sampler.MetropolisFermionHop(
        hi, graph=g, n_chains=16, sweep_size=64, spin_symmetric=True
    )
    Path(ED_CACHE).parent.mkdir(exist_ok=True)
    e_ed = compute_ed_reference(H, cache_path=ED_CACHE)
    return H, hi, g, sampler, e_ed


def score_from_log(log_path: str, e_ed_ref: float) -> float:
    """Return best_energy - E_ED from a NetKet VMC log. Lower is better."""
    with open(log_path) as f:
        data = json.load(f)
    energies = data["Energy"]["Mean"]["real"]
    return float(min(energies) - e_ed_ref)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && uv run pytest tests/test_harness.py -v
```

Expected: PASS.

- [ ] **Step 5: Implement `train.py` (agent-editable)**

`src/autoresearch_hubbard/train.py`:
```python
"""Agent-editable training loop.

The agent may modify ANYTHING in this file — model hyperparameters, optimizer
config, sampler kwargs, training schedule. The agent MAY NOT modify:
  - `prepare.py`
  - the Hamiltonian, Hilbert space, or particle sector
  - the `score_from_log` metric or the log-path convention

Wall-clock budget (`TRIAL_SECONDS` in `prepare.py`) is enforced externally by the
harness (subprocess timeout); `n_iters` here is calibrated so that one `driver.run`
call fits inside the budget on the current machine.
"""

from pathlib import Path

import netket as nk

from autoresearch_hubbard.ansatz.sit_backflow import SiTBackflow
from autoresearch_hubbard.prepare import build_problem, score_from_log


def run_trial(log_dir: str = "runs/trial", n_iters: int = 500) -> float:
    H, hi, g, sampler, e_ed = build_problem()

    model = SiTBackflow(hi, d_model=32, n_heads=4, n_layers=2, n_determinants=1)
    vs = nk.vqs.MCState(sampler, model, n_samples=1024, n_discard_per_chain=16)

    opt = nk.optimizer.Sgd(learning_rate=0.03)
    driver = nk.driver.VMC_SR(
        H, opt, variational_state=vs,
        diag_shift=1e-3, momentum=0.8, mode="real",
    )

    Path(log_dir).parent.mkdir(exist_ok=True)
    driver.run(n_iters, out=log_dir)  # NetKet appends .log

    gap = score_from_log(f"{log_dir}.log", e_ed)
    print(f"trial score (lower=better): {gap:.3e}")
    return gap


if __name__ == "__main__":
    run_trial()
```

- [ ] **Step 6: Write `program.md`**

`src/autoresearch_hubbard/program.md`:
```markdown
# Autoresearch task: minimize variational energy on 4×4 Hubbard

## Task
Minimize `score_from_log(log_path, e_ed_ref) = min(E_trace) - E_ED` at the end of one trial.
Lower is better. A trial runs for a fixed wall-clock budget (`TRIAL_SECONDS`).

## Constraints (must not change)
- `src/autoresearch_hubbard/prepare.py` is frozen.
- `src/autoresearch_hubbard/hamiltonian.py` is frozen.
- `TRIAL_SECONDS`, `score_from_log`, and the log-path convention in `prepare.py`.
- The Hamiltonian (`L=4`, `U=8`, `t=1`, `pbc=False`) and particle sector
  (`n_fermions_per_spin=(8, 8)`).

## What you may change
Everything in `src/autoresearch_hubbard/train.py`:
- `SiTBackflow` hyperparameters (`d_model`, `n_heads`, `n_layers`, `mlp_ratio`, `param_dtype`).
- Sampler kwargs (`n_chains`, `sweep_size`, `n_samples`, `n_discard_per_chain`).
- Optimizer and driver kwargs (`learning_rate`, `diag_shift`, `momentum`, `proj_reg`, `mode`, `use_ntk`).
- `n_iters` (number of VMC steps per trial — calibrate so that one trial fits inside `TRIAL_SECONDS`).
- The training loop body (schedules, chunking, intermediate ramps, resets).

## How to run a trial
```bash
uv run python -c "from autoresearch_hubbard.train import run_trial; run_trial('runs/trial_<id>')"
```
Wall-clock budget is enforced externally (e.g., `timeout <TRIAL_SECONDS>s uv run ...`). Then read the metric printed on stdout and/or `runs/trial_<id>.log`.

## Iteration loop
1. Edit `train.py`.
2. Run a trial to a fresh log dir.
3. Compare score to the best score so far.
4. If lower → keep (git commit). If not → revert (git checkout).
5. Repeat.
```

- [ ] **Step 7: Full test suite green**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && uv run pytest -v
```

Expected: all tests PASS (4 Hamiltonian + 4 ansatz + 1 harness = 9 tests).

- [ ] **Step 8: Commit the harness**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && git add src/autoresearch_hubbard/prepare.py src/autoresearch_hubbard/train.py src/autoresearch_hubbard/program.md tests/test_harness.py && git commit -m "feat(harness): refactor into prepare/train/program.md autoresearch layout"
```

- [ ] **Step 9: Sanity-run the harness**

```bash
cd /Users/zhou/pycode/Autoresearch-Hubbard && uv run python -c "from autoresearch_hubbard.train import run_trial; run_trial('runs/harness_sanity')"
```

Expected: the trial runs for `TRIAL_SECONDS` and prints a finite `trial score`. Value should be comparable to the Task 10 milestone (since the config is identical).

No commit — this is a sanity check.

---

## Phase 1 exit criteria

- All 9 tests green (4 Hamiltonian + 4 ansatz + 1 harness).
- `scripts/train_oneoff.py` has hit `E_var − E_ED < 5e-3` at least once.
- `runs/ed_reference_4x4_obc_half_U8.json` committed.
- Harness (`prepare.py` / `train.py` / `program.md`) runs end-to-end with the same milestone.
- No agentic trials yet — Phase 1 stops at "harness is runnable with a human driver". The first agent-driven trial belongs to a follow-up task outside this plan.

## Out of scope for this plan

- **Pretraining pipeline** (paper supp §2): NNB + supervised MSE pretrain before MARCH. The paper reports this is critical for large systems. At 4×4 with K=1 random init may still converge; if it does not, defer to a Phase 1.5 spec that implements:
  - `NNBAnsatz` (MLP → block-diagonal `M^{nnb}(n) ∈ ℝ^{2N×N_e}`)
  - VMC training of NNB → saved checkpoint
  - Dataset generation `S_pre = {(n_i, M_nnb(n_i))}`
  - Supervised pretraining of `SiTBackflow` with MSE loss on orbital matrices
  - Resume energy-minimization VMC from the pretrained weights
- MARCH optimizer (Phase 2 spec).
- Doped δ=1/8 and t' = −0.2 (Phase 3 spec).
- Larger lattices (Phase 4 spec).
- Upstream NetKet PR (Phase 5 spec).
- Actually running the agentic trial loop — that's an operator activity, not a code task.
