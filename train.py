"""
Autoresearch training script for the 16x4 Hubbard model.

Agent-editable single file. Architecture, optimizer, hyperparameters,
phase splits, and loop structure are all fair game. Frozen pieces live in
`prepare.py` and `src/autoresearch_hubbard/hamiltonian.py`.

Paper baseline (Gu et al. 2025, Table S2 at 16x4 OBC half-filled):
    HF   = -0.52499
    PEPS = -0.68304(5)    (D >= 20)
    NQS  = -0.68325       (paper's own result)
    DMRG = -0.68537       (beat-target for "NQS wins" — hard at Ly = 4)

Usage:
    uv run train.py > run.log 2>&1

Summary printed at end:
    ---
    final_energy:    ...
    min_energy:      ...
    elapsed_seconds: ...
    nnb_steps:       ...
    pretrain_steps:  ...
    spring_steps:    ...        (MARCH/SPRING VMC iterations)
"""

from __future__ import annotations

import datetime
import math
import time

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import optax
from netket.utils import struct

from prepare import (
    LATTICE_LX,
    LATTICE_LY,
    LATTICE_PBC,
    TRIAL_SECONDS,
    build_system,
    verify_frozen_surface,
)
from autoresearch_hubbard.ansatz import NNB, SiTBackflow
from autoresearch_hubbard.driver import NormSchedule, VMC_SR_clipped
from autoresearch_hubbard.pretrain import supervised_pretrain_step

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly — no CLI flags)
# ---------------------------------------------------------------------------

SEED = 0

jax.config.update("jax_enable_x64", True)
DTYPE = jnp.float64

N_SITES = LATTICE_LX * LATTICE_LY

# Phase budget fractions of TRIAL_SECONDS (VMC/MARCH gets the remainder).
NNB_FRACTION = 0.2
PRETRAIN_FRACTION = 0.1

# Sampler. Paper Table S7: MCMC step = 2.5 * Lx * Ly.
N_CHAINS = 128
SWEEP_SIZE = max(round(2.5 * N_SITES), 1)
N_SAMPLES = 4096

# NNB warm-start (paper Table S5: Adam; here SGD+MinSR+clipping — the only
# clipping-aware driver netket exposes today).
NNB_HIDDEN_DIM = 256
NNB_LR = 1e-4


def nnb_schedule(step: int) -> float:
    return NNB_LR / (1.0 + step / 1.0e4)


# SiT transformer backflow (paper Table S7).
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 4
N_DETERMINANTS = 4
MODEL_ID = f"SiT(d={D_MODEL},L={N_LAYERS},K={N_DETERMINANTS})+MARCH"

# Pretraining (paper Table S6).
PRETRAIN_SAMPLES = 4096
PRETRAIN_LR = 3e-4

# MARCH main phase (paper Table S7).
LEARNING_RATE = 1e-2
DIAG_SHIFT = 1e-3          # λ
MOMENTUM = 0.95             # μ (SPRING)
BETA = 0.995                # β (MARCH second-moment decay)
MOMENT_ADAPTIVE = True      # MARCH on top of SPRING
CLIP_C = 5.0                # FermiNet-style local-energy clip


# Norm-constraint schedule: 10^-1 * (1 + max(t-8000,0)/8000)^-1 (paper Table S7).
def norm_bound(step: int) -> float:
    return 1e-1 / (1.0 + max(step - 8000, 0) / 8000.0)


# Logging cadence.
LOG_EVERY_VMC = 10
LOG_EVERY_PRETRAIN = 50


class NaNError(RuntimeError):
    """Raised when training diverges to NaN — let the loop log it as `crash`."""


def _check_finite(value: float, phase: str, step: int, what: str) -> None:
    if not math.isfinite(value):
        raise NaNError(f"{phase} step {step}: {what}={value} (training diverged)")


class ProgressLogger(nk.callbacks.AbstractCallback):
    """Per-step energy/variance print + NaN tripwire, shared by NNB and MARCH."""

    name: str = struct.field(pytree_node=False, serialize=False)
    log_every: int = struct.field(pytree_node=False, serialize=False)
    _start: float = struct.field(pytree_node=False, serialize=False, default=0.0)

    def __init__(self, name: str, log_every: int):
        super().__init__()
        self.name = name
        self.log_every = log_every

    def on_run_start(self, step, driver):
        self._start = time.perf_counter()

    def on_step_end(self, step, log_data, driver):
        k = step + 1  # netket step is 0-based at on_step_end; show 1-based.
        energy = log_data["Energy"]
        e = float(jnp.real(energy.mean))
        _check_finite(e, self.name, k, "energy")
        if k == 1 or k % self.log_every == 0:
            v = float(jnp.real(energy.variance))
            warm = f" (warmup, {time.perf_counter() - self._start:.1f}s)" if k == 1 else ""
            print(f"{self.name} step {k}{warm}: energy={e:.6f} variance={v:.4f}", flush=True)


def run_pretraining_phase(
    params, sit_model, nnb_state, nnb_model,
    seconds_budget: float, learning_rate: float, log_every: int, n_samples: int,
):
    """Online supervised pretraining of SiT against NNB (paper §S.2, Eq. S28).

    Each step resamples from the NNB Markov chain; targets are the NNB
    backflow orbitals evaluated on those fresh configs.
    """
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    nnb_vars = nnb_state.variables
    t0 = time.perf_counter()
    deadline = t0 + seconds_budget
    step = 0
    while step == 0 or time.perf_counter() < deadline:
        samples = nnb_state.sample(n_samples=n_samples, n_discard_per_chain=0)
        configs = samples.reshape((-1, samples.shape[-1])).astype(DTYPE)
        params, opt_state, loss = supervised_pretrain_step(
            params, sit_model, nnb_vars, nnb_model, configs,
            optimizer=optimizer, opt_state=opt_state,
        )
        step += 1
        if step == 1 or step % log_every == 0:
            loss_val = float(loss)
            _check_finite(loss_val, "pretrain", step, "loss")
            warm = f" (warmup, {time.perf_counter() - t0:.1f}s)" if step == 1 else ""
            print(f"pretrain step {step}{warm}: loss={loss_val:.6f}", flush=True)
    return params, step


def main() -> None:
    verify_frozen_surface()
    t_start = time.perf_counter()
    nnb_seconds = TRIAL_SECONDS * NNB_FRACTION
    pretrain_seconds = TRIAL_SECONDS * PRETRAIN_FRACTION
    march_seconds = TRIAL_SECONDS - nnb_seconds - pretrain_seconds
    print(
        f"System: {LATTICE_LX}x{LATTICE_LY} {'PBC' if LATTICE_PBC else 'OBC'} "
        f"({N_SITES} sites), sweep_size={SWEEP_SIZE}",
        flush=True,
    )
    print(
        f"Trial budget: {TRIAL_SECONDS}s  "
        f"phase split: nnb={nnb_seconds:.1f} pretrain={pretrain_seconds:.1f} march={march_seconds:.1f}",
        flush=True,
    )

    hamiltonian, hilbert, graph = build_system()
    sampler = nk.sampler.MetropolisFermionHop(
        hilbert, graph=graph, n_chains=N_CHAINS, sweep_size=SWEEP_SIZE, spin_symmetric=True,
    )

    # Phase 1: NNB warm-start (paper Table S5).
    print("=== Phase 1: NNB warm-start ===", flush=True)
    nnb = NNB(hilbert, hidden_dim=NNB_HIDDEN_DIM, param_dtype=DTYPE)
    nnb_state = nk.vqs.MCState(
        sampler, nnb, n_samples=N_SAMPLES, seed=SEED, sampler_seed=SEED
    )
    nnb_driver = VMC_SR_clipped(
        hamiltonian, optax.sgd(nnb_schedule),
        variational_state=nnb_state,
        diag_shift=DIAG_SHIFT, clip_c=CLIP_C,
        use_ntk=True, on_the_fly=True, mode="real",
    )
    nnb_logger = nk.logging.RuntimeLog()
    nnb_driver.run(
        n_iter=10**9, out=nnb_logger, show_progress=False,
        callback=[nk.callbacks.Timeout(nnb_seconds), ProgressLogger("nnb", LOG_EVERY_VMC)],
    )
    nnb_trace = np.real(nnb_logger.data["Energy"]["Mean"]).tolist()

    # Phase 2: online supervised pretraining of SiT against NNB (paper §S.2).
    print("=== Phase 2: supervised pretraining ===", flush=True)
    sit = SiTBackflow(
        hilbert, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        n_determinants=N_DETERMINANTS, param_dtype=DTYPE,
    )
    init_samples = nnb_state.sample(n_samples=PRETRAIN_SAMPLES, n_discard_per_chain=0)
    init_configs = init_samples.reshape((-1, init_samples.shape[-1])).astype(DTYPE)
    sit_variables = sit.init(jax.random.PRNGKey(SEED + 1), init_configs)
    sit_variables, pretrain_steps = run_pretraining_phase(
        sit_variables, sit, nnb_state, nnb,
        pretrain_seconds, PRETRAIN_LR, LOG_EVERY_PRETRAIN, PRETRAIN_SAMPLES,
    )

    # Phase 3: MARCH VMC (paper Table S7).
    print("=== Phase 3: MARCH VMC ===", flush=True)
    sit_state = nk.vqs.MCState(
        sampler, sit, n_samples=N_SAMPLES, variables=sit_variables,
        seed=SEED + 2, sampler_seed=SEED + 2,
    )
    march_driver = VMC_SR_clipped(
        hamiltonian, optax.sgd(LEARNING_RATE), variational_state=sit_state,
        diag_shift=DIAG_SHIFT, momentum=MOMENTUM,
        moment_adaptive=MOMENT_ADAPTIVE, beta=BETA,
        clip_c=CLIP_C,
        use_ntk=True, on_the_fly=True, mode="complex",
    )
    march_logger = nk.logging.RuntimeLog()
    march_driver.run(
        n_iter=10**9, out=march_logger, show_progress=False,
        callback=[
            nk.callbacks.Timeout(march_seconds),
            NormSchedule(norm_bound),
            ProgressLogger("march", LOG_EVERY_VMC),
        ],
    )
    march_trace = np.real(march_logger.data["Energy"]["Mean"]).tolist()

    elapsed = time.perf_counter() - t_start
    final_energy = march_trace[-1] if march_trace else float("inf")
    all_energies = nnb_trace + march_trace
    min_energy = min(all_energies) if all_energies else float("inf")
    final_per_site = final_energy / N_SITES
    min_per_site = min_energy / N_SITES

    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    print(
        f"---\n"
        f"final_energy:    {final_energy:.6f}\n"
        f"final_per_site:  {final_per_site:.6f}\n"
        f"min_energy:      {min_energy:.6f}\n"
        f"min_per_site:    {min_per_site:.6f}\n"
        f"elapsed_seconds: {elapsed:.1f}\n"
        f"nnb_steps:       {len(nnb_trace)}\n"
        f"pretrain_steps:  {pretrain_steps}\n"
        f"spring_steps:    {len(march_trace)}\n"
        f"tsv_entry:       {timestamp}\t{final_per_site:.6f}\t{MODEL_ID}\t"
        f"{len(march_trace)}\t{elapsed:.1f}\t[STATUS]\t[DESC]",
        flush=True,
    )


if __name__ == "__main__":
    main()
