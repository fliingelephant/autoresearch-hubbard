"""
Autoresearch training script for the 4x4 Hubbard model.

Agent-editable single file. Architecture, optimizer, hyperparameters,
phase splits, and loop structure are all fair game. Frozen pieces live in
`prepare.py` and `src/autoresearch_hubbard/hamiltonian.py`.

Usage:
    uv run train.py > run.log 2>&1

At the end the script prints a summary block:

    ---
    final_energy:    ...
    min_energy:      ...
    elapsed_seconds: ...
    nnb_steps:       ...
    pretrain_steps:  ...
    spring_steps:    ...

Extract the metric with:  grep "^final_energy:" run.log
"""

from __future__ import annotations

import datetime
import math
import time

import jax
import jax.numpy as jnp
import netket as nk
import optax

from prepare import TRIAL_SECONDS, build_system, verify_frozen_surface
from autoresearch_hubbard.ansatz import NNB, SiTBackflow
from autoresearch_hubbard.pretrain import supervised_pretrain_step

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly — no CLI flags)
# ---------------------------------------------------------------------------

SEED = 0

# Numerical precision. Agent-tunable: flip to False + float32 for cheaper runs
# if the ansatz still converges; keep x64 for the paper-faithful baseline.
jax.config.update("jax_enable_x64", True)
DTYPE = jnp.float64

# Phase budget fractions of TRIAL_SECONDS (SPRING gets the remainder).
NNB_FRACTION = 0.2
PRETRAIN_FRACTION = 0.1

# Sampler
N_CHAINS = 16
SWEEP_SIZE = 40
N_SAMPLES = 4096

# NNB warm-start
NNB_HIDDEN_DIM = 256
NNB_LR = 1e-4

# SiT transformer backflow
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
N_DETERMINANTS = 4

# Short model identifier for results.tsv. Update when changing model family
# (e.g. "Slater", "Slater+Jastrow", "MLP-backflow", "RBM").
MODEL_ID = f"SiT(d={D_MODEL},L={N_LAYERS},K={N_DETERMINANTS})"

# Supervised pretraining
PRETRAIN_SAMPLES = 4096
PRETRAIN_LR = 3e-4

# SPRING
SPRING_LR = 1e-2
DIAG_SHIFT = 1e-3
MOMENTUM = 0.95

# Logging cadence
LOG_EVERY_VMC = 10
LOG_EVERY_PRETRAIN = 50


class NaNError(RuntimeError):
    """Raised when training diverges to NaN — let the loop log it as `crash`."""


def _check_finite(value: float, phase: str, step: int, what: str) -> None:
    if not math.isfinite(value):
        raise NaNError(f"{phase} step {step}: {what}={value} (training diverged)")


def run_vmc_phase(driver, seconds_budget: float, phase_name: str, log_every: int) -> list[float]:
    """Run VMC steps until the deadline. The first step (which includes JIT
    compile) runs as a warmup outside the timer, so the budget covers actual
    training only. Raises NaNError if energy goes non-finite."""
    trace: list[float] = []
    latest: dict[str, float] = {}

    def capture(_step, log_data, _driver):
        energy = log_data["Energy"]
        latest["energy"] = float(jnp.real(energy.mean))
        latest["variance"] = float(jnp.real(energy.variance))
        return True

    t_warm = time.perf_counter()
    driver.run(1, out=None, show_progress=False, callback=capture)
    _check_finite(latest["energy"], phase_name, 1, "energy")
    trace.append(latest["energy"])
    print(
        f"{phase_name} step 1 (warmup, {time.perf_counter() - t_warm:.1f}s): "
        f"energy={latest['energy']:.6f} variance={latest['variance']:.4f}",
        flush=True,
    )

    deadline = time.perf_counter() + seconds_budget
    step = 1
    while time.perf_counter() < deadline:
        driver.run(1, out=None, show_progress=False, callback=capture)
        step += 1
        _check_finite(latest["energy"], phase_name, step, "energy")
        trace.append(latest["energy"])
        if step % log_every == 0:
            print(
                f"{phase_name} step {step}: energy={latest['energy']:.6f} variance={latest['variance']:.4f}",
                flush=True,
            )
    return trace


def run_pretraining_phase(
    params,
    model,
    configs: jax.Array,
    targets: jax.Array,
    seconds_budget: float,
    learning_rate: float,
    log_every: int,
):
    """Pretrain orbitals until the deadline. First step warmup absorbs JIT
    compile outside the timer. Raises NaNError if loss goes non-finite."""
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    t_warm = time.perf_counter()
    params, opt_state, loss = supervised_pretrain_step(
        params, model, configs, targets, optimizer=optimizer, opt_state=opt_state,
    )
    loss_val = float(loss)
    _check_finite(loss_val, "pretrain", 1, "loss")
    print(
        f"pretrain step 1 (warmup, {time.perf_counter() - t_warm:.1f}s): loss={loss_val:.6f}",
        flush=True,
    )

    deadline = time.perf_counter() + seconds_budget
    step = 1
    while time.perf_counter() < deadline:
        params, opt_state, loss = supervised_pretrain_step(
            params, model, configs, targets,
            optimizer=optimizer, opt_state=opt_state,
        )
        step += 1
        if step % log_every == 0:
            loss_val = float(loss)
            _check_finite(loss_val, "pretrain", step, "loss")
            print(f"pretrain step {step}: loss={loss_val:.6f}", flush=True)
    return params, step


def main() -> None:
    verify_frozen_surface()
    t_start = time.perf_counter()
    nnb_seconds = TRIAL_SECONDS * NNB_FRACTION
    pretrain_seconds = TRIAL_SECONDS * PRETRAIN_FRACTION
    spring_seconds = TRIAL_SECONDS - nnb_seconds - pretrain_seconds
    print(
        f"Trial budget: {TRIAL_SECONDS}s  "
        f"phase split: nnb={nnb_seconds:.1f} pretrain={pretrain_seconds:.1f} spring={spring_seconds:.1f}",
        flush=True,
    )

    hamiltonian, hilbert, graph = build_system()
    sampler = nk.sampler.MetropolisFermionHop(
        hilbert, graph=graph, n_chains=N_CHAINS, sweep_size=SWEEP_SIZE, spin_symmetric=True,
    )

    # Phase 1: NNB warm-start (paper supplementary §2).
    print("=== Phase 1: NNB warm-start ===", flush=True)
    nnb = NNB(hilbert, hidden_dim=NNB_HIDDEN_DIM, param_dtype=DTYPE)
    nnb_state = nk.vqs.MCState(
        sampler, nnb, n_samples=N_SAMPLES, seed=SEED, sampler_seed=SEED
    )

    def nnb_schedule(step):
        return NNB_LR / (1.0 + step / 1.0e4)

    nnb_driver = nk.driver.VMC(
        hamiltonian,
        optax.adam(nnb_schedule, b1=0.9, b2=0.999),
        variational_state=nnb_state,
    )
    nnb_trace = run_vmc_phase(nnb_driver, nnb_seconds, "nnb", LOG_EVERY_VMC)

    # Phase 2: supervised pretraining of the SiT orbitals against the NNB ansatz.
    print("=== Phase 2: supervised pretraining ===", flush=True)
    samples = nnb_state.sample(n_samples=PRETRAIN_SAMPLES, n_discard_per_chain=0)
    pretrain_configs = samples.reshape((-1, samples.shape[-1])).astype(DTYPE)
    pretrain_targets = nnb.apply(
        nnb_state.variables, pretrain_configs, method=nnb.backflow_orbitals
    )

    sit = SiTBackflow(
        hilbert, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        n_determinants=N_DETERMINANTS, param_dtype=DTYPE,
    )
    sit_variables = sit.init(jax.random.PRNGKey(SEED + 1), pretrain_configs)
    sit_variables, pretrain_steps = run_pretraining_phase(
        sit_variables, sit, pretrain_configs, pretrain_targets,
        pretrain_seconds, PRETRAIN_LR, LOG_EVERY_PRETRAIN,
    )

    # Phase 3: SPRING variational Monte Carlo.
    print("=== Phase 3: SPRING VMC ===", flush=True)
    sit_state = nk.vqs.MCState(
        sampler, sit, n_samples=N_SAMPLES, variables=sit_variables,
        seed=SEED + 2, sampler_seed=SEED + 2,
    )
    spring_driver = nk.driver.VMC_SR(
        hamiltonian, optax.sgd(SPRING_LR), variational_state=sit_state,
        diag_shift=DIAG_SHIFT, momentum=MOMENTUM, mode="complex",
    )
    spring_trace = run_vmc_phase(spring_driver, spring_seconds, "spring", LOG_EVERY_VMC)

    elapsed = time.perf_counter() - t_start
    final_energy = spring_trace[-1] if spring_trace else float("inf")
    all_energies = nnb_trace + spring_trace
    min_energy = min(all_energies) if all_energies else float("inf")

    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    print(
        f"---\n"
        f"final_energy:    {final_energy:.6f}\n"
        f"min_energy:      {min_energy:.6f}\n"
        f"elapsed_seconds: {elapsed:.1f}\n"
        f"nnb_steps:       {len(nnb_trace)}\n"
        f"pretrain_steps:  {pretrain_steps}\n"
        f"spring_steps:    {len(spring_trace)}\n"
        f"tsv_entry:       {timestamp}\t{final_energy:.6f}\t{MODEL_ID}\t"
        f"{len(spring_trace)}\t{elapsed:.1f}\t[STATUS]\t[DESC]",
        flush=True,
    )


if __name__ == "__main__":
    main()
