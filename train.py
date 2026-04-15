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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def allocate_phase_seconds(
    total_seconds: float,
    *,
    nnb_fraction: float,
    pretrain_fraction: float,
) -> dict[str, float]:
    if total_seconds <= 0:
        raise ValueError("total_seconds must be positive")
    if nnb_fraction < 0 or pretrain_fraction < 0:
        raise ValueError("phase fractions must be non-negative")
    spring_fraction = 1.0 - nnb_fraction - pretrain_fraction
    if spring_fraction < 0:
        raise ValueError("phase fractions exceed total budget")
    return {
        "nnb": total_seconds * nnb_fraction,
        "pretrain": total_seconds * pretrain_fraction,
        "spring": total_seconds * spring_fraction,
    }


def format_summary_lines(summary: dict) -> list[str]:
    return [
        "---",
        f"final_energy:    {summary['final_energy']:.6f}",
        f"min_energy:      {summary['min_energy']:.6f}",
        f"elapsed_seconds: {summary['elapsed_seconds']:.1f}",
        f"nnb_steps:       {summary['nnb_steps']}",
        f"pretrain_steps:  {summary['pretrain_steps']}",
        f"spring_steps:    {summary['spring_steps']}",
    ]


def _flatten_samples(samples):
    return samples.reshape((-1, samples.shape[-1])).astype(jnp.float32)


def run_vmc_phase(driver, seconds_budget: float, phase_name: str, log_every: int) -> list[float]:
    trace: list[float] = []
    deadline = time.perf_counter() + max(seconds_budget, 0.0)
    step = 0
    while time.perf_counter() < deadline:
        latest: dict[str, float] = {}

        def capture(_step, log_data, _driver):
            energy = log_data["Energy"]
            latest["energy"] = float(jnp.real(energy.mean))
            latest["variance"] = float(jnp.real(energy.variance))
            return True

        driver.run(1, out=None, show_progress=False, callback=capture)
        trace.append(latest["energy"])
        step += 1
        if step == 1 or step % log_every == 0:
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
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    deadline = time.perf_counter() + max(seconds_budget, 0.0)
    step = 0
    while time.perf_counter() < deadline:
        params, opt_state, loss = supervised_pretrain_step(
            params,
            model,
            configs,
            targets,
            optimizer=optimizer,
            opt_state=opt_state,
        )
        step += 1
        if step == 1 or step % log_every == 0:
            print(f"pretrain step {step}: loss={float(loss):.6f}", flush=True)
    return params, step


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    verify_frozen_surface()
    t_start = time.perf_counter()
    phase_seconds = allocate_phase_seconds(
        TRIAL_SECONDS,
        nnb_fraction=NNB_FRACTION,
        pretrain_fraction=PRETRAIN_FRACTION,
    )
    print(f"Trial budget: {TRIAL_SECONDS}s  phase split: {phase_seconds}", flush=True)

    hamiltonian, hilbert, graph = build_system()
    sampler = nk.sampler.MetropolisFermionHop(
        hilbert,
        graph=graph,
        n_chains=N_CHAINS,
        sweep_size=SWEEP_SIZE,
        spin_symmetric=True,
    )

    # Phase 1: NNB warm-start (paper supplementary §2).
    print("=== Phase 1: NNB warm-start ===", flush=True)
    nnb = NNB(hilbert, hidden_dim=NNB_HIDDEN_DIM)
    nnb_state = nk.vqs.MCState(
        sampler, nnb, n_samples=N_SAMPLES, seed=SEED, sampler_seed=SEED
    )
    nnb_schedule = lambda step: NNB_LR / (1.0 + step / 1.0e4)
    nnb_driver = nk.driver.VMC(
        hamiltonian,
        optax.adam(nnb_schedule, b1=0.9, b2=0.999),
        variational_state=nnb_state,
    )
    nnb_trace = run_vmc_phase(nnb_driver, phase_seconds["nnb"], "nnb", LOG_EVERY_VMC)

    # Phase 2: supervised pretraining of the SiT orbitals against the NNB ansatz.
    print("=== Phase 2: supervised pretraining ===", flush=True)
    pretrain_configs = _flatten_samples(
        nnb_state.sample(n_samples=PRETRAIN_SAMPLES, n_discard_per_chain=0)
    )
    pretrain_targets = nnb.apply(
        nnb_state.variables, pretrain_configs, method=nnb.backflow_orbitals
    )

    sit = SiTBackflow(
        hilbert,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        n_determinants=N_DETERMINANTS,
    )
    sit_variables = sit.init(jax.random.PRNGKey(SEED + 1), pretrain_configs)
    sit_variables, pretrain_steps = run_pretraining_phase(
        sit_variables,
        sit,
        pretrain_configs,
        pretrain_targets,
        phase_seconds["pretrain"],
        PRETRAIN_LR,
        LOG_EVERY_PRETRAIN,
    )

    # Phase 3: SPRING variational Monte Carlo.
    print("=== Phase 3: SPRING VMC ===", flush=True)
    sit_state = nk.vqs.MCState(
        sampler,
        sit,
        n_samples=N_SAMPLES,
        variables=sit_variables,
        seed=SEED + 2,
        sampler_seed=SEED + 2,
    )
    spring_driver = nk.driver.VMC_SR(
        hamiltonian,
        optax.sgd(SPRING_LR),
        variational_state=sit_state,
        diag_shift=DIAG_SHIFT,
        momentum=MOMENTUM,
        mode="complex",
    )
    spring_trace = run_vmc_phase(
        spring_driver, phase_seconds["spring"], "spring", LOG_EVERY_VMC
    )

    # Summary.
    elapsed = time.perf_counter() - t_start
    final_energy = spring_trace[-1] if spring_trace else float("inf")
    all_energies = nnb_trace + spring_trace
    min_energy = min(all_energies) if all_energies else float("inf")
    summary = {
        "final_energy": final_energy,
        "min_energy": min_energy,
        "elapsed_seconds": elapsed,
        "nnb_steps": len(nnb_trace),
        "pretrain_steps": pretrain_steps,
        "spring_steps": len(spring_trace),
    }
    print("\n".join(format_summary_lines(summary)), flush=True)


if __name__ == "__main__":
    main()
