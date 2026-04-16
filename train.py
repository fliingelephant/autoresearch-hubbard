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

import flax.linen as nn
import jax
import jax.numpy as jnp
import netket as nk
import optax

from prepare import TRIAL_SECONDS, build_system, verify_frozen_surface

SEED = 0

jax.config.update("jax_enable_x64", True)
DTYPE = jnp.float64

# Sampler
N_CHAINS = 48
SWEEP_SIZE = 40
N_SAMPLES = 2560

# SPRING
SPRING_LR = 0.03
DIAG_SHIFT = 1e-3
MOMENTUM = 0.9

# Logging cadence
LOG_EVERY_VMC = 25

N_DETERMINANTS = 8
RBM_ALPHA = 1
MODEL_ID = f"MultiSlater(K={N_DETERMINANTS})+Jastrow+RBM(a={RBM_ALPHA})"


class MultiSlaterJastrow(nn.Module):
    hilbert: nk.hilbert.SpinOrbitalFermions
    n_determinants: int = 2
    rbm_alpha: int = 1

    @nn.compact
    def __call__(self, x):
        log_slater = nk.models.MultiSlater2nd(
            self.hilbert, n_determinants=self.n_determinants,
            generalized=False, restricted=False,
            param_dtype=jnp.complex128,
        )(x)
        log_jastrow = nk.models.Jastrow(param_dtype=jnp.complex128)(x)
        log_rbm = nk.models.RBM(alpha=self.rbm_alpha, param_dtype=jnp.complex128)(x)
        return log_slater + log_jastrow + log_rbm


class NaNError(RuntimeError):
    pass


def _check_finite(value: float, phase: str, step: int, what: str) -> None:
    if not math.isfinite(value):
        raise NaNError(f"{phase} step {step}: {what}={value} (training diverged)")


def run_vmc_phase(driver, seconds_budget: float, phase_name: str, log_every: int) -> list[float]:
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


def main() -> None:
    verify_frozen_surface()
    t_start = time.perf_counter()
    print(f"Trial budget: {TRIAL_SECONDS}s  (single SPRING phase)", flush=True)

    hamiltonian, hilbert, graph = build_system()
    sampler = nk.sampler.MetropolisFermionHop(
        hilbert, graph=graph, n_chains=N_CHAINS, sweep_size=SWEEP_SIZE, spin_symmetric=True,
    )

    model = MultiSlaterJastrow(hilbert, n_determinants=N_DETERMINANTS, rbm_alpha=RBM_ALPHA)
    state = nk.vqs.MCState(
        sampler, model, n_samples=N_SAMPLES, seed=SEED, sampler_seed=SEED,
    )
    print(f"Model: {MODEL_ID}, n_params={state.n_parameters}", flush=True)

    print("=== SPRING VMC ===", flush=True)
    spring_driver = nk.driver.VMC_SR(
        hamiltonian, optax.sgd(SPRING_LR), variational_state=state,
        diag_shift=DIAG_SHIFT, momentum=None, mode="complex",
        use_ntk=True, on_the_fly=True,
    )
    spring_trace = run_vmc_phase(spring_driver, TRIAL_SECONDS, "spring", LOG_EVERY_VMC)

    elapsed = time.perf_counter() - t_start
    final_energy = spring_trace[-1] if spring_trace else float("inf")
    min_energy = min(spring_trace) if spring_trace else float("inf")

    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    print(
        f"---\n"
        f"final_energy:    {final_energy:.6f}\n"
        f"min_energy:      {min_energy:.6f}\n"
        f"elapsed_seconds: {elapsed:.1f}\n"
        f"nnb_steps:       0\n"
        f"pretrain_steps:  0\n"
        f"spring_steps:    {len(spring_trace)}\n"
        f"tsv_entry:       {timestamp}\t{final_energy:.6f}\t{MODEL_ID}\t"
        f"{len(spring_trace)}\t{elapsed:.1f}\t[STATUS]\t[DESC]",
        flush=True,
    )


if __name__ == "__main__":
    main()
