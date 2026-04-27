"""Standalone 4x4 Hubbard smoke optimization for the exterior ansatz.

This script intentionally avoids the frozen 16x4 training path. It is a small
trial runner that prints intermediate VMC energies so we can see whether the
exterior amplitude evaluator initializes, samples, and optimizes.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import netket as nk
import optax
from netket.utils import struct

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoresearch_hubbard.hamiltonian import build_hamiltonian
from exterior.model import ExteriorAmplitude


class EnergyPrinter(nk.callbacks.AbstractCallback):
    trace: list = struct.field(pytree_node=False, serialize=False)
    variance_trace: list = struct.field(pytree_node=False, serialize=False)

    def __init__(self):
        super().__init__()
        self.trace = []
        self.variance_trace = []

    def on_step_end(self, step, log_data, driver):
        energy = log_data["Energy"]
        mean = float(jnp.real(energy.mean))
        variance = float(jnp.real(energy.variance))
        if not math.isfinite(mean):
            raise FloatingPointError(f"step {step + 1}: non-finite energy {mean}")
        self.trace.append(mean)
        self.variance_trace.append(variance)
        print(f"step {step + 1:03d}: energy={mean:.8f} variance={variance:.8f}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--samples", type=int, default=2592)
    parser.add_argument("--chains", type=int, default=48)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--diag-shift", type=float, default=0.1)
    parser.add_argument("--sr", choices=("minsr", "sr"), default="minsr")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jax.config.update("jax_enable_x64", True)

    hamiltonian, hilbert, graph = build_hamiltonian(Lx=4, Ly=4, U=8.0, t=1.0, pbc=False)
    sampler = nk.sampler.MetropolisFermionHop(
        hilbert,
        graph=graph,
        n_chains=args.chains,
        sweep_size=40,
        spin_symmetric=True,
    )
    model = ExteriorAmplitude(
        hilbert,
        d_model=64,
        n_heads=4,
        n_layers=2,
        ext_dim=6,
        ext_channels=8,
        param_dtype=jnp.float64,
    )
    state = nk.vqs.MCState(
        sampler,
        model,
        n_samples=args.samples,
        seed=args.seed,
        sampler_seed=args.seed,
    )

    print(
        "4x4 OBC Hubbard exterior trial: "
        f"samples={args.samples} chains={args.chains} lr={args.lr} "
        f"diag_shift={args.diag_shift} sr={args.sr}",
        flush=True,
    )
    print(
        "model: ExteriorAmplitude("
        "site_tokens,d_model=64,n_heads=4,n_layers=2,ext_dim=6,ext_channels=8)",
        flush=True,
    )

    use_minsr = args.sr == "minsr"
    driver = nk.driver.VMC_SR(
        hamiltonian,
        optax.sgd(args.lr),
        variational_state=state,
        diag_shift=args.diag_shift,
        use_ntk=use_minsr,
        on_the_fly=use_minsr,
        mode="complex",
    )
    energy_printer = EnergyPrinter()
    driver.run(
        n_iter=args.iters,
        out=(),
        show_progress=False,
        callback=energy_printer,
    )

    trace = energy_printer.trace
    print("---", flush=True)
    print(f"initial_energy: {trace[0]:.8f}", flush=True)
    print(f"final_energy:   {trace[-1]:.8f}", flush=True)
    print(f"min_energy:     {min(trace):.8f}", flush=True)


if __name__ == "__main__":
    main()
