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
from exterior.model import CascadedExteriorGramAmplitude, ExteriorAmplitude


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
    parser.add_argument("--ansatz", choices=("cascaded", "latent"), default="cascaded")
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--geom-dim", type=int, default=8)
    parser.add_argument("--projection-channels", type=int, default=2)
    parser.add_argument("--gram-eps", type=float, default=1.0e-4)
    parser.add_argument("--ext-dim", type=int, default=6)
    parser.add_argument("--ext-channels", type=int, default=8)
    return parser.parse_args()


def build_model(args: argparse.Namespace, hilbert):
    if args.ansatz == "cascaded":
        return CascadedExteriorGramAmplitude(
            hilbert,
            d_model=args.d_model,
            n_heads=args.heads,
            n_layers=args.layers,
            n_groups=args.groups,
            group_size=args.group_size,
            geom_dim=args.geom_dim,
            projection_channels=args.projection_channels,
            gram_eps=args.gram_eps,
            param_dtype=jnp.float64,
        )

    return ExteriorAmplitude(
        hilbert,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        ext_dim=args.ext_dim,
        ext_channels=args.ext_channels,
        param_dtype=jnp.float64,
    )


def describe_model(args: argparse.Namespace) -> str:
    if args.ansatz == "cascaded":
        return (
            "CascadedExteriorGramAmplitude("
            f"physical_modes,d_model={args.d_model},n_heads={args.heads},"
            f"n_layers={args.layers},groups={args.groups},"
            f"group_size={args.group_size},geom_dim={args.geom_dim},"
            f"projection_channels={args.projection_channels})"
        )

    return (
        "ExteriorAmplitude("
        f"site_tokens,d_model={args.d_model},n_heads={args.heads},"
        f"n_layers={args.layers},ext_dim={args.ext_dim},"
        f"ext_channels={args.ext_channels})"
    )


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
    model = build_model(args, hilbert)
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
        f"diag_shift={args.diag_shift} sr={args.sr} ansatz={args.ansatz}",
        flush=True,
    )
    print(f"model: {describe_model(args)}", flush=True)
    print(f"parameters: {state.n_parameters}", flush=True)

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
