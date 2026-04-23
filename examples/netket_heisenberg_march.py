"""
Faithful NetKet Heisenberg-tutorial reproduction using native MARCH.

Tutorial:   https://netket.readthedocs.io/en/latest/tutorials/gs-heisenberg.html
System:     22-site 1D Heisenberg AFM chain, PBC, total_Sz=0 sector.
Ansatz:     RBM alpha=1 (tutorial-faithful).
Sampler:    MetropolisExchange (preserves total Sz).
N_samples:  1008 (tutorial value).
Exact E0:   -39.147523 (Lanczos; also hard-coded in the tutorial).

The tutorial uses VMC_SR(diag_shift=0.1) + Sgd(lr=0.05).  This script swaps in
MARCH via NetKet's own kwargs:
    momentum        = 0.95   (SPRING μ)
    moment_adaptive = True   (MARCH)
    beta            = 0.995  (MARCH second-moment decay)
All other knobs match the tutorial. No system-size shrink; no cheating.
"""

from __future__ import annotations

import time
import jax
import jax.numpy as jnp
import numpy as np
import netket as nk

jax.config.update("jax_enable_x64", True)


def main() -> None:
    L = 22
    g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
    hi = nk.hilbert.Spin(s=0.5, total_sz=0, N=g.n_nodes)
    ha = nk.operator.Heisenberg(hilbert=hi, graph=g)

    exact_gs_energy = -39.14752260706246
    print(f"L = {L}, Hilbert sector dim = {hi.n_states}")
    print(f"Exact E0 = {exact_gs_energy:.6f}")

    ma = nk.models.RBM(alpha=1, param_dtype=jnp.complex128)
    sa = nk.sampler.MetropolisExchange(hilbert=hi, graph=g)
    vs = nk.vqs.MCState(sa, ma, n_samples=1008, seed=0, sampler_seed=0)
    print(f"RBM n_params = {vs.n_parameters}")

    n_iter = 300
    print(f"\n=== Running MARCH for {n_iter} iterations ===")
    import optax
    gs = nk.driver.VMC_SR(
        ha, optax.sgd(0.05), variational_state=vs,
        diag_shift=0.1,
        momentum=0.95,
        moment_adaptive=True, beta=0.995,
        use_ntk=True, on_the_fly=False, mode="complex",
    )
    logger = nk.logging.RuntimeLog()
    t0 = time.perf_counter()
    gs.run(n_iter=n_iter, out=logger, show_progress=False)
    elapsed = time.perf_counter() - t0

    energies = np.real(logger.data["Energy"]["Mean"])
    final = float(energies[-1])
    best = float(energies.min())
    rel_err = abs(final - exact_gs_energy) / abs(exact_gs_energy)
    print("\n--- Summary ---")
    print(f"Iterations:     {len(energies)}")
    print(f"Wall clock:     {elapsed:.1f}s")
    print(f"Exact E0:       {exact_gs_energy:.6f}")
    print(f"MARCH final:    {final:.6f}")
    print(f"MARCH best:     {best:.6f}")
    print(f"Relative err:   {rel_err:.3e}")
    print(f"Accuracy:       {(1 - rel_err) * 100:.3f}%")


if __name__ == "__main__":
    main()
