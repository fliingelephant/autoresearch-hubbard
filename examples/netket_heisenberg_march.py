"""
Faithful NetKet Heisenberg-tutorial reproduction using our MARCH driver.

Tutorial:   https://netket.readthedocs.io/en/latest/tutorials/gs-heisenberg.html
System:     22-site 1D Heisenberg AFM chain, PBC, total_Sz=0 sector.
Ansatz:     RBM alpha=1 (tutorial-faithful).
Sampler:    MetropolisExchange (preserves total Sz).
N_samples:  1008 (tutorial value).
Exact E0:   -39.147523 (Lanczos; also hard-coded in the tutorial).

The tutorial uses VMC_SR(diag_shift=0.1) + Sgd(lr=0.05).  This script swaps in
our MARCH driver with:
    momentum   = 0.95  (SPRING / μ)
    beta       = 0.995 (MARCH / β)
    moment_adaptive = True
    clip_c     = 5.0   (FermiNet-style local-energy clip)
All other knobs match the tutorial.  No system-size shrink; no cheating.
"""

from __future__ import annotations

import time
import jax
import jax.numpy as jnp
import netket as nk

from autoresearch_hubbard.driver import run_march_phase

jax.config.update("jax_enable_x64", True)


def main() -> None:
    # --- Tutorial setup (verbatim) -----------------------------------------
    L = 22
    g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
    hi = nk.hilbert.Spin(s=0.5, total_sz=0, N=g.n_nodes)
    ha = nk.operator.Heisenberg(hilbert=hi, graph=g)

    # Exact Lanczos ground-state energy (hard-coded in the tutorial; also
    # recomputable via nk.exact.lanczos_ed(ha, compute_eigenvectors=False)).
    exact_gs_energy = -39.14752260706246
    print(f"L = {L}, Hilbert sector dim = {hi.n_states}")
    print(f"Exact E0 = {exact_gs_energy:.6f}")

    # --- RBM ansatz (tutorial: alpha=1) ------------------------------------
    ma = nk.models.RBM(alpha=1, param_dtype=jnp.complex128)

    sa = nk.sampler.MetropolisExchange(hilbert=hi, graph=g)
    vs = nk.vqs.MCState(sa, ma, n_samples=1008, seed=0, sampler_seed=0)
    print(f"RBM n_params = {vs.n_parameters}")

    # --- MARCH run ----------------------------------------------------------
    n_iter = 300
    print(f"\n=== Running MARCH for {n_iter} iterations ===")
    t0 = time.perf_counter()
    trace = run_march_phase(
        vs, ha,
        n_iter=n_iter,
        learning_rate=0.05,     # same as tutorial's Sgd(lr=0.05)
        diag_shift=0.1,         # same as tutorial's VMC_SR(diag_shift=0.1)
        momentum=0.95,          # SPRING μ
        moment_adaptive=True,   # MARCH on top
        beta=0.995,             # MARCH β
        clip_c=5.0,             # FermiNet-style clip
        norm_bound_fn=None,     # tutorial doesn't clip gradients
        mode="complex",
        log_every=20,
        phase_name="heisenberg",
    )
    elapsed = time.perf_counter() - t0

    final = trace[-1]
    best = min(trace)
    rel_err = abs(final - exact_gs_energy) / abs(exact_gs_energy)
    print("\n--- Summary ---")
    print(f"Iterations:     {len(trace)}")
    print(f"Wall clock:     {elapsed:.1f}s")
    print(f"Exact E0:       {exact_gs_energy:.6f}")
    print(f"MARCH final:    {final:.6f}")
    print(f"MARCH best:     {best:.6f}")
    print(f"Relative err:   {rel_err:.3e}")
    print(f"Accuracy:       {(1 - rel_err) * 100:.3f}%")


if __name__ == "__main__":
    main()
