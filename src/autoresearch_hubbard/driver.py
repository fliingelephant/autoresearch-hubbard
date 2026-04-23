"""MARCH / SPRING training driver — thin wrapper over netket's VMC_SR internals.

Orchestrates:  sample → local_energies → clip → netket's SR/MARCH solve → update

Use when you need wall-clock budgeting, FermiNet-style local-energy clipping,
or per-step norm constraint — features the public ``nk.driver.VMC_SR`` does not
yet expose. For everything else, use ``nk.driver.VMC_SR(moment_adaptive=True)``
directly; this wrapper calls exactly the same kernels under the hood.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import netket as nk
from netket.optimizer.solver import cholesky_with_fallback
from netket._src.ngd.sr_srt_common import get_samples_and_pdf, srt
from netket._src.ngd.srt_onthefly import srt_onthefly

from autoresearch_hubbard.clip import clip_local_energies


def _norm_clip(updates_pytree, bound: float | None):
    """Rescale ``updates_pytree`` so its flat L2 norm ≤ bound. bound=None disables."""
    if bound is None:
        return updates_pytree
    flat, unravel = ravel_pytree(updates_pytree)
    norm = jnp.linalg.norm(flat)
    factor = jnp.minimum(1.0, bound / (norm + 1e-12))
    return unravel(flat * factor)


def run_march_phase(
    state: nk.vqs.MCState,
    hamiltonian,
    *,
    seconds_budget: float | None = None,
    n_iter: int | None = None,
    learning_rate: float = 1e-2,
    diag_shift: float = 1e-3,
    momentum: float | None = 0.95,
    moment_adaptive: bool = True,
    beta: float = 0.995,
    clip_c: float = 5.0,
    norm_bound_fn: Callable[[int], float] | None = None,
    mode: str = "complex",
    solver_fn=cholesky_with_fallback,
    on_the_fly: bool = True,
    log_every: int = 10,
    phase_name: str = "march",
) -> list[float]:
    """Run a MARCH (or SPRING / MinSR) phase under a wall-clock or iteration budget.

    Parameters
    ----------
    on_the_fly:
        If True (default), uses ``srt_onthefly`` — no Jacobian materialized
        (memory-optimal for large models). If False, uses ``srt`` (dense NTK).
    momentum:
        SPRING coefficient μ. ``None`` disables momentum (MinSR / pure MARCH).
    moment_adaptive:
        If True, enables MARCH per-parameter adaptive ridge (Gu et al. 2025).
        Orthogonal to ``momentum``: all four combinations are valid.
    beta:
        EMA decay for the MARCH second-moment estimator. Paper: 0.995.
    clip_c:
        FermiNet-style local-energy clip factor (paper Table S7: 5.0).
    norm_bound_fn:
        Optional ``step → max_norm`` schedule for the per-step update.
        ``None`` disables.

    Budget: exactly one of ``seconds_budget`` (wall-clock) or ``n_iter`` must
    be set. The first step runs outside the timer as JIT warmup.

    Returns
    -------
    Trace of per-step mean energies (real scalar per iteration).
    """
    if (seconds_budget is None) == (n_iter is None):
        raise ValueError("Exactly one of seconds_budget or n_iter must be set.")

    compute_update = srt_onthefly if on_the_fly else srt

    old_updates = None
    v = None
    prev_updates = None
    trace: list[float] = []

    def one_step(old_updates, v, prev_updates, step):
        state.sample()
        samples, pdf = get_samples_and_pdf(state)

        e_loc_raw = state.local_estimators(hamiltonian)
        e_loc = clip_local_energies(e_loc_raw, c=clip_c)

        updates, old_updates, v, prev_updates, _info = compute_update(
            state._apply_fun,
            e_loc,
            state.parameters,
            state.model_state,
            samples,
            diag_shift=diag_shift,
            solver_fn=solver_fn,
            mode=mode,
            momentum=momentum,
            old_updates=old_updates,
            moment_adaptive=moment_adaptive,
            beta=beta,
            v=v,
            prev_updates=prev_updates,
            weights=pdf,
        )

        if norm_bound_fn is not None:
            updates = _norm_clip(updates, norm_bound_fn(step))

        state.parameters = jax.tree.map(
            lambda p, u: p - learning_rate * u, state.parameters, updates
        )
        return float(jnp.real(e_loc_raw.mean())), old_updates, v, prev_updates

    # --- Warmup step (JIT compile) outside the timer
    t_warm = time.perf_counter()
    energy, old_updates, v, prev_updates = one_step(old_updates, v, prev_updates, step=0)
    trace.append(energy)
    print(
        f"{phase_name} step 1 (warmup, {time.perf_counter() - t_warm:.1f}s): "
        f"energy={energy:.6f}",
        flush=True,
    )

    # --- Budgeted loop
    step = 1
    if seconds_budget is not None:
        deadline = time.perf_counter() + seconds_budget
        should_continue = lambda: time.perf_counter() < deadline
    else:
        should_continue = lambda: step < n_iter

    while should_continue():
        energy, old_updates, v, prev_updates = one_step(
            old_updates, v, prev_updates, step=step
        )
        step += 1
        trace.append(energy)
        if step % log_every == 0:
            print(f"{phase_name} step {step}: energy={energy:.6f}", flush=True)
    return trace
