"""MARCH / SPRING training driver.

Minimal replacement for ``nk.driver.VMC_SR`` that runs one SR-flavored step
per iteration:

    sample → local energies → clip → Jacobian → MARCH solve → param update

Set ``moment_adaptive=False`` to fall back to SPRING, ``momentum=None`` to
fall back to MinSR/SR, or both to disable everything (plain SR).

The loop is written as a Python ``while`` with a wall-clock budget; each
iteration calls a jitted kernel for the SR solve.  No ``jax.lax.scan`` — the
MCMC sampler drives the loop and its state lives outside jax.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import netket as nk
from netket import jax as nkjax
from netket._src.ngd.sr_srt_common import _prepare_input, get_samples_and_pdf
from netket.optimizer.solver import cholesky_with_fallback

from autoresearch_hubbard.clip import clip_local_energies
from autoresearch_hubbard.march import MARCHState, init_march_state, _compute_march_update


def _apply_updates(parameters, updates_pytree, learning_rate: float):
    """θ ← θ - lr · dθ.  Returns the new parameter pytree."""
    return jax.tree.map(
        lambda p, u: p - learning_rate * u, parameters, updates_pytree
    )


def _norm_clip(updates_pytree, bound: float | None):
    """Rescale ``updates_pytree`` so that its flat L2 norm is ≤ bound.
    ``bound=None`` disables clipping."""
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
    log_every: int = 10,
    phase_name: str = "march",
) -> list[float]:
    """Run a MARCH (or SPRING / MinSR) phase.

    Budget: exactly one of ``seconds_budget`` (wall-clock) or ``n_iter``
    (iteration count) must be set.  The first step runs outside the timer
    as JIT warmup.

    Parameters
    ----------
    momentum:
        SPRING coefficient μ.  ``None`` disables momentum (plain MinSR/SR).
    moment_adaptive:
        If True, use MARCH's per-parameter adaptive ridge.  Orthogonal to
        ``momentum``: all four combinations are valid.
    beta:
        EMA decay for the MARCH second-moment estimator.  Ignored if
        ``moment_adaptive=False``.
    clip_c:
        FermiNet-style local-energy clipping factor (paper Table S7: 5.0).
    norm_bound_fn:
        Optional ``step → max_norm`` schedule implementing paper Table S7's
        norm constraint.  None to disable.

    Returns
    -------
    Trace of per-step mean energies (real scalar per iteration).
    """
    has_momentum = momentum is not None
    mu = 0.0 if momentum is None else momentum
    params_structure = jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), state.parameters
    )

    # NetKet's internal jacobian plumbing calls apply_fun(variables, samples),
    # where it has already assembled variables = {"params": reassemble(W), **ms}.
    def log_psi(variables, samples):
        return state.model.apply(variables, samples)

    march_state: MARCHState | None = None
    trace: list[float] = []

    def one_step(march_state, step):
        """One full training iteration.  Mutates ``state`` in place (samples
        and parameters).  Returns (energy_mean, march_state)."""
        state.sample()
        samples, pdf = get_samples_and_pdf(state)

        e_loc_raw = state.local_estimators(hamiltonian)
        e_loc = clip_local_energies(e_loc_raw, c=clip_c)

        jacobians = nkjax.jacobian(
            log_psi,
            state.parameters,
            samples,
            state.model_state,
            mode=mode,
            dense=True,
            center=True,
            pdf=pdf,
        )

        n_samples = samples.shape[0]
        mass = 1.0 / n_samples
        O_L, dv = _prepare_input(jacobians, e_loc, mode=mode, scaling_factor=mass)

        if march_state is None:
            march_state = init_march_state(O_L.shape[-1], dtype=O_L.dtype)

        updates_flat, march_state, _info = _compute_march_update(
            O_L, dv, march_state,
            diag_shift=diag_shift,
            momentum=mu,
            beta=beta,
            solver_fn=solver_fn,
            mode=mode,
            moment_adaptive=moment_adaptive,
            has_momentum=has_momentum,
            params_structure=params_structure,
        )

        # Unravel the flat update to match the parameter pytree.
        _, unravel_params = ravel_pytree(state.parameters)
        updates_pytree = unravel_params(updates_flat)

        # Per-step norm constraint (Table S7).
        if norm_bound_fn is not None:
            updates_pytree = _norm_clip(updates_pytree, norm_bound_fn(step))

        state.parameters = _apply_updates(
            state.parameters, updates_pytree, learning_rate
        )

        return float(jnp.real(e_loc_raw.mean())), march_state

    if (seconds_budget is None) == (n_iter is None):
        raise ValueError("Exactly one of seconds_budget or n_iter must be set.")

    # --- Warmup step (JIT compile) outside the timer
    t_warm = time.perf_counter()
    energy, march_state = one_step(march_state, step=0)
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
        should_continue = lambda: step < n_iter  # warmup already counted step 1

    while should_continue():
        energy, march_state = one_step(march_state, step=step)
        step += 1
        trace.append(energy)
        if step % log_every == 0:
            print(
                f"{phase_name} step {step}: energy={energy:.6f}",
                flush=True,
            )
    return trace
