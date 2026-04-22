"""MARCH — Moment-Adaptive ReConfiguration (Gu et al. 2025).

Extends SPRING (NetKet's VMC_SR with momentum) by a per-parameter adaptive
ridge weight in the SR solve. The ridge weights are learned online from
squared second-differences of the iterate trajectory:

    dθ_k = argmin  (1/λ) ‖Õ dθ' - ε̃‖²  +  ‖diag(v)^(1/4) (dθ' - φ_{k-1})‖²

state updates:
    φ_k = μ dθ_k                   (SPRING: old_updates = dθ_{k-1} in this file,
                                    μ applied at use-time to match NetKet)
    v_k = β v_{k-1} + (dθ_k - dθ_{k-1})²   (MARCH-only)

Solved via Woodbury (eq. S23 of the paper):
    dθ_k = diag(v)^(-1/2) Õ^T (Õ diag(v)^(-1/2) Õ^T + λI)^(-1)
            (ε̃ - Õ φ_{k-1}) + φ_{k-1}

Set ``moment_adaptive=False`` to recover SPRING. Set ``has_momentum=False`` on top
to recover plain MinSR/SR. The two flags are orthogonal.

Paper reference: Gu et al. 2025, arXiv:2507.02644, Eqs. S17–S23, Table S7.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from netket import jax as nkjax


class MARCHState(NamedTuple):
    """Optimizer state for MARCH / SPRING.

    All components live in the flat real-parameter space (shape ``(P_real,)``)
    regardless of whether the wavefunction uses complex parameters.  For a
    complex ansatz with ``mode='complex'`` in NetKet, ``P_real = 2 * n_complex``.

    old_updates:  Previous step's dθ_{k-1}.  μ is applied at use-time (see
                  _compute_march_update), matching NetKet's convention.
    v:            Per-parameter second moment of step differences.
    prev_updates: dθ_{k-2}, kept so that v_k = β v_{k-1} + (dθ_k - dθ_{k-1})²
                  can be evaluated inside the jitted kernel.
    """

    old_updates: jax.Array
    v: jax.Array
    prev_updates: jax.Array


def init_march_state(n_params_real: int, dtype=jnp.float64) -> MARCHState:
    """Initialize MARCH state.

    v_0 = 1 (ones) gives an initial isotropic ridge equivalent to SPRING's
    uniform λI.  v then adapts via EMA, so at step 0 MARCH reduces to SPRING
    and gradually becomes anisotropic as iterate roughness accumulates.
    This avoids the v_0 = 0 singularity in v^(-1/2).
    """
    return MARCHState(
        old_updates=jnp.zeros(n_params_real, dtype=dtype),
        v=jnp.ones(n_params_real, dtype=jnp.float64),
        prev_updates=jnp.zeros(n_params_real, dtype=dtype),
    )


@partial(
    jax.jit,
    static_argnames=("solver_fn", "mode", "moment_adaptive", "has_momentum"),
)
def _compute_march_update(
    O_L: jax.Array,
    dv: jax.Array,
    state: MARCHState,
    *,
    diag_shift: float | jax.Array,
    momentum: float | jax.Array,
    beta: float | jax.Array,
    solver_fn,
    mode: str,
    moment_adaptive: bool,
    has_momentum: bool,
    params_structure,
):
    """Single MARCH step.  Returns (updates_flat, new_state, info_dict).

    Shapes
    ------
    O_L: ``(B_eff, P_real)`` — centered jacobian from ``_prepare_input``.
    dv:  ``(B_eff,)``      — centered, scaled local-energy residuals.

    JIT contract
    ------------
    - solver_fn, mode, moment_adaptive, has_momentum are static; Python-level
      branches on them resolve at trace time, not during execution.
    - diag_shift, momentum, beta are traced scalars (allow schedules).
    - state components have fixed shape ``(P_real,)`` across all calls.
    - No data-dependent Python control flow.
    """
    old_updates = state.old_updates
    v = state.v
    prev_updates = state.prev_updates

    # --- SPRING: subtract the momentum-predicted piece from the residual.
    #     φ_{k-1} = μ * old_updates in paper notation.
    if has_momentum:
        dv = dv - momentum * (O_L @ old_updates)

    # --- MARCH: rescale Jacobian columns by v^(-1/4) before forming the Gram.
    #     This realizes the change-of-variables π = diag(v)^(1/4)(dθ' - φ_{k-1})
    #     that reduces eq. S17 to a standard ridge solve (paper S18).
    if moment_adaptive:
        v_quart_inv = jnp.power(v.astype(O_L.real.dtype), -0.25)
        O_L_scaled = O_L * v_quart_inv[None, :]
        matrix = O_L_scaled @ O_L_scaled.T
    else:
        matrix = O_L @ O_L.T

    matrix_side = matrix.shape[-1]
    shifted = matrix + diag_shift * jnp.eye(matrix_side, dtype=matrix.dtype)

    aus_vector = solver_fn(shifted, dv)
    if isinstance(aus_vector, tuple):
        aus_vector, info = aus_vector
        if info is None:
            info = {}
    else:
        info = {}

    # --- Back to parameter space.  Per eq. S23:
    #         dθ - φ_{k-1} = diag(v)^(-1/2) O^T a
    #     which factors as v^(-1/4) * (v^(-1/4) O^T a) via the Woodbury form.
    if moment_adaptive:
        updates = v_quart_inv * v_quart_inv * (O_L.T @ aus_vector)
    else:
        updates = O_L.T @ aus_vector

    if has_momentum:
        updates = updates + momentum * old_updates
        new_old_updates = updates
    else:
        new_old_updates = old_updates

    if moment_adaptive:
        diff = updates - prev_updates
        new_v = beta * v + (diff * diff.conj()).real.astype(v.dtype)
        new_prev_updates = updates
    else:
        new_v = v
        new_prev_updates = prev_updates

    new_state = MARCHState(
        old_updates=new_old_updates,
        v=new_v,
        prev_updates=new_prev_updates,
    )

    # --- Complex repacking (matches NetKet's SPRING convention).
    #     State stays in the real-flat form; only the returned update is
    #     repacked so that unravel_params_fn can reconstruct the pytree.
    if mode == "complex" and nkjax.tree_leaf_iscomplex(params_structure):
        num_p = updates.shape[-1] // 2
        updates = updates[:num_p] + 1j * updates[num_p:]

    return updates, new_state, info
