"""Paper-fidelity extensions around ``nk.driver.VMC_SR``.

Two thin pieces that plug into netket's standard optimization loop:

* :class:`VMC_SR_clipped` — subclass that applies FermiNet-style local-energy
  clipping between ``local_estimators`` and the SR/MARCH solve, via netket's
  ``_compute_local_energies`` hook.
* :class:`NormSchedule` — callback that enforces a step-dependent L2 bound on
  the parameter update (paper Table S7), via ``before_parameter_update``.

Combine with :class:`nk.callbacks.Timeout` for wall-clock budgeted runs.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax

import netket as nk
from netket.utils import struct

from autoresearch_hubbard.clip import clip_local_energies


class VMC_SR_clipped(nk.driver.VMC_SR):
    """``VMC_SR`` with FermiNet-style local-energy clipping.

    See :mod:`autoresearch_hubbard.clip` for the clipping specification.
    """

    _clip_c: float = struct.field(pytree_node=False, serialize=False, default=5.0)

    def __init__(self, *args, clip_c: float = 5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._clip_c = clip_c

    def _compute_local_energies(self):
        return clip_local_energies(super()._compute_local_energies(), c=self._clip_c)


class NormSchedule(nk.callbacks.AbstractCallback):
    """Cap the L2 norm of ``driver._dp`` at ``norm_bound_fn(step)`` per step.

    Reproduces the paper Table S7 norm-constraint schedule.
    """

    def __init__(self, norm_bound_fn: Callable[[int], float]):
        super().__init__()
        self.norm_bound_fn = norm_bound_fn

    def before_parameter_update(self, step, log_data, driver):
        bound = self.norm_bound_fn(step)
        factor = jnp.minimum(1.0, bound / (optax.global_norm(driver._dp) + 1e-12))
        driver._dp = jax.tree.map(lambda t: t * factor, driver._dp)
