"""Robust local-energy clipping for VMC gradient stabilization.

Paper reference: Gu et al. 2025 (arXiv:2507.02644), Table S7, c = 5.0.
"""

import jax
import jax.numpy as jnp


@jax.jit
def clip_local_energies(e_loc: jnp.ndarray, c: float = 5.0) -> jnp.ndarray:
    """Cap |Re(E_loc) - mean| at c * MAD around the batch mean.

    FermiNet-style robust clipping: single-sample outliers get pulled back to
    the c * median-absolute-deviation boundary, preserving the batch mean.

    Args:
        e_loc: Per-sample local energies (real or complex).
        c:     Clip factor. Gu et al. 2025 uses 5.0.

    Returns:
        Clipped local energies with the same dtype as the input. Only the
        real part is clipped; any imaginary part (MCMC noise) is preserved.
    """
    real = e_loc.real
    mean = real.mean()
    dev = real - mean
    mad = jnp.median(jnp.abs(dev))
    clipped_real = mean + jnp.clip(dev, -c * mad, c * mad)
    if jnp.iscomplexobj(e_loc):
        return clipped_real + 1j * e_loc.imag
    return clipped_real
