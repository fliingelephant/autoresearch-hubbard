"""Small truncated exterior algebra for latent grades 0, 1, and 2."""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations

import jax
import jax.numpy as jnp


Blade = tuple[int, ...]


def basis_blades(ext_dim: int) -> tuple[Blade, ...]:
    """Return ordered blades for grades 0, 1, and 2."""
    vectors = tuple((i,) for i in range(ext_dim))
    bivectors = tuple(combinations(range(ext_dim), 2))
    return ((),) + vectors + bivectors


def blade_index(blades: tuple[Blade, ...], blade: Blade) -> int:
    return blades.index(tuple(blade))


def _permutation_sign(values: tuple[int, ...]) -> int:
    inversions = 0
    for i, left in enumerate(values):
        for right in values[i + 1 :]:
            inversions += int(left > right)
    return -1 if inversions % 2 else 1


def _wedge_blades(left: Blade, right: Blade, max_grade: int) -> tuple[Blade, int] | None:
    if set(left).intersection(right):
        return None
    out = tuple(sorted(left + right))
    if len(out) > max_grade:
        return None
    return out, _permutation_sign(left + right)


def _interior_blades(left: Blade, right: Blade, max_grade: int) -> tuple[Blade, int] | None:
    if len(left) > len(right) or not set(left).issubset(right):
        return None
    out = tuple(item for item in right if item not in left)
    if len(out) > max_grade:
        return None
    wedge = _wedge_blades(left, out, max_grade)
    if wedge is None:
        return None
    _, sign = wedge
    return out, sign


@lru_cache(maxsize=None)
def _product_table(ext_dim: int, kind: str) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    blades = basis_blades(ext_dim)
    max_grade = max(len(blade) for blade in blades)
    lhs: list[int] = []
    rhs: list[int] = []
    out: list[int] = []
    signs: list[float] = []
    product = _wedge_blades if kind == "wedge" else _interior_blades

    for i, left in enumerate(blades):
        for j, right in enumerate(blades):
            result = product(left, right, max_grade)
            if result is None:
                continue
            blade, sign = result
            lhs.append(i)
            rhs.append(j)
            out.append(blade_index(blades, blade))
            signs.append(float(sign))

    return (
        jnp.asarray(lhs, dtype=jnp.int32),
        jnp.asarray(rhs, dtype=jnp.int32),
        jnp.asarray(out, dtype=jnp.int32),
        jnp.asarray(signs),
    )


def _bilinear_product(left: jax.Array, right: jax.Array, *, ext_dim: int, kind: str) -> jax.Array:
    lhs, rhs, out, signs = _product_table(ext_dim, kind)
    terms = left[..., lhs] * right[..., rhs] * signs.astype(left.dtype)
    result = jnp.zeros(left.shape[:-1] + (len(basis_blades(ext_dim)),), dtype=left.dtype)
    return result.at[..., out].add(terms)


def wedge(left: jax.Array, right: jax.Array, *, ext_dim: int) -> jax.Array:
    """Truncated wedge product on arrays with blade axis last."""
    return _bilinear_product(left, right, ext_dim=ext_dim, kind="wedge")


def interior(left: jax.Array, right: jax.Array, *, ext_dim: int) -> jax.Array:
    """Left interior product on arrays with blade axis last."""
    return _bilinear_product(left, right, ext_dim=ext_dim, kind="interior")


def scalar_pair(left: jax.Array, right: jax.Array) -> jax.Array:
    """Orthonormal coefficient pairing over the stored blades."""
    return jnp.sum(left * right, axis=-1)
