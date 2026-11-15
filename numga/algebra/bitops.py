"""Basis blades in an orthogonal geometric algebra map to non-negative integer bit masks in which
bit `i` marks the presence of generator `i` up to 64 dimensions. Scalar and NumPy-array bit
operations count blade grade via population count, convert inversion parities into `+1` and `-1`
orientation signs, pack generator index sequences into canonical masks, and select the narrowest
unsigned integer dtype (`uint8` through `uint64`) able to hold every blade mask of a given
dimension."""

from __future__ import annotations

from collections.abc import Iterable
from operator import index

import numpy as np


_bit_count_radix = (
    np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1)
    .sum(axis=1)
    .astype(np.uint8)
)


def bit_count(masks: int | np.ndarray) -> int | np.ndarray:
    """Population count gives the grade of a single integer blade mask or an array of blade masks."""
    if isinstance(masks, (int, np.integer)):
        return int(masks).bit_count()
    masks = np.asarray(masks)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(masks)
    raw = masks.view(np.uint8).reshape(masks.shape + (-1,))
    return _bit_count_radix[raw].sum(axis=-1).astype(np.uint8)


def parity_to_sign(parity: int | np.ndarray) -> int | np.ndarray:
    """Even parities map to `+1` and odd parities map to `-1`."""
    if isinstance(parity, (int, np.integer)):
        return 1 if index(parity) % 2 == 0 else -1
    return (1 - (np.asarray(parity) % 2) * 2).astype(np.int8)


parity_sign = parity_to_sign


def permutation_sign(entries: Iterable[int]) -> int:
    """Pairwise inversions of distinct integer entries determine `+1` for even permutations and
    `-1` for odd permutations."""
    items = tuple(index(value) for value in entries)
    if len(set(items)) != len(items):
        raise ValueError("a permutation cannot contain duplicate entries")
    inversions = sum(
        left > right
        for position, left in enumerate(items)
        for right in items[position + 1 :]
    )
    return parity_sign(inversions)


def mask_from_indices(indices: Iterable[int]) -> int:
    """Distinct non-negative generator indices pack into a canonical bit mask with bit `i` set for
    generator `i`."""
    result = 0
    for value in indices:
        generator = index(value)
        if generator < 0:
            raise ValueError("generator indices must be non-negative")
        bit = 1 << generator
        if result & bit:
            raise ValueError("a basis blade cannot repeat a generator")
        result |= bit
    return result


def unsigned_dtype(dimension: int) -> np.dtype:
    """An algebra of `dimension` generators fits its canonical blade masks in the narrowest
    unsigned NumPy integer type up to `uint64`."""
    dimension = index(dimension)
    if dimension < 0:
        raise ValueError("dimension must be non-negative")
    if dimension <= 8:
        return np.dtype(np.uint8)
    if dimension <= 16:
        return np.dtype(np.uint16)
    if dimension <= 32:
        return np.dtype(np.uint32)
    if dimension <= 64:
        return np.dtype(np.uint64)
    raise ValueError("NumPy blade-mask arrays support at most 64 generators")

