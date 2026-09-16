"""Small, context-free operations on canonical integer blade masks."""

from __future__ import annotations

from operator import index
from typing import Iterable

import numpy as np


_bit_count_radix = (
    np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1)
    .sum(axis=1)
    .astype(np.uint8)
)


def bit_count(b: int | np.ndarray) -> int | np.ndarray:
    """Count the set bits in an integer or array of integers."""
    if isinstance(b, (int, np.integer)):
        return int(b).bit_count()
    b = np.asarray(b)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(b)
    raw = b.view(np.uint8).reshape(b.shape + (-1,))
    return _bit_count_radix[raw].sum(axis=-1).astype(np.uint8)


def parity_to_sign(parity: int | np.ndarray) -> int | np.ndarray:
    """Map even/odd parity to ``+1``/``-1``."""
    if isinstance(parity, (int, np.integer)):
        return 1 if index(parity) % 2 == 0 else -1
    return (1 - (np.asarray(parity) % 2) * 2).astype(np.int8)


parity_sign = parity_to_sign


def permutation_sign(values: Iterable[int]) -> int:
    """Return the sign of the permutation represented by unique integers."""

    items = tuple(index(value) for value in values)
    if len(set(items)) != len(items):
        raise ValueError("a permutation cannot contain duplicate entries")
    inversions = sum(
        left > right
        for position, left in enumerate(items)
        for right in items[position + 1 :]
    )
    return parity_sign(inversions)


def mask_from_indices(indices: Iterable[int]) -> int:
    """Pack distinct, non-negative generator indices into a blade mask."""

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
    """Return the smallest NumPy dtype able to store this algebra's masks."""

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

