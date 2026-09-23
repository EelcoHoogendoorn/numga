"""Small, explicit strong-pool flyweight support.

Factories own their pools.  Value classes remain unaware of allocation policy
and retain structural equality as their correctness contract.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, Hashable, TypeVar


Value = TypeVar("Value")


class FlyweightFactory(Generic[Value]):
    """Base class for a factory-scoped, strong flyweight pool."""

    __slots__ = ("_flyweight_pool",)

    def __init__(self) -> None:
        object.__setattr__(self, "_flyweight_pool", {})

    def factory_construct(
        self,
        key: Hashable,
        construct: Callable[[], Value],
    ) -> Value:
        try:
            return self._flyweight_pool[key]
        except KeyError:
            value = construct()
            self._flyweight_pool[key] = value
            return value

    @property
    def flyweight_count(self) -> int:
        return len(self._flyweight_pool)
