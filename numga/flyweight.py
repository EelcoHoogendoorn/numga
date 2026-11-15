"""Per-factory strong flyweight pools intern canonical structural objects by key without coupling
value classes to allocation policy. Each factory owns its dictionary pool over the lifetime of its
parent `Algebra`, while the pooled classes (`SubSpace` and `GAType`) retain structural equality and
hashing when constructed directly."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, Hashable, TypeVar


Value = TypeVar("Value")


class FlyweightFactory(Generic[Value]):
    """Subclasses cache constructed instances strongly by hashable key for the lifetime of the
    factory."""

    __slots__ = ("_flyweight_pool",)

    def __init__(self) -> None:
        object.__setattr__(self, "_flyweight_pool", {})

    def factory_construct(
        self,
        key: Hashable,
        construct: Callable,
    ) -> Value:
        """Repeated lookups of `key` return the first instance produced by `construct()`."""
        try:
            return self._flyweight_pool[key]
        except KeyError:
            value = construct()
            self._flyweight_pool[key] = value
            return value

    @property
    def flyweight_count(self) -> int:
        return len(self._flyweight_pool)
