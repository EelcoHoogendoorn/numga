"""Grade and named-SubSpace selection on an Extensor's output axis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from numga.subspace import SubSpace

if TYPE_CHECKING:
    from numga.extensor.extensor import Extensor


class SelectNamespace:
    def __init__(self, extensor: Extensor) -> None:
        self.extensor = extensor
        self.factory = extensor.algebra.subspace

    def _select(self, subspace: SubSpace) -> Extensor:
        return self.extensor.select_subspace(subspace)

    def __getattr__(self, name: str) -> Extensor | Callable[..., Extensor]:
        constructor = getattr(self.factory, name)
        if isinstance(constructor, SubSpace):
            return self._select(constructor)
        return lambda *args, **kwargs: self._select(constructor(*args, **kwargs))

    def __getitem__(self, grades: int | tuple[int, ...]) -> Extensor:
        grades = grades if isinstance(grades, tuple) else (grades,)
        return self._select(self.factory.from_grades(grades))


class RestrictNamespace(SelectNamespace):
    def _select(self, subspace: SubSpace) -> Extensor:
        return self.extensor.restrict_subspace(subspace)
