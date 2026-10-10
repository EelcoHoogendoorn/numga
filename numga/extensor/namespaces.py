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


class SitesNamespace:
    """Reductions over the sites of an extensor's output: a field summed or averaged over its sites,
    where `.sum()` and `.mean()` reduce batch axes, the independent copies."""

    def __init__(self, extensor: Extensor) -> None:
        sites = dict(extensor.gatype.fields)
        if 0 not in sites:
            raise TypeError(f"the output of {extensor.gatype.signature} does not range over sites")
        self.extensor = extensor
        self.count = sites[0]

    def sum(self) -> Extensor:
        """The sum over the sites of the output."""
        extensor = self.extensor
        kernel = extensor.context.xp.sum(extensor.kernel, axis=extensor.ndim)
        fields = tuple((slot, count) for slot, count in extensor.gatype.fields if slot)
        gatype = extensor.gatype.derive.structural.derive.with_fields(fields)
        return type(extensor)._from_prepared_kernel(extensor.context, gatype, kernel)

    def mean(self) -> Extensor:
        """The mean over the sites of the output."""
        return self.sum() / self.count
