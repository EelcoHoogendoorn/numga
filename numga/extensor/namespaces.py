"""Bound property namespaces on `Extensor` group output-subspace projections (`extensor.select` and
`extensor.restrict`) and field-site reductions (`extensor.sites`) separately from batch operations.
Both `SelectNamespace` and `RestrictNamespace` expose the algebra's `SubSpace` constructors by
attribute lookup (`extensor.select.bivector()`) and grade selection by indexing (`extensor.select[2]`):
`SelectNamespace` delegates to `select_subspace`, zero-padding absent target blades, whereas
`RestrictNamespace` delegates to `restrict_subspace`, intersecting the target with the extensor's
existing support.

`SitesNamespace` reduces the output site axis (`axis == 0` in `gatype.fields`) of a field while
preserving leading batch axes and any input field axes, keeping reductions across coupled sites
distinct from `extensor.sum()` and `extensor.mean()` across independent batch elements. Subspace
selection versus restriction is summarized in `docs/extensor_syntax.md`, and `.sites` reductions in
`docs/fields.md`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from numga.subspace import SubSpace

if TYPE_CHECKING:
    from numga.extensor.extensor import Extensor


class SelectNamespace:
    """Attribute and grade-index lookups project the output axis onto a target `SubSpace` via
    `select_subspace`, zero-padding any target blades absent from the extensor."""

    def __init__(self, extensor: Extensor) -> None:
        self.extensor = extensor
        self.factory = extensor.algebra.subspace

    def _select(self, subspace: SubSpace) -> Extensor:
        return self.extensor.select_subspace(subspace)

    def __getattr__(self, name: str) -> Extensor | Callable:
        constructor = getattr(self.factory, name)
        if isinstance(constructor, SubSpace):
            return self._select(constructor)
        return lambda *args, **kwargs: self._select(constructor(*args, **kwargs))

    def __getitem__(self, grades: int | tuple[int, ...]) -> Extensor:
        grades = grades if isinstance(grades, tuple) else (grades,)
        return self._select(self.factory.from_grades(grades))


class RestrictNamespace(SelectNamespace):
    """Attribute and grade-index lookups intersect the output axis with a target `SubSpace` via
    `restrict_subspace`, keeping only blades already present in the extensor's support."""

    def _select(self, subspace: SubSpace) -> Extensor:
        return self.extensor.restrict_subspace(subspace)


class SitesNamespace:
    """Site reductions collapse the output field axis while leaving leading batch axes and input
    field axes intact."""

    def __init__(self, extensor: Extensor) -> None:
        sites = dict(extensor.gatype.fields)
        if 0 not in sites:
            raise TypeError(f"the output of {extensor.gatype.signature} does not range over sites")
        self.extensor = extensor
        self.count = sites[0]

    def sum(self) -> Extensor:
        """Summing across the output sites removes site axis `0` and widens the output to its structural type."""
        extensor = self.extensor
        kernel = extensor.context.xp.sum(extensor.kernel, axis=extensor.ndim)
        fields = tuple((axis, count) for axis, count in extensor.gatype.fields if axis)
        gatype = extensor.gatype.derive.structural.derive.with_fields(fields)
        return type(extensor)._from_prepared_kernel(extensor.context, gatype, kernel)

    def mean(self) -> Extensor:
        """Averaging across the output sites divides `sum()` by the output site count."""
        return self.sum() / self.count
