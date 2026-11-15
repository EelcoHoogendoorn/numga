"""Elementwise scalar extension methods share canonical-sign normalization across `norm`, `exp`,
`log`, and square roots."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from numga.extension import ExtensionMethod
from numga.extensor import Extensor


def register_scalar_function(method: ExtensionMethod, function: Callable) -> None:
    """Scalars stored against the `-1` basis element are converted to canonical `+1` coefficients
    first so `function` acts on the scalar value itself rather than on its negated storage."""

    @method.register(lambda t: t.is_reoriented_scalar)
    def reoriented_scalar(value: Extensor) -> Extensor:
        return method(value.select_subspace(value.subspace.canonical))

    @method.register(lambda t: t.is_scalar)
    def scalar(value: Extensor) -> Extensor:
        return Extensor._from_prepared_kernel(value.context, value.gatype.derive.structural, function(value.context, value._kernel))
