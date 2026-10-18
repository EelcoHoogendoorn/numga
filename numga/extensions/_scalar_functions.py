"""The two scalar cases of an elementwise function, shared by norm, exp, log and the roots."""

from __future__ import annotations

from typing import Any, Callable

from numga.extension import ExtensionMethod
from numga.extensor import Extensor


def register_scalar_function(method: ExtensionMethod, function: Callable[[Any, Any], Any]) -> None:
    """A scalar stored against -1 is read back against +1 first, so the function acts on the scalar
    and not on its negation; a scalar applies `function(context, kernel)` to its coefficients."""

    @method.register(lambda t: t.is_reoriented_scalar)
    def reoriented_scalar(value: Extensor) -> Extensor:
        return method(value.select_subspace(value.subspace.canonical))

    @method.register(lambda t: t.is_scalar)
    def scalar(value: Extensor) -> Extensor:
        return Extensor._from_prepared_kernel(value.context, value.gatype.derive.structural, function(value.context, value._kernel))
