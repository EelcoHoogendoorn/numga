"""Grade transforms (`reverse`, `clifford_conjugate`, `involute`, `scalar_negation`, and
`pseudoscalar_negation`) apply fixed per-grade signs to the output subspace of any extensor.

Each transform dispatches across two staged cases on both dense extensors and fields. When every
grade in the operand's output subspace carries `+1` under the transform, the operand is returned
unchanged without copying or kernel work. Otherwise a diagonal sign operator and output `GAType`
are built once per input `GAType` and composed onto the output axis, leaving input subspaces and
orientations untouched. Grade-sign conventions and the reverse are introduced in `docs/conventions.md`
and `docs/tutorial/geometric-algebra/12-the-reverse.md`.
"""

from __future__ import annotations

from collections.abc import Callable

from numga.extension import ExtensionMethod
from numga.extensor import Extensor
from numga.gatype import GAType, rules


def _register(method: ExtensionMethod, transform: str) -> None:
    rule = rules.GRADE_TRANSFORMS[transform]

    @method.register(lambda gatype: gatype.grade_transform_is_identity(transform))
    def unchanged(extensor: Extensor) -> Extensor:
        return extensor

    def plan(gatype: GAType) -> Callable:
        operator = gatype.algebra.operator._grade_transform(gatype.derive.plain.derive.output, transform)
        axes = operator.axes[:1] + gatype.input_subspaces
        result = rule.result((gatype,), axes)
        if gatype.fields:
            result = result.derive.with_fields(gatype.fields)

        def transformed(extensor: Extensor) -> Extensor:
            applied = operator(extensor)
            return Extensor._from_prepared_kernel(applied.context, result, applied._kernel)

        return transformed

    method.register(lambda gatype: True, staged=True)(plan)
    method.register(lambda gatype: gatype.grade_transform_is_identity(transform), fields=True)(unchanged)
    method.register(lambda gatype: True, fields=True, staged=True)(plan)


_register(Extensor.reverse, "reverse")
_register(Extensor.clifford_conjugate, "clifford_conjugate")
_register(Extensor.involute, "involute")
_register(Extensor.scalar_negation, "scalar_negation")
_register(Extensor.pseudoscalar_negation, "pseudoscalar_negation")
