"""The grade transforms: the reverse, the Clifford conjugate, the grade involution, and the negations
of the scalar and the pseudoscalar part, each a sign per grade on the output.

Each is the same two cases. Where the transform is the identity on every grade of the operand's
output, the result is the operand itself. Otherwise the transform is a diagonal map on the output,
built with its result type once per operand type and applied to every value of that type; a map's
output is transformed by composition.
"""

from __future__ import annotations

from typing import Callable

from numga.extension import ExtensionMethod
from numga.extensor import Extensor
from numga.gatype import GAType, rules


def _register(method: ExtensionMethod, transform: str) -> None:
    rule = rules.GRADE_TRANSFORMS[transform]

    @method.register(lambda gatype: gatype.grade_transform_is_identity(transform))
    def unchanged(value: Extensor) -> Extensor:
        return value

    def plan(gatype: GAType) -> Callable[[Extensor], Extensor]:
        operator = gatype.algebra.operator._grade_transform(gatype.derive.plain.derive.output, transform)
        axes = operator.axes[:1] + gatype.input_subspaces
        result = rule.result((gatype,), axes)
        if gatype.fields:
            result = result.derive.with_fields(gatype.fields)

        def transformed(value: Extensor) -> Extensor:
            applied = operator(value)
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
