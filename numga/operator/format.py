"""Any unbatched `Extensor` (whether exact in `SymbolicKernel` or numeric in a backend array)
stores its coordinates output-first along axis `0` and input axes `1..arity`. Three views render
those coordinates without altering layout or orientation: `formula` expands each output blade as a
symbolic polynomial in blade-labelled inputs `a0[x]`, `coefficient_table` aligns blade-labelled
rows and columns at NumPy's active print precision, and `python_code` emits a standalone Python
function over plain indexable sequences.

Signed basis layouts (`SubSpace.signs`) appear explicitly in `formula` and `coefficient_table`
headers so negated basis blades (`-zx` versus `xz`) read directly from the labels, whereas
`python_code` indexes raw positional slots `a0[0]` matching the extensor's underlying kernel array.
Formula and Python code views of exact tables are illustrated in `docs/internals.md` (section 2).
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import product
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numga.extensor import Extensor
    from numga.subspace import SubSpace


def expressions(extensor: Extensor, component: Callable) -> list[str]:
    """Expanding every non-zero kernel entry produces one sum-of-products string per output blade,
    formatting input factors via `component(axis, index)`."""
    kernel = extensor.kernel.values if hasattr(extensor.kernel, "values") else np.asarray(extensor.kernel)
    result = []
    for output in range(len(extensor.output_subspace)):
        terms = []
        for indices in product(*(range(len(axis)) for axis in extensor.input_subspaces)):
            coefficient = kernel[(output,) + indices].item()
            if not coefficient:
                continue
            factors = [component(axis, index) for axis, index in enumerate(indices)]
            magnitude = abs(coefficient)
            if magnitude != 1 or not factors:
                factors.insert(0, str(int(magnitude)) if int(magnitude) == magnitude else str(magnitude))
            terms.append(("-" if coefficient < 0 else "+", " * ".join(factors)))
        expression = " ".join(f"{sign} {term}" for sign, term in terms)
        result.append(expression.removeprefix("+ ") or "0")
    return result


def formula(extensor: Extensor) -> str:
    """Formatting each output blade as a signed polynomial labels input coordinates by blade name
    (`out[x] = a0[1] * a1[x] + ...`)."""
    def blade(space: SubSpace, index: int) -> str:
        return ("-" if space.signs[index] < 0 else "") + extensor.algebra.blade_name(space.masks[index])

    terms = expressions(extensor, lambda axis, index: f"a{axis}[{blade(extensor.input_subspaces[axis], index)}]")
    return "\n".join(f"out[{blade(extensor.output_subspace, index)}] = {term}"
                     for index, term in enumerate(terms))


def number(coefficient: complex, precision: int) -> str:
    """Formatting a real or complex scalar to `precision` decimal places writes `i` for the
    imaginary unit and omits zero parts."""
    def real(part: float) -> str:
        return np.format_float_positional(round(part, precision) + 0.0, precision=precision, trim="-")

    scalar = complex(coefficient)
    imaginary = round(scalar.imag, precision) + 0.0
    if imaginary == 0:
        return real(scalar.real)
    unit = "i" if abs(imaginary) == 1 else real(abs(imaginary)) + "i"
    if round(scalar.real, precision) == 0:
        return ("-" if imaginary < 0 else "") + unit
    return real(scalar.real) + ("-" if imaginary < 0 else "+") + unit


def coefficient_table(extensor: Extensor) -> str:
    """Formatting an unbatched multivector writes a single row beneath blade headers, while a
    linear map labels rows by output blades and columns by input blades at NumPy's print precision."""
    if extensor.ndim:
        raise ValueError("coefficient_table requires an unbatched extensor")
    kernel = extensor.kernel.values if hasattr(extensor.kernel, "values") else np.asarray(extensor.kernel)
    if extensor.arity:
        (input_space,) = extensor.input_subspaces
        column_names = input_space.blade_names
        row_names = extensor.output_subspace.blade_names
        rows = kernel
    else:
        column_names = extensor.output_subspace.blade_names
        row_names = ("",)
        rows = kernel[None, :]
    precision = np.get_printoptions()["precision"]
    cells = [["", *column_names]] + [
        [name, *(number(coefficient, precision) for coefficient in row)]
        for name, row in zip(row_names, rows)
    ]
    widths = [max(len(row[column]) for row in cells) for column in range(len(column_names) + 1)]
    return "\n".join(
        "  ".join([row[0].ljust(widths[0])] + [cell.rjust(width) for cell, width in zip(row[1:], widths[1:])])
        for row in cells
    ).rstrip()


def python_code(extensor: Extensor, function_name: str = "apply") -> str:
    """Standalone Python source defines `function_name(a0, ...)` returning the output blade
    coefficients as a list of polynomial expressions."""
    terms = expressions(extensor, lambda axis, index: f"a{axis}[{index}]")
    arguments = ", ".join(f"a{axis}" for axis in range(extensor.arity))
    return (f"def {function_name}({arguments}):\n"
            + "    return [\n"
            + "".join(f"        {term},\n" for term in terms)
            + "    ]\n")
