"""Expanded formulas, coefficient tables and standalone Python for output-first kernels."""

from itertools import product


import numpy as np


def expressions(value, component):
    kernel = value.kernel.values if hasattr(value.kernel, "values") else np.asarray(value.kernel)
    result = []
    for output in range(len(value.output_subspace)):
        terms = []
        for indices in product(*(range(len(axis)) for axis in value.input_subspaces)):
            coefficient = kernel[(output,) + indices].item()
            if not coefficient:
                continue
            factors = [component(slot, index) for slot, index in enumerate(indices)]
            magnitude = abs(coefficient)
            if magnitude != 1 or not factors:
                factors.insert(0, str(int(magnitude)) if int(magnitude) == magnitude else str(magnitude))
            terms.append(("-" if coefficient < 0 else "+", " * ".join(factors)))
        expression = " ".join(f"{sign} {term}" for sign, term in terms)
        result.append(expression.removeprefix("+ ") or "0")
    return result


def formula(value) -> str:
    """Name coefficients by their signed basis blades, at any arity."""
    def blade(space, index):
        return ("-" if space.signs[index] < 0 else "") + value.algebra.blade_name(space.masks[index])

    terms = expressions(value, lambda slot, index: f"a{slot}[{blade(value.input_subspaces[slot], index)}]")
    return "\n".join(f"out[{blade(value.output_subspace, index)}] = {term}"
                     for index, term in enumerate(terms))


def coefficient_table(value) -> str:
    """Lay out blade-labelled coefficients of an unbatched multivector or linear map.

    A multivector has one row beneath its blade labels. A map labels its rows with
    output blades and its columns with input blades.
    Numbers follow NumPy's print precision.
    """
    if value.ndim:
        raise ValueError("coefficient_table requires an unbatched extensor")
    kernel = value.kernel.values if hasattr(value.kernel, "values") else np.asarray(value.kernel)
    if value.arity:
        (input_space,) = value.input_subspaces
        column_names = input_space.blade_names
        row_names = value.output_subspace.blade_names
        rows = kernel
    else:
        column_names = value.output_subspace.blade_names
        row_names = ("",)
        rows = kernel[None, :]
    precision = np.get_printoptions()["precision"]
    cells = [["", *column_names]] + [
        [name, *(np.format_float_positional(round(float(coefficient), precision) + 0.0, precision=precision, trim="-") for coefficient in row)]
        for name, row in zip(row_names, rows)
    ]
    widths = [max(len(row[column]) for row in cells) for column in range(len(column_names) + 1)]
    return "\n".join(
        "  ".join([row[0].ljust(widths[0])] + [cell.rjust(width) for cell, width in zip(row[1:], widths[1:])])
        for row in cells
    ).rstrip()


def python_code(value, name: str = "apply") -> str:
    """Generate a function taking coefficient sequences and returning a list."""
    terms = expressions(value, lambda slot, index: f"a{slot}[{index}]")
    arguments = ", ".join(f"a{slot}" for slot in range(value.arity))
    return (f"def {name}({arguments}):\n"
            + "    return [\n"
            + "".join(f"        {term},\n" for term in terms)
            + "    ]\n")
