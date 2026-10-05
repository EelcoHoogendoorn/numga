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
    """Lay out an unbatched linear map's coefficients, its output blades naming the rows and its
    input blades the columns, numbers formatted to NumPy's print precision."""
    (input_space,) = value.input_subspaces
    kernel = value.kernel.values if hasattr(value.kernel, "values") else np.asarray(value.kernel)
    precision = np.get_printoptions()["precision"]
    cells = [["", *input_space.blade_names]] + [
        [name, *(np.format_float_positional(round(float(coefficient), precision) + 0.0, precision=precision, trim="-") for coefficient in row)]
        for name, row in zip(value.output_subspace.blade_names, kernel)
    ]
    widths = [max(len(row[column]) for row in cells) for column in range(len(cells[0]))]
    return "\n".join(
        "  ".join([row[0].ljust(widths[0])] + [cell.rjust(width) for cell, width in zip(row[1:], widths[1:])])
        for row in cells
    )


def python_code(value, name: str = "apply") -> str:
    """Generate a function taking coefficient sequences and returning a list."""
    terms = expressions(value, lambda slot, index: f"a{slot}[{index}]")
    arguments = ", ".join(f"a{slot}" for slot in range(value.arity))
    return (f"def {name}({arguments}):\n"
            + "    return [\n"
            + "".join(f"        {term},\n" for term in terms)
            + "    ]\n")
