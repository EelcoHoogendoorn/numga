"""Expanded formulas and standalone Python for exact output-first kernels."""

from itertools import product


def expressions(value, component):
    kernel = value.kernel.to_object_array()
    result = []
    for output in range(len(value.output_subspace)):
        terms = []
        for indices in product(*(range(len(axis)) for axis in value.input_subspaces)):
            coefficient = kernel[(output,) + indices]
            if not coefficient:
                continue
            factors = [component(slot, index) for slot, index in enumerate(indices)]
            magnitude = abs(coefficient)
            if magnitude != 1 or not factors:
                factors.insert(0, str(magnitude.numerator) if magnitude.denominator == 1
                               else f"Fraction({magnitude.numerator}, {magnitude.denominator})")
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


def python_code(value, name: str = "apply") -> str:
    """Generate a function taking coefficient sequences and returning a list."""
    terms = expressions(value, lambda slot, index: f"a{slot}[{index}]")
    arguments = ", ".join(f"a{slot}" for slot in range(value.arity))
    return ("from fractions import Fraction\n\n"
            + f"def {name}({arguments}):\n"
            + "    return [\n"
            + "".join(f"        {term},\n" for term in terms)
            + "    ]\n")
