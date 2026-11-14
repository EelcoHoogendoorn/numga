"""Every algebraic operation on extensors owns an `OperationRule` determining the `GAType` of its
result from its operand `GAType`s and output-first result `SubSpace`s. Because `SubSpace`s carry no
orientation of their own, orientation is never inferred by default: an operation's rule computes the
output-first `Orientation` tuple explicitly from operand orientations, and attaches any trait laws
preserved or introduced by the operation via `propagation`.

Binary products multiply operand orientations like signs (`multiplied`), while the regressive
product passes through the pseudoscalar once and flips that product orientation (`through_volume`).
Versor sandwich actions preserve the passenger's orientation (`carries_passenger`), transposes
reverse axis order together with orientations (`reversed_axes`), unary scalings and grade transforms
keep operand orientations (`kept`), and coefficient-wise sums or stacks require identical
orientations across all operands (`equal`). Resolved result `GAType`s are cached per rule, operand
`GAType` tuple, and result `SubSpace` tuple. The orientation rules and refusals enforced here follow
`docs/orientation.md` and `docs/conventions.md`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from .errors import GATypeError
from .orientation import Orientation
from .propagation import (
    blade_laws,
    coefficient_orthogonality_laws,
    conjugation_laws,
    geometric_product_laws,
    negation_laws,
    no_laws,
    shared_laws,
    transform_laws,
    transpose_laws,
    unless_identity,
)
from .traits import CliffordConjugateProduct, ReverseProduct, TraitSet

if TYPE_CHECKING:
    from numga.subspace import SubSpace

    from .gatype import GAType

Orientations = Callable
Laws = Callable


@dataclass(frozen=True, eq=False)
class OperationRule:
    """One operation's static type rule pairs an axis-orientation rule with a trait-propagation law,
    comparing and hashing by object identity."""

    name: str
    orientations: Orientations
    laws: Laws = no_laws

    def result(self, operands: tuple[GAType, ...], axes: tuple[SubSpace, ...]) -> GAType:
        """Applying this rule's orientation and trait laws to output-first axes produces the cached
        result `GAType`."""
        return _result(self, operands, axes)


@lru_cache(maxsize=None)
def _result(rule: OperationRule, operands: tuple[GAType, ...], axes: tuple[SubSpace, ...]) -> GAType:
    algebra = (operands[0] if operands else axes[0]).algebra
    return algebra.gatype.oriented(axes, rule.orientations(operands), rule.laws(operands, axes))


# --- orientation rules ----------------------------------------------------------------
def multiplied(operands: tuple[GAType, ...]) -> tuple[Orientation, ...]:
    """Binary products give the output the product of the two operand orientations."""
    left, right = operands
    return (left.orientation * right.orientation, left.orientation, right.orientation)


def through_volume(operands: tuple[GAType, ...]) -> tuple[Orientation, ...]:
    """Passing once through the pseudoscalar flips the binary product's output orientation."""
    left, right = operands
    return ((left.orientation * right.orientation).flipped, left.orientation, right.orientation)


def carries_passenger(operands: tuple[GAType, ...]) -> tuple[Orientation, ...]:
    """Sandwich actions preserve the passenger's output orientation regardless of versor orientation."""
    versor, passenger, _ = operands
    return (passenger.orientation, versor.orientation, passenger.orientation, versor.orientation)


def reversed_axes(operands: tuple[GAType, ...]) -> tuple[Orientation, ...]:
    """Transposing a map reverses its axis order and per-axis orientations together."""
    return tuple(reversed(operands[0].orientations))


def kept(operands: tuple[GAType, ...]) -> tuple[Orientation, ...]:
    """Unary coefficient transforms preserve every axis orientation of their operand."""
    return operands[0].orientations


def equal(operands: tuple[GAType, ...]) -> tuple[Orientation, ...]:
    """Coefficient-wise sums and stacks require matching orientations on every axis and retain them."""
    first = operands[0]
    if any(gatype.orientations != first.orientations for gatype in operands[1:]):
        signatures = " and ".join(gatype.signature for gatype in operands)
        raise GATypeError(f"cannot combine {signatures}: their orientations differ")
    return first.orientations


# --- rules -----------------------------------------------------------------------------
GEOMETRIC = OperationRule("geometric product", multiplied, geometric_product_laws)
MULTIPLIED = OperationRule("product", multiplied)
WEDGE = OperationRule("wedge", multiplied, blade_laws)
REGRESSIVE = OperationRule("regressive product", through_volume, blade_laws)
TRANSFORM = OperationRule("transform", carries_passenger, transform_laws)
TRANSPOSE = OperationRule("transpose", reversed_axes, transpose_laws)
ADD = OperationRule("add", equal)
STACK = OperationRule("stack", equal, shared_laws)
NEGATE = OperationRule("negate", kept, negation_laws)
SCALE = OperationRule("scale", kept)
SQUEEZE = OperationRule("squeeze", kept)
REVERSE = OperationRule("reverse", kept, unless_identity("reverse", conjugation_laws(ReverseProduct)))
CLIFFORD_CONJUGATE = OperationRule(
    "clifford conjugate", kept, unless_identity("clifford_conjugate", conjugation_laws(CliffordConjugateProduct)),
)
INVOLUTE = OperationRule("involute", kept, unless_identity("involute", negation_laws))
SCALAR_NEGATION = OperationRule("scalar negation", kept, unless_identity("scalar_negation", coefficient_orthogonality_laws))
PSEUDOSCALAR_NEGATION = OperationRule(
    "pseudoscalar negation", kept, unless_identity("pseudoscalar_negation", coefficient_orthogonality_laws),
)
# Grade transforms keyed by the names used on the operator factory.
GRADE_TRANSFORMS = {
    "reverse": REVERSE,
    "clifford_conjugate": CLIFFORD_CONJUGATE,
    "involute": INVOLUTE,
    "scalar_negation": SCALAR_NEGATION,
    "pseudoscalar_negation": PSEUDOSCALAR_NEGATION,
}

