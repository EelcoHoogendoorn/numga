"""Type rules of operations: how the type of an operation's result follows from its operands' types.

Each operation has one rule object. Its orientation rule gives the orientation of every axis of the
result, output first, from the operands' orientations; orientation is never assumed, so an operation
without a rule has no result type. Its laws are the traits the result inherits, from `propagation`.
Results are cached per rule, operand types and result axes.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Callable

from .errors import GATypeError
from .orientation import Orientation
from .propagation import (
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

Orientations = Callable[[tuple["GAType", ...]], tuple[Orientation, ...]]
Laws = Callable[[tuple["GAType", ...], tuple["SubSpace", ...]], TraitSet]


@dataclass(frozen=True, eq=False)
class OperationRule:
    """The type rule of one operation: the orientations of its result's axes, and the laws its result
    inherits. Rules compare and hash by identity."""

    name: str
    orientations: Orientations
    laws: Laws = no_laws

    def result(self, operands: tuple["GAType", ...], axes: tuple["SubSpace", ...]) -> "GAType":
        """The result type of this operation on operands of these types, with these result axes."""
        return _result(self, operands, axes)


@lru_cache(maxsize=None)
def _result(rule: OperationRule, operands: tuple["GAType", ...], axes: tuple["SubSpace", ...]) -> "GAType":
    algebra = (operands[0] if operands else axes[0]).algebra
    return algebra.gatype.oriented(axes, rule.orientations(operands), rule.laws(operands, axes))


# --- orientation rules ----------------------------------------------------------------
def multiplied(operands: tuple["GAType", ...]) -> tuple[Orientation, ...]:
    """A product of two operands: the output's orientation is the product of theirs, as signs multiply."""
    left, right = operands
    return (left.orientation * right.orientation, left.orientation, right.orientation)


def through_volume(operands: tuple["GAType", ...]) -> tuple[Orientation, ...]:
    """The regressive product passes through the volume form once: the product's orientation, flipped."""
    left, right = operands
    return ((left.orientation * right.orientation).flipped, left.orientation, right.orientation)


def carries_passenger(operands: tuple["GAType", ...]) -> tuple[Orientation, ...]:
    """A versor's action keeps the passenger's orientation."""
    versor, passenger, _ = operands
    return (passenger.orientation, versor.orientation, passenger.orientation, versor.orientation)


def reversed_axes(operands: tuple["GAType", ...]) -> tuple[Orientation, ...]:
    """A transpose reverses the axes, and their orientations with them."""
    return tuple(reversed(operands[0].orientations))


def kept(operands: tuple["GAType", ...]) -> tuple[Orientation, ...]:
    """Every axis keeps its operand's orientation."""
    return operands[0].orientations


def equal(operands: tuple["GAType", ...]) -> tuple[Orientation, ...]:
    """Values combined coefficient by coefficient, as in a sum or a stack, need equal orientations on
    every axis, and keep them."""
    first = operands[0]
    if any(gatype.orientations != first.orientations for gatype in operands[1:]):
        signatures = " and ".join(gatype.signature for gatype in operands)
        raise GATypeError(f"cannot combine {signatures}: their orientations differ")
    return first.orientations


# --- rules -----------------------------------------------------------------------------
GEOMETRIC = OperationRule("geometric product", multiplied, geometric_product_laws)
MULTIPLIED = OperationRule("product", multiplied)
REGRESSIVE = OperationRule("regressive product", through_volume)
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
# The grade transforms by the name the operator factory builds them under.
GRADE_TRANSFORMS = {
    "reverse": REVERSE,
    "clifford_conjugate": CLIFFORD_CONJUGATE,
    "involute": INVOLUTE,
    "scalar_negation": SCALAR_NEGATION,
    "pseudoscalar_negation": PSEUDOSCALAR_NEGATION,
}

