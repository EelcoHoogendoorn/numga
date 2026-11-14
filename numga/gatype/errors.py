"""Type errors raised from static `GAType` metadata before evaluating any blade coefficients."""

from __future__ import annotations


class GATypeError(TypeError):
    """Incompatible operand `GAType`s or missing dispatch specializations fail from static types
    alone before any blade coefficient is evaluated."""


class UnorientedSubspace(TypeError):
    """A `SubSpace` names blade support and layout without orientation, so operations requiring a
    `GAType` refuse bare subspaces until an orientation is supplied."""

    def __init__(self) -> None:
        super().__init__(
            "a subspace has no orientation: build the type from a named constructor, or as "
            "ga.gatype(subspace, orientation=...)"
        )
