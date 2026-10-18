"""Errors of types: an operation given operands whose types it is not defined for."""

from __future__ import annotations


class GATypeError(TypeError):
    """Operands whose GATypes an operation is not defined for: kinds that must match and do not, or a
    method with no case for these types. The refusal follows from the types alone, before any
    coefficient is computed."""


class UnorientedSubspace(TypeError):
    """A subspace given where a type is needed: a subspace names blades, and a type also says how they
    are oriented."""

    def __init__(self) -> None:
        super().__init__(
            "a subspace has no orientation: build the type from a named constructor, or as "
            "ga.gatype(subspace, orientation=...)"
        )
