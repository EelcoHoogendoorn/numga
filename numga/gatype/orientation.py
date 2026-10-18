"""Intrinsic and extrinsic orientation of the elements of a type.

Every flat can be oriented within itself (intrinsic orientation: a point by a sign, a line by an arrow
along it, a plane by a circulation in it) or by what crosses it (extrinsic orientation: a plane by a side,
a line by a circulation around it, a point by a handedness around it). The algebra's vectors, read as
planes, are extrinsic, and so is everything built from them by products and meets; the antivectors,
points, are intrinsic, and so is everything built from them by joins. A reflection reverses the
orientation of space, so the two kinds transform with opposite signs under it.

Orientations combine as signs multiply, extrinsic as plus and intrinsic as minus: a product of equal
orientations is extrinsic, of different ones intrinsic. The regressive product passes through the volume
form once and gives the opposite.

The names, and the reading of a projective dual as the complementary orientation, follow Leo Dorst,
"Projective duality encodes complementary orientations in geometric algebras", Mathematical Methods in
the Applied Sciences 47 (2024). That paper suggests labelling intrinsic elements with a complementarity
bit, after Gunn; a GAType's orientation is that bit. In exterior calculus (Schouten, Tonti, Bossavit)
the two are called inner and outer orientation."""

from __future__ import annotations

from enum import Enum


class Orientation(Enum):
    intrinsic = "intrinsic"
    extrinsic = "extrinsic"

    def __mul__(self, other: Orientation) -> Orientation:
        """The orientation of a product: extrinsic for equal orientations, intrinsic for different ones."""
        if not isinstance(other, Orientation):
            return NotImplemented
        return Orientation.extrinsic if self is other else Orientation.intrinsic

    @property
    def flipped(self) -> Orientation:
        return Orientation.intrinsic if self is Orientation.extrinsic else Orientation.extrinsic

    def __repr__(self) -> str:
        return f"Orientation.{self.value}"


class UnorientedSubspace(TypeError):
    """A subspace given where a type is needed: a subspace names blades, and a type also says how they
    are oriented."""

    def __init__(self) -> None:
        super().__init__(
            "a subspace has no orientation: build the type from a named constructor, or as "
            "ga.gatype(subspace, orientation=...)"
        )
