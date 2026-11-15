"""Every flat in geometric algebra admits two geometric orientations: intrinsic orientation within
the flat (a point by a sign, a line by an arrow along it, a plane by a circulation inside it) and
extrinsic orientation across the flat (a plane by a normal side, a line by a circulation around it,
a point by a surrounding handedness). Basis vectors represent planes and start extrinsic, as do all
elements built from them via geometric, exterior, and interior products; antivectors represent
points and start intrinsic, as do elements joined from them via the regressive product. Because a
reflection reverses ambient space orientation, intrinsic and extrinsic elements transform with
opposite signs under odd versors.

Orientations compose under multiplication like signs, with `extrinsic` acting as `+1` and
`intrinsic` as `-1`: multiplying equal orientations gives `extrinsic`, while multiplying opposite
orientations gives `intrinsic`. Passing once through the pseudoscalar flips orientation, so `dual`,
`dual_inverse`, and the regressive product `&` invert the product orientation.

The terminology and projective-duality correspondence follow Leo Dorst, "Projective duality encodes
complementary orientations in geometric algebras", Mathematical Methods in the Applied Sciences 47
(2024), where intrinsic elements carry Gunn's complementarity bit. In exterior calculus (Schouten,
Tonti, Bossavit), `intrinsic` and `extrinsic` correspond to inner and outer orientation. The
geometric motivation and propagation rules in `numga` are developed in `docs/why_orientation.md`
and `docs/orientation.md`.
"""

from __future__ import annotations

from enum import Enum
from types import NotImplementedType


class Orientation(Enum):
    """Intrinsic and extrinsic orientations form a two-element group under multiplication."""

    intrinsic = "intrinsic"
    extrinsic = "extrinsic"

    def __mul__(self, other: object) -> Orientation | NotImplementedType:
        """Equal orientations multiply to `extrinsic` and opposite orientations to `intrinsic`."""
        if not isinstance(other, Orientation):
            return NotImplemented
        return Orientation.extrinsic if self is other else Orientation.intrinsic

    @property
    def flipped(self) -> Orientation:
        """Complementary duality exchanges `extrinsic` and `intrinsic`."""
        return Orientation.intrinsic if self is Orientation.extrinsic else Orientation.extrinsic

    def __repr__(self) -> str:
        return f"Orientation.{self.value}"

