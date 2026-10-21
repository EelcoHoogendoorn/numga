"""Electrons from polarized cobalt-60, and the mirror image of the experiment.

A coil's current polarizes the nuclei: their spin is a bivector, the plane the current circulates in,
and it reverses with the current. A nucleus decays by emitting an electron, and the rate along a unit
direction is one plus the decay's asymmetry times the spin's dot product with that direction,
`spin.dot(direction)`. The spin is extrinsic and the direction, an antivector, intrinsic, so their dot
product is an intrinsic scalar; for the rate to be an ordinary number, the asymmetry must have a sense
too.

Mirroring the whole apparatus and keeping the decay law gives rates that are not the mirror image of
the original ones: the law holds an intrinsic constant that no mirror of the apparatus reverses.
"""

from __future__ import annotations

from numga import NumpyContext, Orientation
from numga.algebras import VGA3D

ga = VGA3D
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# Directions are antivectors, intrinsic like all geometry.
Direction = ga.gatype.antivector()
Mirror = ga.gatype.vector()
Spin = ga.gatype.bivector()
# A constant of the decay with a sense of its own: measured against a handedness.
Asymmetry = ga.gatype(ga.subspace.scalar(), orientation=Orientation.intrinsic)


# --- math -----------------------------------------------------------------------------
def emission(spin: Spin, directions: Direction, asymmetry: Asymmetry) -> Scalar:
    """The rate of electrons leaving along each unit direction, relative to unpolarized nuclei, from
    nuclei polarized by a unit spin."""
    return 1 + asymmetry * spin.dot(directions)                                  # [...] Scalar
