"""Electrons from polarized cobalt-60, and the mirror image of the experiment.

A coil's current polarizes the nuclei. Their spin is an angular momentum, intrinsic like a momentum: the
patch the current circulates around, the join of two directions in the coil's plane, which reverses with
the current. A nucleus decays by emitting an electron, and the rate along a unit direction is one plus
the decay's asymmetry times the spin's pairing with that direction, `spin & direction`. Both are
intrinsic, so their pairing is an intrinsic scalar; for the rate to be an ordinary number, the asymmetry
must have a sense too.

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
# A spin is an angular momentum: the patch its current circulates around, intrinsic.
Spin = ga.gatype.vector().derive.flipped_orientation
# A constant of the decay with a sense of its own: measured against a handedness.
Asymmetry = ga.gatype(ga.subspace.scalar(), orientation=Orientation.intrinsic)


# --- math -----------------------------------------------------------------------------
def emission(spin: Spin, directions: Direction, asymmetry: Asymmetry) -> Scalar:
    """The rate of electrons leaving along each unit direction, relative to unpolarized nuclei, from
    nuclei polarized by a unit spin."""
    return 1 + asymmetry * (spin & directions)                                   # [...] Scalar
