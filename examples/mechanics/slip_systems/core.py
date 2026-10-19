"""Slip in metal crystals: which of a crystal's slip systems a pull makes yield first, and how slipping turns
the crystal under the pull, in the geometric algebra of three-dimensional space.

Space is the crystal's own. A face-centred cubic metal, such as copper or aluminium, deforms by sliding layers
of atoms over one another: along the four densest planes, the {111} planes, in the densest directions within
each, the <110> directions, which are the lines where two of those planes meet. Each plane meets the other
three in three slip directions, twelve slip systems in all.

A slip plane is a vector, the pull's axis a direction, an antivector, and a slip direction the line where two
slip planes meet, `plane ^ other`, a bivector whose sense the order of the planes sets. A pull along an axis is the stress map `axis * (Plane & axis)`,
the traction on a plane pointing along the axis by the plane's pairing with it. The part of that traction along
a slip direction, the resolved shear stress, is `~direction | stress(plane)`: the Schmid factor
`(~direction | axis) * (plane & axis)` times the pull, at most a half. The system with the largest Schmid factor
slips first.

A small slip on a system shears the crystal, the map `Direction + amount * direction * (plane & Direction)`:
each point moves along the slip direction by its height above the plane, and slips on several systems add. Each
system slips at a rate rising steeply with its Schmid factor, so the leading system takes almost all of it, and
two systems with equal factors share it alike. Under a pull the specimen stays lined up with the pull, so in
the crystal's own frame the axis is what moves: the part of its move across it turns it, by the bivector
`axis.commutator(move)`, towards the leading slip direction, until a second system catches up and the two slip together.

In the index notation of crystal plasticity the Schmid factor reads as $m = \\cos\\phi\\,\\cos\\lambda$, the
cosines of the axis's angles to the plane normal and to the slip direction, and the shear as the displacement
gradient $\\gamma\\, b \\otimes n$.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# A slip plane through the origin.
Plane = ga.gatype.vector()
# A direction in the crystal, the pull's axis: an antivector with intrinsic orientation.
Direction = ga.gatype.antivector()
# A slip direction, the line where two slip planes meet: a bivector, its sense set by the order of the planes.
Line = ga.gatype.bivector()
# Traction on a plane.
Stress = ga.gatype((Direction, Plane))                     # Direction <- Plane


# --- math -----------------------------------------------------------------------------
def slip_directions(planes: Plane) -> Line:
    """The three slip directions of each slip plane: the lines where it meets each of the others, each
    normalized."""
    count = planes.shape[-1]
    others = (np.arange(count)[:, None] + np.arange(1, count)) % count         # [planes, directions]
    return (planes[:, None] ^ planes[others]).normalized()                    # [planes, directions] Line


def tension(axis: Direction) -> Stress:
    """A unit pull along each axis: the traction on a plane is the axis, by the plane's pairing with it."""
    return axis * (Plane & axis)                                              # [...] Direction <- Plane


def schmid(axis: Direction, planes: Plane, directions: Line) -> Scalar:
    """The shear stress each pull resolves along each slip system, per unit pull: the traction on the slip
    plane, along the slip direction."""
    # A bivector squares to minus its length squared, so the metric reads one along another by its reverse.
    return ~directions | tension(axis[..., None, None])(planes[:, None])      # [..., planes, directions] Scalar


def slipped(axis: Direction, planes: Plane, directions: Line, amounts: Scalar) -> Direction:
    """Each axis after a small slip on every system by its amount `[..., planes, directions]`: each slip moves
    the axis along its direction by its height above its plane, the moves add, and the part of their sum across
    the axis turns it, in the plane of the axis and the move."""
    # Each slip line enters twice, through its Schmid factor and here, so the sense the order of its planes
    # gave it drops out of the move.
    moves = amounts * directions * (planes[:, None] & axis[..., None, None])  # [..., planes, directions] Direction
    move = moves.sum(axis=-1).sum(axis=-1)                                    # [...] Direction
    # The commutator of two directions is the bivector of the plane they span.
    return (axis.commutator(move) / 2).exp() >> axis                         # [...] Direction
