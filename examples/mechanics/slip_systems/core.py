"""Slip in metal crystals: which of a crystal's slip systems a pull makes yield first, and how slipping turns
the crystal under the pull, in the geometric algebra of three-dimensional space.

Space is the crystal's own. A face-centred cubic metal, such as copper or aluminium, deforms by sliding layers
of atoms over one another: along the four densest planes, the {111} planes, in the densest directions within
each, the <110> directions, which are the lines where two of those planes meet. Each plane meets the other
three in three slip directions, twelve slip systems in all.

A slip plane is a covector, a vector, its components the plane's Miller indices; a direction in the crystal is an
antivector. Four neighbouring atoms make the Thompson tetrahedron, whose four faces lie in the slip planes and whose
six edges, the differences of its atoms, are the slip directions, each shared by two faces. A pull along an axis is
the stress map `axis.dual_inverse() * (Plane & axis)`, the force on a plane along the axis by the plane's pairing
with it, the metric entering once, to turn the axis into a force. The force a slip direction feels, the resolved
shear stress, is the force's pairing with the direction, `tension(axis)(plane) & direction`: the Schmid factor
times the pull, at most a half. The system with the largest Schmid factor slips first.

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
Scalar = ga.gatype.scalar()                                # Scalar+
# A slip plane through the origin, given by its Miller indices, the components of a covector.
Plane = ga.gatype.vector()                                 # Vector+
# A direction in the crystal, an atom's position, the pull's axis or a slip direction: an antivector with
# intrinsic orientation.
Direction = ga.gatype.antivector()                         # Antivector-
# A force on a plane is a covector, read by its pairing with a direction; a pull's stress maps a plane to the force
# on it.
Force = ga.gatype.vector()                                 # Vector+
Stress = ga.gatype((Force, Plane))


# --- math -----------------------------------------------------------------------------
def slip_directions(atoms: Direction) -> Direction:
    """The three slip directions of each slip plane, the face of the Thompson tetrahedron opposite each atom: the
    edges it shares with the other three faces, each the difference of the two atoms both faces hold, normalized."""
    count = atoms.shape[-1]
    others = (np.arange(count)[:, None] + np.arange(1, count)) % count         # [planes, directions]
    # The two atoms neither face lies opposite: the ends of the edge the two faces share.
    ends = np.array([[sorted(set(range(count)) - {face, other}) for other in row]
                     for face, row in enumerate(others)])                       # [planes, directions, 2]
    return (atoms[ends[..., 1]] - atoms[ends[..., 0]]).normalized()           # [planes, directions] Direction-


def tension(axis: Direction) -> Stress:
    """A unit pull along each axis: the force on a plane is the force along the axis, by the plane's pairing with
    the axis; the metric turns the axis into that force."""
    return axis.dual_inverse() * (Plane & axis)                               # [...] Force+ <- Plane+


def schmid(axis: Direction, planes: Plane, directions: Direction) -> Scalar:
    """The shear stress each pull resolves along each slip system, per unit pull: the traction on the slip
    plane, along the slip direction."""
    return tension(axis[..., None, None])(planes[:, None]) & directions       # [..., planes, directions] Scalar+


def slipped(axis: Direction, planes: Plane, directions: Direction, amounts: Scalar) -> Direction:
    """Each axis after a small slip on every system by its amount `[..., planes, directions]`: each slip moves
    the axis along its direction by its height above its plane, the moves add, and the part of their sum across
    the axis turns it, in the plane of the axis and the move."""
    # Each slip direction enters twice, through its Schmid factor and here, so the sense the order of its atoms
    # gave it drops out of the move.
    moves = amounts * directions * (planes[:, None] & axis[..., None, None])  # [..., planes, directions] Direction-
    move = moves.sum(axis=-1).sum(axis=-1)                                    # [...] Direction-
    # The commutator of two directions is the bivector of the plane they span.
    return (axis.commutator(move) / 2).exp() >> axis                         # [...] Direction-
