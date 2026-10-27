"""Slip in metal crystals: which of a crystal's slip systems a pull makes yield first, and how slipping turns
the crystal under the pull, in the geometric algebra of three-dimensional space.

Space is the crystal's own. A face-centred cubic metal, such as copper or aluminium, deforms by sliding layers
of atoms over one another. A slip moves the crystal on one side of a plane by a step that brings every atom onto a
neighbour's site, so a slip direction is the difference of two neighbouring atoms, a direction, an antivector; and
it slides along one of the densest planes, which holds it. Four mutually neighbouring atoms give all of them: their
six differences are the slip directions, and each plane is spanned by two of them, their join, given a side by the
atoms' winding. Each plane holds three slip directions, twelve slip systems in all.

A pull along an axis puts on each plane the force along the axis, by the plane's pairing with it,
`axis.dual_inverse() * (planes & axis)`, the metric entering once, to turn the axis into a force. The force a slip
direction feels, the resolved shear stress, is that force's pairing with the direction, `forces & direction`: the
Schmid factor times the pull, at most a half. The system with the largest Schmid factor slips first.

A small slip on a system shears the crystal, the map `Direction + amount * direction * (plane & Direction)`:
each point moves along the slip direction by its height above the plane, and slips on several systems add. Each
system slips at a rate rising steeply with its Schmid factor, so the leading system takes almost all of it, and
two systems with equal factors share it alike. Under a pull the specimen stays lined up with the pull, so in
the crystal's own frame the axis is what moves: the part of its move across it turns it, by the bivector
`axis.commutator(move)`, towards the leading slip direction, until a second system catches up and the two slip together.

In the index notation of crystallography the slip planes are the {111} planes, written by their Miller indices,
and a slip direction the Burgers vector $b$ of a <110> step; in that of crystal plasticity the Schmid factor reads
as $m = \\cos\\phi\\,\\cos\\lambda$, the cosines of the axis's angles to the plane normal and to the slip
direction, and the shear as the displacement gradient $\\gamma\\, b \\otimes n$.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                # Scalar+
# A slip plane through the origin.
Plane = ga.gatype.vector()                                 # Vector+
# A direction in the crystal, an atom's position, the pull's axis or a slip direction: an antivector with
# intrinsic orientation.
Direction = ga.gatype.antivector()                         # Antivector-
# A force on a plane: a vector, its pairing with a direction, `force & direction`, the work it does along it.
Force = ga.gatype.vector()                                 # Vector+


# --- math -----------------------------------------------------------------------------
def slip_directions(atoms: Direction) -> Direction:
    """The three slip directions of each slip plane, the one across from each atom: the differences it shares with
    the other three planes, each the difference of the two atoms both planes hold, normalized."""
    count = atoms.shape[-1]
    others = (np.arange(count)[:, None] + np.arange(1, count)) % count         # [planes, directions]
    # The two atoms neither plane lies across from: the ends of the difference the two planes share.
    ends = np.array([[sorted(set(range(count)) - {plane, other}) for other in row]
                     for plane, row in enumerate(others)])                       # [planes, directions, 2]
    return (atoms[ends[..., 1]] - atoms[ends[..., 0]]).normalized()           # [planes, directions] Direction-


def slip_planes(atoms: Direction, directions: Direction) -> Plane:
    """Each slip plane, spanned by two of its slip directions: their join, a patch, given a side by the winding of
    the atoms, the join of their steps from the first, an intrinsic scalar from the data."""
    winding = (atoms[1] - atoms[0]) & (atoms[2] - atoms[0]) & (atoms[3] - atoms[0])   # [] Scalar-
    return ((directions[:, 0] & directions[:, 1]) * winding).normalized()           # [planes] Plane+


def schmid(axis: Direction, planes: Plane, directions: Direction) -> Scalar:
    """The shear stress each pull resolves along each slip system, per unit pull: the force on the slip plane,
    paired with the slip direction."""
    # A unit pull along each axis puts on each plane the force along the axis, by the plane's pairing with the axis;
    # the metric turns the axis into that force.
    forces = axis.dual_inverse()[..., None] * (planes & axis[..., None])      # [..., planes] Force+
    return forces[..., None] & directions                                     # [..., planes, directions] Scalar+


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
