"""Applied tension selects active slip systems and rotates the tensile axis in a face-centred cubic crystal.

Four mutually neighbouring atoms form a regular tetrahedron whose six edge differences give the unit slip
directions (`Direction-`) and whose four oriented faces give the unit slip planes (`Plane+`), forming twelve
slip systems in three-dimensional Euclidean geometric algebra. Unit tension along `axis` places a traction
covector `axis.dual_inverse() * (planes & axis)` on each plane, and pairing that traction with each slip
direction `tractions & directions` gives the Schmid factor (`Scalar+`).

Each slip increment shears crystal directions by `Direction + amount * direction * (plane & Direction)` at a
steep power-law rate in the Schmid factor, and the resulting displacement `move` rotates the tensile axis in
the crystal frame through `(axis.commutator(move) / 2).exp() >> axis` until two conjugate systems share the
slip along a symmetry boundary.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                # Scalar+
Plane = ga.gatype.vector()                                 # Vector+
Direction = ga.gatype.antivector()                         # Antivector-


# --- math -----------------------------------------------------------------------------
def slip_directions(atoms: Direction) -> Direction:
    """Nearest-neighbour atom differences along shared tetrahedral edges give the three unit slip directions (`[4, 3] Direction-`) in each slip plane."""
    others = (np.arange(4)[:, None] + np.arange(1, 4)) % 4                    # [4, 3]
    # Each plane shares a slip direction along the edge joining the two atoms that neither plane lies opposite.
    ends = np.array([[sorted(set(range(4)) - {plane, other}) for other in row]
                     for plane, row in enumerate(others)])                    # [4, 3, 2]
    return (atoms[ends[..., 1]] - atoms[ends[..., 0]]).normalized()           # [4, 3] Direction-


def slip_planes(atoms: Direction, directions: Direction) -> Plane:
    """Joining two slip directions in each tetrahedral face and multiplying by the atom winding orients the four unit slip planes (`[4] Plane+`)."""
    winding = (atoms[1] - atoms[0]) & (atoms[2] - atoms[0]) & (atoms[3] - atoms[0])   # [] Scalar-
    return ((directions[:, 0] & directions[:, 1]) * winding).normalized()             # [4] Plane+


def schmid_factors(axis: Direction, planes: Plane, directions: Direction) -> Scalar:
    """Pairing the plane tractions of unit tension along `axis` with each slip direction gives the Schmid factors (`[..., 4, 3] Scalar+`)."""
    # Unit tension along each axis places a work-conjugate traction covector on each slip plane.
    tractions = axis.dual_inverse()[..., None] * (planes & axis[..., None])   # [..., 4] Plane+
    return tractions[..., None] & directions                                  # [..., 4, 3] Scalar+


def disk_directions(n_samples: int) -> Direction:
    """Stereographic reflection across `[001]` lifts a square grid on the equatorial plane to pull directions (`[n_samples, n_samples] Direction-`)."""
    across = np.linspace(-1, 1, n_samples)                                    # [n_samples]
    offsets = mv.x * across + mv.y * across[:, None]                          # [n_samples, n_samples] Plane+
    mirrors = (mv.z + offsets).normalized()                                   # [n_samples, n_samples] Plane+
    return mirrors >> mv.z.dual()                                             # [n_samples, n_samples] Direction-


def slip_amounts(factors: Scalar, step_slip: float, steepness: int) -> Scalar:
    """A power-law flow rule scales each system's slip increment (`[..., 4, 3] Scalar+`) by its Schmid factor relative to the leading system."""
    magnitudes = factors.abs()                                                # [..., 4, 3] Scalar+
    leading = magnitudes.max(axis=(-2, -1), keepdims=True)                    # [..., 1, 1] Scalar+
    return step_slip * factors * (magnitudes ** (steepness - 1) / leading ** steepness)   # [..., 4, 3] Scalar+


def turned_axes(axis: Direction, planes: Plane, directions: Direction, amounts: Scalar) -> Direction:
    """Simultaneous slips `amounts` shear `axis` and rotate each tensile axis (`[...] Direction-`) in the crystal frame."""
    # Each slip direction enters both `schmid_factors` and `moves`, so arbitrary sign choices on edge directions cancel out.
    moves = amounts * directions * (planes[:, None] & axis[..., None, None])  # [..., 4, 3] Direction-
    move = moves.sum(axis=-1).sum(axis=-1)                                    # [...] Direction-
    # The commutator extracts the rotation plane spanned by `axis` and `move`, discarding radial stretch along `axis`.
    rotors = (axis.commutator(move) / 2).exp()                                # [...] Rotor+
    return rotors >> axis                                                     # [...] Direction-


def slip_steps(
    axis: Direction,
    planes: Plane,
    directions: Direction,
    step_slip: float,
    steepness: int,
    n_steps: int,
) -> Iterator[tuple[Direction, Scalar]]:
    """Yields the rotated tensile axes (`[...] Direction-`) and system slip increments (`[..., 4, 3] Scalar+`) over `n_steps` steps."""
    for _ in range(n_steps):
        amounts = slip_amounts(schmid_factors(axis, planes, directions), step_slip, steepness)   # [..., 4, 3] Scalar+
        axis = turned_axes(axis, planes, directions, amounts)                                    # [...] Direction-
        yield axis, amounts
