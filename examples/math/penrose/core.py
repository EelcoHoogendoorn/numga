"""Penrose tilings by cut and project slice the five-dimensional integer lattice at a five-fold slope.

A Penrose tiling covers two-dimensional space with thick and thin rhombi in a pattern that never repeats yet shows five-fold symmetry everywhere. In the cut-and-project construction, the tiling is modelled as the shadow of the five-dimensional integer lattice cut by a two-dimensional slice tilted so that it passes through at most one lattice point. Each square face of the lattice that the cut passes close enough to projects onto a rhombus tile on the cut. Shifting the cut perpendicularly without tilting it changes which faces qualify, rearranging the tiling by local three-tile flips rather than sliding it as a whole.

Points and lattice axes are intrinsic `Antivector-` directions dual to the five coordinate mirrors `Plane+`. Four reflections across the mirrors between consecutive axes compose into an order-five rotor `turn` that cycles the five axes; averaging the five powers of `turn >> Point` weighted by cosines of fifths of a turn isolates the harmonic projections `projection(1)` onto the cut and `projection(2)` onto the perpendicular phason plane, making the cut the join `along(axes[0]) & along(axes[1])`. A lattice face becomes a tile when the cut intersects its dual three-cube along the remaining three axes, and Cramer's rule in joins alone builds the reciprocal reading planes `(second & third & cut) / (first & second & third & cut)` that test all three offsets by metric-free pairings `frames & offset`. Comparing the projected tile norm `(along(first) & along(second)).norm()` against the perpendicular dual-cube shadow norm proves that thick rhombi outnumber thin ones by the golden ratio.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext, stack

ga = Algebra("a+b+c+d+e+")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
# Mirrors and reading planes that pair with points and offsets.
Plane = ga.gatype.vector()                                                    # Plane+
# Lattice points and axis directions are intrinsic antivectors; joins of two or three span areas and volumes.
Point = ga.gatype.antivector()                                                # Point-
Area = ga.gatype.antibivector()                                               # Area-
Projection = ga.gatype((Point, Point))                                        # Point- <- Point-

# Coordinate mirrors across the five axes and their dual axis directions.
mirrors = mv.basis()                                                          # [5] Plane+
axes = mirrors.dual()                                                         # [5] Point-
# The ten axis pairs spanning lattice square faces, and the three remaining axes of each dual cube.
face_pairs = list(itertools.combinations(range(5), 2))
side_indices = np.array(face_pairs)                                           # [10, 2]
other_indices = np.array([[index for index in range(5) if index not in pair] for pair in face_pairs])  # [10, 3]


# --- math -----------------------------------------------------------------------------
# Each mirror between consecutive axes swaps them; four swaps in sequence cycle all five axes.
neighbour_mirrors = (mirrors[:-1] - mirrors[1:]).normalized()                 # [4] Plane+
turn = neighbour_mirrors[0] * neighbour_mirrors[1] * neighbour_mirrors[2] * neighbour_mirrors[3]  # [] Rotor+
turning = turn >> Point                                                       # [] Point- <- Point-


def cycle_powers(step_map: Projection, n_powers: int) -> Iterator[Projection]:
    """Yields the first `n_powers` iterated compositions (`[] Point- <- Point-`) of `step_map`."""
    power = step_map                                                          # [] Point- <- Point-
    for _ in range(n_powers):
        yield power
        power = step_map(power)


def harmonic_projection(harmonic: int) -> Projection:
    """Cosine-weighted Fourier average over the five powers of `turning` projects onto the plane turned by `harmonic` fifths of a turn (`[] Point- <- Point-`)."""
    maps = stack(list(cycle_powers(turning, 5)))                              # [5] Point- <- Point-
    weights = 2 / 5 * np.cos(2 * np.pi * harmonic * np.arange(1, 6) / 5)
    return (maps * weights).sum(axis=0)                                       # [] Point- <- Point-


def reciprocal_frames(cut: Area) -> Plane:
    """Cramer's rule in joins builds the three reading planes (`[10, 3] Plane+`) of each face's dual cube that vanish on `cut`."""
    others = axes[other_indices]                                              # [10, 3] Point-
    # Joining two complementary axes with the cut gives a plane blind to both and to the cut; dividing by the full five-volume normalizes its pairing on the remaining axis.
    cofactors = others[:, [1, 2, 0]] & others[:, [2, 0, 1]] & cut             # [10, 3] Plane+
    volumes = others[:, 0] & others[:, 1] & others[:, 2] & cut                # [10] Scalar+
    return cofactors / volumes[:, None]                                       # [10, 3] Plane+


def cube_margins(shift: Point, centres: Point, frames: Plane) -> Scalar:
    """Pairings of `frames` with the offset from `centres` to `shift` give the squared-half-width margins (`[..., 10, 3] Scalar+`) inside each dual cube."""
    offsets = frames & (shift - centres)[..., None]                           # [..., 10, 3] Scalar+
    return 1 / 4 - offsets.squared()                                          # [..., 10, 3] Scalar+


def shadow_measures(cut_projection: Projection) -> tuple[Scalar, Scalar]:
    """Projected side joins and perpendicular dual-cube joins measure each face's tile area and selection frequency (`[10] Scalar+`, `[10] Scalar+`)."""
    sides = axes[side_indices]                                                # [10, 2] Point-
    others = axes[other_indices]                                              # [10, 3] Point-
    # Subtracting the cut projection leaves the complementary projection square to the cut.
    perpendicular_projection = Point - cut_projection                         # [] Point- <- Point-
    tiles = cut_projection(sides[:, 0]) & cut_projection(sides[:, 1])         # [10] Area-
    cube_shadows = (
        perpendicular_projection(others[:, 0])
        & perpendicular_projection(others[:, 1])
        & perpendicular_projection(others[:, 2])
    )                                                                         # [10] Antitrivector-
    return tiles.norm(), cube_shadows.norm()


def lattice_faces(cut_projection: Projection, lattice_radius: int) -> tuple[Point, Point]:
    """Integer lattice points within `lattice_radius` carry ten square face centres (`[n_points, 10] Point-`) and projected corner loops (`[n_points, 10, 4] Point-`)."""
    n_span = 2 * lattice_radius + 1
    n_points = n_span ** 5
    coordinates = np.array(list(itertools.product(range(-lattice_radius, lattice_radius + 1), repeat=5)))  # [n_points, 5]
    lattice_points = (axes * coordinates).sum(axis=-1)                        # [n_points] Point-
    sides = axes[side_indices]                                                # [10, 2] Point-
    centres = lattice_points[:, None] + sides.sum(axis=-1) / 2                # [n_points, 10] Point-
    # Displace half a unit along both face axes in cyclic order around each face centre.
    first_steps, second_steps = np.array([-1, 1, 1, -1]) / 2, np.array([-1, -1, 1, 1]) / 2
    corner_offsets = sides[:, None, 0] * first_steps + sides[:, None, 1] * second_steps  # [10, 4] Point-
    return centres, cut_projection(centres[..., None] + corner_offsets)


def reading_planes(cut_projection: Projection) -> Plane:
    """Orthonormal directions in the cut dualize to two reading planes (`[2] Plane+`) aligned with the first projected axis."""
    right = cut_projection(axes[0]).normalized()                              # [] Point-
    # The commutator of two non-collinear directions in the cut generates rotations within the cut plane.
    cut_bivector = right.commutator(cut_projection(axes[1]))                  # [] Bivector+
    quarter_turn = (cut_bivector / cut_bivector.scalar_norm_squared().square_root() * (-np.pi / 4)).exp()  # [] Rotor+
    up = quarter_turn >> right                                                # [] Point-
    return stack([right, up]).dual()                                          # [2] Plane+


def phason_loop(phason_projection: Projection, origin_shift: Point, radius: float, n_frames: int) -> Point:
    """Rotors in the perpendicular phason plane sweep `origin_shift` around a closed circle of `n_frames` shifts (`[n_frames] Point-`)."""
    start = phason_projection(axes[0])                                        # [] Point-
    loop_bivector = start.commutator(phason_projection(axes[1]))              # [] Bivector+
    angles = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    rotors = (loop_bivector / loop_bivector.scalar_norm_squared().square_root() * (angles / 2)).exp()  # [n_frames] Rotor+
    circle_points = rotors >> start                                           # [n_frames] Point-
    return origin_shift + radius * (circle_points - start) / start.norm()     # [n_frames] Point-


def margin_stream(shifts: Point, centres: Point, frames: Plane) -> Iterator[Scalar]:
    """Yields the dual-cube margins (`[..., 10, 3] Scalar+`) for each shift point along `shifts`."""
    for frame_shift in shifts:
        yield cube_margins(frame_shift, centres, frames)
