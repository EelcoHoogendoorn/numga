"""Scenes for Penrose tilings: the tiling cut from the five-dimensional lattice, and its phason flips
as the cut is shifted around a loop square to the plane it is laid out in."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.math.penrose import core

mv, axes = core.mv, core.axes
# The lattice points within this radius of the origin along every axis, and the coordinates of the point
# the cut passes through, summing to zero.
LATTICE_RADIUS = 3
SHIFT = np.array([0.12, 0.29, -0.11, 0.33, -0.63])
# How far the cut is shifted around its loop, the frames around it, and the duration of a frame.
PHASON = 0.35
N_FRAMES = 90
DURATION_MS = 60
# Half the width of the view, in units of the tiles' edges.
EXTENT = 2.6


# --- math -----------------------------------------------------------------------------
def cut() -> tuple[core.Projection, core.Projection, core.Join]:
    """The projections onto the plane the turn turns by a fifth and by two fifths of a turn, and the
    first plane, the cut: the join of two axes laid flat onto it."""
    along, across = core.projection(1), core.projection(2)                   # [] Antivector <- Antivector each
    return along, across, along(axes[0]) & along(axes[1])                    # [] Join


def faces(along: core.Projection) -> tuple[core.Antivector, core.Antivector]:
    """Every square face of the lattice within its radius, by its lattice point and its pair of axes: its
    centre `[points, faces]`, and its corners laid flat onto the cut `[points, faces, 4]`."""
    sides = axes[core.SIDES]                                                  # [faces, 2] Antivector
    centres = core.lattice(LATTICE_RADIUS)[:, None] + sides.sum(axis=-1) / 2  # [points, faces] Antivector
    # The corners from the centre, half a unit back and forth along both axes, around the face.
    first, second = np.array([-1, 1, 1, -1]) / 2, np.array([-1, -1, 1, 1]) / 2
    corners = sides[:, None, 0] * first + sides[:, None, 1] * second          # [faces, 4] Antivector
    return centres, along(centres[..., None] + corners)


def view(along: core.Projection) -> core.Plane:
    """The planes that read the tiling in the cut `[2]`: across the first axis laid flat, and across that
    turned a quarter turn in the cut."""
    right = along(axes[0]) / along(axes[0]).norm()                            # [] Antivector
    # The commutator of two directions in the cut turns one towards the other within it.
    turning = right.commutator(along(axes[1]))                                # [] Bivector
    up = (turning / turning.scalar_norm_squared().square_root() * (-np.pi / 4)).exp() >> right   # [] Antivector
    return stack([right, up]).dual()                                          # [2] Plane


def phasons(across: core.Projection) -> core.Antivector:
    """The points the cut passes through around its loop `[frames]`: the starting point shifted by a
    turn in the plane square to the cut and to the diagonal."""
    start = across(axes[0])                                                   # [] Antivector
    loop = start.commutator(across(axes[1]))                                  # [] Bivector
    angles = np.linspace(0, 2 * np.pi, N_FRAMES, endpoint=False)
    turned = (loop / loop.scalar_norm_squared().square_root() * (angles / 2)).exp() >> start   # [frames] Antivector
    return shift() + PHASON * (turned - start) / start.norm()                 # [frames] Antivector


def shift() -> core.Antivector:
    """The point the cut passes through."""
    return (axes * SHIFT).sum()                                               # [] Antivector


def flips(shifts: core.Antivector, centres: core.Antivector, frames: core.Plane) -> Iterator[core.Scalar]:
    """Per frame, the margins of every face within the lattice's radius `[points, faces, 3]`."""
    for shift in shifts:
        yield core.margins(shift, centres, frames)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.math.penrose import render

    along, across, plane = cut()
    frames = core.reciprocal(plane)                                           # [faces, 3] Plane
    centres, corners = faces(along)
    areas, volumes = core.shadows(along)                                      # [faces] Scalar each
    reading = view(along)                                                     # [2] Plane
    margins = core.margins(shift(), centres, frames)                          # [points, faces, 3] Scalar
    save_figure(render.draw_tiling(reading, corners, areas, margins, EXTENT), "penrose_tiling")
    images = render.animate_flips(reading, corners, areas, flips(phasons(across), centres, frames), EXTENT)
    save_animation(images, "penrose_phasons", DURATION_MS)

    # checks
    # The turn is of order five, and the reciprocal frame of each face reads its three other axes as a
    # unit frame and the cut as nothing.
    fifth = list(core.powers(core.turn() >> core.Antivector, 5))[-1]          # [] Antivector <- Antivector
    np.testing.assert_allclose((fifth(axes) - axes).norm().to_array(), 0, atol=1e-12)
    others = axes[core.OTHERS]                                                # [faces, 3] Antivector
    np.testing.assert_allclose((frames[:, None] & others[..., None]).to_array(),
                               np.broadcast_to(np.eye(3), (len(core.FACES), 3, 3)), atol=1e-12)
    np.testing.assert_allclose((frames & along(axes)[:, None, None]).to_array(), 0, atol=1e-12)
    # Each dual cube's shadow is as large as its face laid flat, thick against thin in the golden
    # ratio; the tiles in the patch follow it to within its edge.
    golden = (1 + np.sqrt(5)) / 2
    np.testing.assert_allclose(volumes.to_array(), areas.to_array(), atol=1e-12)
    thick = areas.to_array() > areas.to_array().mean()                        # [faces]
    np.testing.assert_allclose(areas.to_array()[thick] / areas.to_array()[~thick][:, None], golden, rtol=1e-12)
    tiles = np.all(margins.to_array() > 0, axis=-1)                           # [points, faces]
    np.testing.assert_allclose(tiles[:, thick].sum() / tiles[:, ~thick].sum(), golden, rtol=5e-2)


if __name__ == "__main__":
    main()
