"""Renders of the arm and its ellipsoids, ray traced on the GPU: the arm's boxes opaque, an ellipsoid translucent
among them.

A box is the unit cube carried by its body map, bounded by six planes. A plane carried by the map pairs with each
carried point as the cube's plane did with the point, so the box's planes in the world are the cube's, through the
adjugate's inverse: `bodies.adjugate().solve(cube)`.
"""

from __future__ import annotations

from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.mechanics.manipulability import core
from examples.util.wgsl import room, show_pixels, solids

mv, w = core.mv, core.w
box_colours = np.array([[0.45, 0.47, 0.52], [0.62, 0.64, 0.70], [0.78, 0.60, 0.30], [0.30, 0.55, 0.62], [0.55, 0.40, 0.60]])
velocity_colour, force_colour = np.array([0.25, 0.45, 0.85]), np.array([0.92, 0.55, 0.20])
# How much each of an ellipsoid's surfaces, front and back, hides of what lies behind it.
quadric_opacity = 0.33
background = np.ones(3)
# The unit cube's six planes, half a unit from the origin across the axes, the cube on their negative sides.
cube = solids.boxes(mv.rotor(), np.full(3, 0.5))                         # [6] Plane+
# Its eight corners.
cube_corners = mv.point(list(product((-0.5, 0.5), repeat=3)))            # [8] Point-
# The room a still's panel leaves around the arm and its ellipsoid on every side, as a share of their larger
# half-width.
margin = 0.3


# --- plumbing -------------------------------------------------------------------------
def angles(view: tuple[float, float]) -> tuple[float, float]:
    """An elevation and an azimuth in degrees, the azimuth measured from x, as `solids` takes them: in radians, the
    azimuth measured from y."""
    return np.radians(view[0]), np.radians(view[1] + 90)


def held(bodies: core.arm.PointMap, surface: core.Quadric, view: tuple[float, float], pixels: int) -> solids.View:
    """A square view, from an elevation and azimuth in degrees, that holds the arm's boxes and a quadric with a margin
    of `margin`, `pixels` across."""
    elevation, azimuth = angles(view)
    boxes = bodies[:, None](cube_corners)                                    # [boxes, 8] Point-
    # The planes through the origin square to the image's right and up; shifted by `w * shift` they touch the quadric
    # where they pair to zero with its dual, `plane & dual(plane)`, a quadratic in the shift.
    screen = solids.aim(solids.origin, elevation, azimuth) >> stack((mv.x, mv.y))   # [2] Plane+
    dual = surface.inverse()                                                 # [] Point- <- Plane+
    square, linear, constant = (w & dual(w)).to_array(), (screen & dual(w)).to_array(), (screen & dual(screen)).to_array()
    root = np.sqrt(linear**2 - square * constant)
    shifts = np.stack([(-linear - root) / square, (-linear + root) / square])   # [sides, 2]
    # The quadric's outermost points across and up the image: the poles of its tangent planes there.
    outermost = dual(screen + w * shifts)                                    # [sides, 2] Point-
    return solids.view([boxes, outermost], 0.0, elevation, azimuth, margin, pixels).squared()


def image(bodies: core.arm.PointMap, surface: core.Quadric, colour: np.ndarray, seen: solids.View) -> np.ndarray:
    """One image of the arm's boxes, opaque, under a translucent quadric, as a view sees them: `[rows, columns, 3]`
    bytes."""
    boxes = bodies[:, None].adjugate().solve(cube)                           # [boxes, 6] Plane+
    return solids.draw([(boxes, box_colours)], seen, background, inks=[],
                       translucent=[(surface[None, None], colour, quadric_opacity)], meshes=[])


def side_by_side(velocity: np.ndarray, force: np.ndarray) -> np.ndarray:
    """The velocity ellipsoid's image beside the force ellipsoid's, a twentieth of a panel apart."""
    gap = np.full((velocity.shape[0], velocity.shape[1] // 20, 3), 255, dtype=np.uint8)
    return np.concatenate([velocity, gap, force], axis=1)


def frame(pose, view, centre: np.ndarray, extent: float, pixels: int) -> np.ndarray:
    """One animation frame: the velocity ellipsoid beside the force ellipsoid, in one fixed square view of half-width
    extent about the centre, seen from an elevation and azimuth in degrees."""
    bodies, _, velocity, force = pose
    seen = solids.View(solids.aim(mv.point(centre), *angles(view)), np.full(2, extent), pixels)
    return side_by_side(image(bodies, velocity, velocity_colour, seen), image(bodies, force, force_colour, seen))


def draw_ellipsoids(poses, view) -> plt.Figure:
    """One row per pose: the velocity ellipsoid beside the force ellipsoid, at the gripper, each panel framed to hold
    the arm and its ellipsoid."""
    figure, axes = plt.subplots(len(poses), 1, figsize=(5.5, 2.75 * len(poses)))
    axes = np.atleast_1d(axes)
    # The rows' rooms alike: each panel as many pixels across as the two and the gap between them, a twentieth of a
    # panel, fit in a row's room.
    columns, rows = room(axes[0])
    pixels = min(rows, columns * 20 // 41)
    for ax, (bodies, _, velocity, force) in zip(axes, poses):
        show_pixels(ax, side_by_side(image(bodies, velocity, velocity_colour, held(bodies, velocity, view, pixels)),
                                     image(bodies, force, force_colour, held(bodies, force, view, pixels))))
    return figure
