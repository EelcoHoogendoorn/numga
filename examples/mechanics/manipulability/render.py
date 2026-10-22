"""Renders of the arm and its ellipsoids, ray traced on the GPU: the arm's boxes opaque, an ellipsoid translucent
among them.

A box is the unit cube carried by its body map, six half-spaces. A plane carried by the map pairs with each carried
point as the cube's plane did with the point, so the box's quadrics in the world are the cube's, read through the
map's inverse on points and its adjugate's inverse on planes: `bodies.adjugate().solve(cube(bodies.solve(Point)))`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.mechanics.manipulability import core
from examples.wgsl import solids

mv, w = core.mv, core.w
box_colours = np.array([[0.45, 0.47, 0.52], [0.62, 0.64, 0.70], [0.78, 0.60, 0.30], [0.30, 0.55, 0.62], [0.55, 0.40, 0.60]])
velocity_colour, force_colour = np.array([0.25, 0.45, 0.85]), np.array([0.92, 0.55, 0.20])
# How much each of an ellipsoid's surfaces, front and back, hides of what lies behind it.
quadric_opacity = 0.33
background = np.ones(3)
# The unit cube's six half-spaces, within half a unit of the planes across the axes.
cube = solids.boxes(mv.rotor(), np.full(3, 0.5))                         # [6] Quadric


# --- plumbing -------------------------------------------------------------------------
def euclidean(points: core.Point) -> np.ndarray:
    """Euclidean coordinates of points, `[..., 3]`: their pairings with the coordinate planes, over their
    weights."""
    return ((stack((mv.x, mv.y, mv.z)) & points[..., None]) / (w & points)[..., None]).to_array()


def image(bodies: core.arm.PointMap, surface: core.Quadric, colour: np.ndarray, view: tuple[float, float],
          centre: np.ndarray, extent: float, pixels: int) -> np.ndarray:
    """One image of the arm's boxes, opaque, under a translucent quadric, square, of half-width extent about the
    centre, seen from an elevation and azimuth in degrees, the azimuth measured from x: `[pixels, pixels, 3]`
    bytes."""
    elevation, azimuth = np.radians(view[0]), np.radians(view[1] + 90)
    seen = solids.View(solids.aim(mv.point(centre), elevation, azimuth), np.full(2, extent), pixels)
    boxes = bodies[:, None].adjugate().solve(cube(bodies[:, None].solve(core.Point)))   # [boxes, 6] Quadric
    return solids.draw([(boxes, box_colours)], seen, background, translucent=[(surface[None, None], colour, quadric_opacity)])


def frame(pose, view, centre: np.ndarray, extent: float, pixels: int) -> np.ndarray:
    """One animation frame: the velocity ellipsoid beside the force ellipsoid."""
    bodies, _, velocity, force = pose
    gap = np.full((pixels, pixels // 20, 3), 255, dtype=np.uint8)
    return np.concatenate([image(bodies, velocity, velocity_colour, view, centre, extent, pixels), gap,
                           image(bodies, force, force_colour, view, centre, extent, pixels)], axis=1)


def draw_ellipsoids(poses, view, centre: np.ndarray, extent: float, pixels: int) -> plt.Figure:
    """One row per pose: the velocity ellipsoid beside the force ellipsoid, at the gripper."""
    figure, axes = plt.subplots(len(poses), 1, figsize=(11, 5.5 * len(poses)))
    for ax, pose in zip(np.atleast_1d(axes), poses):
        ax.imshow(frame(pose, view, centre, extent, pixels), interpolation="bilinear")
        ax.set_axis_off()
    figure.tight_layout()
    return figure
