"""Drawing for the robot arm, on the GPU: the links as solid boxes, the target as a ball, and the trail of the tip
as a stroke."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.mechanics.robot_arm.core import Point, mv
from examples.util.wgsl import meshes, solids, strokes

LINK_COLOURS = np.array([to_rgb(colour) for colour in ("tab:blue", "tab:cyan", "tab:purple")])
TARGET_COLOUR = np.array(to_rgb("tab:red"))
# The links translucent, so that the target and the trail show through them: the opacity of each of a link's faces,
# front and back alike, and of the trail.
LINK_OPACITY, TRAIL_OPACITY = 0.37, 0.8
BACKGROUND = np.ones(3)
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the point it looks at, half its width
# and height, holding the arm as its tip follows the loop, and the pixels across.
ELEVATION, AZIMUTH = np.radians([22.0, 40.0])
VIEW_CENTRE, VIEW_HALF = np.array([0.43, 0.85, 0.93]), np.array([1.25, 1.25])
N_PIXELS = 500
# The target's radius and the trail's width, in pixels.
TARGET_RADIUS, TRAIL_WIDTH = 6.0, 3.0
# A box's faces, each by three of its corners: the corners lie at -/+ across, -/+ deep and bottom/top of the
# link, numbered in that order with the last fastest.
BOX_FACES = np.array([[0, 1, 2], [4, 5, 6], [0, 1, 4], [2, 3, 6], [0, 2, 4], [1, 3, 5]])


# --- plumbing -------------------------------------------------------------------------
def euclidean(points: Point) -> np.ndarray:
    """Coordinates of points: their pairings with the planes x, y and z, over their weights."""
    return ((stack((mv.x, mv.y, mv.z)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


def solid_boxes(corners: Point) -> solids.Quadric:
    """The boxes of corner points `[boxes, 8]` as solids `[boxes, 6]`: the half-spaces inside each face, the face
    the join of three of its corners, made extrinsic by the intrinsic unit, and turned so that the box's middle lies
    on its negative side."""
    faces = corners[:, BOX_FACES]                                              # [boxes, 6, 3] Point
    planes = (faces[..., 0] & faces[..., 1] & faces[..., 2]) * meshes.unit     # [boxes, 6] Plane
    planes = planes / (planes | planes).square_root()
    middles = corners.sum(axis=-1)[:, None]                                    # [boxes, 1] Point
    return solids.half_space(planes * -np.sign((planes & middles).to_array()))


def animate_tracking(states: Iterable, centre: np.ndarray = VIEW_CENTRE,
                     half: np.ndarray = VIEW_HALF) -> Iterator[np.ndarray]:
    """One frame per solved state: the link boxes, the target, and the tip trail, all from one fixed view looking at
    `centre`, `half` its width and height."""
    view = solids.View(solids.aim(mv.point(centre), ELEVATION, AZIMUTH), half, N_PIXELS)
    path = []
    for boxes, target, tip in states:
        # The tip's path up to this frame's tip.
        path.append(euclidean(tip))
        trail = strokes.polylines([(np.array(path), TARGET_COLOUR)], TRAIL_WIDTH, view)
        ball = solids.ball(target, TARGET_RADIUS * view.pixel)[None, None]      # [1, 1] Quadric
        yield solids.draw([(ball, TARGET_COLOUR)], view, BACKGROUND,
                          translucent=[(solid_boxes(boxes), LINK_COLOURS, LINK_OPACITY)],
                          meshes=[(trail, TRAIL_OPACITY)])
