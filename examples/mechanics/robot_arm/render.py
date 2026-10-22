"""Drawing for the robot arm, on the GPU: the links as solid boxes, the target as a ball, and the trail of the tip
as a tube."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.mechanics.robot_arm.core import Point, mv
from examples.wgsl import meshes, solids

LINK_COLOURS = np.array([to_rgb(colour) for colour in ("tab:blue", "tab:cyan", "tab:purple")])
TARGET_COLOUR = np.array(to_rgb("tab:red"))
# The links translucent, so that the target and the trail show through them: the opacity of each of a link's faces,
# front and back alike, and of the trail.
LINK_OPACITY, TRAIL_OPACITY = 0.37, 0.8
BACKGROUND = np.ones(3)
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the margin as a share of the larger
# side, and the pixels across.
ELEVATION, AZIMUTH = np.radians([22.0, 40.0])
MARGIN = 0.1
N_PIXELS = 500
# The target's radius and the trail's width, in pixels, and the vertices around the trail.
TARGET_RADIUS, TRAIL_WIDTH = 6.0, 3.0
N_AROUND = 8
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


def animate_tracking(states: Iterable) -> list[np.ndarray]:
    """One frame per solved state: the link boxes, the target, and the tip trail, all from one fixed view."""
    boxes, targets, tips = (stack(list(parts)) for parts in zip(*states))       # [frames, links, 8], [frames], [frames] Point
    view = solids.view([boxes.reshape(-1), targets], 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)
    # The tip's whole path as one tube; each frame shows it up to that frame's tip, at least its first step, which
    # the target covers.
    rings, faces = meshes.tube(euclidean(tips)[None], TRAIL_WIDTH / 2 * view.pixel, N_AROUND)
    faces_per_step = 2 * N_AROUND
    normals = meshes.smooth(rings, faces)
    frames = []
    for index in range(boxes.shape[0]):
        drawn = slice(faces_per_step * max(index, 1))
        trail = meshes.samples(rings, faces[drawn], TARGET_COLOUR, {side: normal[drawn] for side, normal in normals.items()},
                               view)
        target = solids.ball(targets[index], TARGET_RADIUS * view.pixel)[None, None]     # [1, 1] Quadric
        frames.append(solids.draw([(target, TARGET_COLOUR)], view, BACKGROUND,
                                  translucent=[(solid_boxes(boxes[index]), LINK_COLOURS, LINK_OPACITY)],
                                  meshes=[(*trail, TRAIL_OPACITY)]))
    return frames
