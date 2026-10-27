"""Drawing for the robot arm, on the GPU: the links as solid boxes, the target as a ball, and the trail of the tip
as a stroke."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.mechanics.robot_arm.core import Point
from examples.util.wgsl import solids, strokes

LINK_COLOURS = np.array([to_rgb(colour) for colour in ("tab:blue", "tab:cyan", "tab:purple")])
TARGET_COLOUR = np.array(to_rgb("tab:red"))
# The links translucent, so that the target and the trail show through them: the opacity of each of a link's faces,
# front and back alike, and of the trail.
LINK_OPACITY, TRAIL_OPACITY = 0.37, 0.8
BACKGROUND = np.ones(3)
# The view's elevation and azimuth in radians, azimuth zero looking along +y; how many times the view framing the arm
# as it starts the view holds, room for the arm as its tip follows the loop; and the pixels across.
ELEVATION, AZIMUTH = np.radians([22.0, 40.0])
VIEW_ROOM = 1.6
N_PIXELS = 500
# The target's radius and the trail's width, in pixels.
TARGET_RADIUS, TRAIL_WIDTH = 6.0, 3.0
# A box's faces, each by three of its corners: the corners lie at -/+ across, -/+ deep and bottom/top of the
# link, numbered in that order with the last fastest.
BOX_FACES = np.array([[0, 1, 2], [4, 5, 6], [0, 1, 4], [2, 3, 6], [0, 2, 4], [1, 3, 5]])


# --- plumbing -------------------------------------------------------------------------
def solid_boxes(corners: Point) -> solids.Plane:
    """The boxes of corner points `[boxes, 8]` as solids bounded by planes `[boxes, 6]`: each face the join of three
    of its corners, made extrinsic by the intrinsic scalar one, and turned so that the box's middle lies on its
    negative side."""
    faces = corners[:, BOX_FACES]                                              # [boxes, 6, 3] Point-
    planes = (faces[..., 0] & faces[..., 1] & faces[..., 2]) * solids.flip     # [boxes, 6] Plane+
    planes = planes / (planes | planes).square_root()
    middles = corners.sum(axis=-1)[:, None]                                    # [boxes, 1] Point-
    return planes * -np.sign((planes & middles).to_array())


def animate_tracking(states: Iterable, room: float = VIEW_ROOM) -> Iterator[np.ndarray]:
    """One frame per solved state: the link boxes, the target, and the tip trail, all from one fixed view framing the
    arm as it starts, `room` times over."""
    states = iter(states)
    first = next(states)
    framed = solids.view([first[0].reshape(-1)], 0.0, ELEVATION, AZIMUTH, 0.0, N_PIXELS)
    view = framed._replace(half=framed.half * room)
    path = []
    for boxes, target, tip in chain([first], states):
        # The tip's path up to this frame's tip.
        path.append(tip)
        trail = strokes.polylines([(stack(path), TARGET_COLOUR)], TRAIL_WIDTH, view)
        ball = solids.ball(target, TARGET_RADIUS * view.pixel)[None, None]      # [1, 1] Plane+ <- Point-
        yield solids.draw([(ball, TARGET_COLOUR)], view, BACKGROUND,
                          inks=[], translucent=[(solid_boxes(boxes), LINK_COLOURS, LINK_OPACITY)],
                          meshes=[(trail, TRAIL_OPACITY)])
