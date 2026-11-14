"""GPU rendering for the robot arm with translucent link boxes, a target sphere, and a tip trail."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from numga.render import simplices, solids
from numga.topology import Topology
from examples.mechanics.robot_arm.core import Point

link_colours = np.array([to_rgb(colour) for colour in ("tab:blue", "tab:cyan", "tab:purple")])
target_colour = np.array(to_rgb("tab:red"))
link_opacity, trail_opacity = 0.37, 0.8
background = np.ones(3)
elevation, azimuth = np.radians([22.0, 40.0])
view_room = 1.6
n_pixels = 500
target_radius, trail_width = 6.0, 3.0
# Six box faces indexed by three corners each across (-/+) width, (-/+) depth, and (0/1) link length.
box_faces = np.array([[0, 1, 2], [4, 5, 6], [0, 1, 4], [2, 3, 6], [0, 2, 4], [1, 3, 5]])


# --- plumbing -------------------------------------------------------------------------
def tracking_frames(states: Iterable[tuple[Point, Point, Point]]) -> Iterator[np.ndarray]:
    """Yields RGB frames (`[rows, columns, 3]`) showing the translucent link boxes, target sphere, and tip trail."""
    state_stream = iter(states)
    first_state = next(state_stream)
    initial_view = solids.view([first_state[0].reshape(-1)], 0.0, elevation, azimuth, 0.0, n_pixels)
    view = initial_view._replace(half=initial_view.half * view_room)
    tip_history = []
    for boxes, target, tip in chain([first_state], state_stream):
        tip_history.append(tip)
        trail = simplices.tubes(
            stack(tip_history),
            Topology.chains([len(tip_history)]),
            target_colour,
            trail_width,
            view,
            gloss=solids.gloss,
        )
        ball = solids.ball(target, target_radius * view.pixel)[None, None]     # [1, 1] Plane+ <- Point-
        links = solids.polyhedra(boxes, box_faces)                             # [3, 6] Plane+
        yield solids.draw(
            [(ball, target_colour)],
            view,
            background,
            inks=[],
            translucent=[(links, link_colours, link_opacity)],
            meshes=[(trail, trail_opacity)],
        )
