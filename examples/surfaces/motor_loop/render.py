"""Render subdivided surfaces over their dashed control nets and frame axes on the GPU."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.surfaces.motor_loop import core
from examples.surfaces.projective_loop.render import SURFACE_COLOUR
from examples.util.wgsl import meshes, solids

AXIS_COLOURS = np.array([[0.85, 0.15, 0.15], [0.2, 0.65, 0.2], [0.15, 0.3, 0.85]])
CONTROL_COLOUR = np.full(3, 0.15)
background = np.ones(3)
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the margin around the ribbon as a
# share of the larger side, and the pixels across each panel.
ELEVATION, AZIMUTH = np.radians(12.0), np.radians(30.0)
MARGIN, N_PIXELS = 0.2, 320
# The dashes along each edge of the control net; the widths of the net and of the frames' axes, and the radius of the
# balls at the axes' tips, in pixels.
N_DASHES = 4
NET_WIDTH, AXIS_WIDTH, TIP_RADIUS = 2.4, 3.0, 4.0


# --- plumbing -------------------------------------------------------------------------
def panel(frames: core.Point, surface: core.Point, control_topology: core.Topology, refined_topology: core.Topology,
          view: solids.View) -> np.ndarray:
    """Render a smooth-shaded surface `[refined] Point-` with its dashed control net and control-frame arrows
    `[axes + 1] Point-[vertices]`, each frame's origin followed by its axes' tips, each arrow a rod with a ball at its
    tip."""
    pixel = view.pixel
    faces = refined_topology.elements[2]
    sheet = meshes.samples(surface, faces, SURFACE_COLOUR, meshes.smooth(surface, refined_topology), view, face_values=np.zeros(4),
                           colour=meshes.plain, gloss=solids.gloss)
    origins, tips = frames[0], frames[1:].batch()                              # Point-[vertices], [axes, vertices] Point-
    # Cut each edge of the control net into equal pieces, one fewer than twice the dash count, so every other piece is a dash.
    starts, ends = (control_topology.incidence[1, 0] * origins).batch()        # [edges] Point-, Point-
    share = np.linspace(0, 1, 2 * N_DASHES).reshape(N_DASHES, 2)               # [dashes, ends]
    dashes = starts[:, None, None] * (1 - share) + ends[:, None, None] * share  # [edges, dashes, ends] Point-
    shafts = solids.rods(origins.batch()[None], tips, AXIS_WIDTH / 2 * pixel)   # [axes, vertices, 3] Plane+ <- Point-
    balls = solids.ball(tips.reshape(-1), TIP_RADIUS * pixel)[:, None]         # [axes * vertices, 1] Plane+ <- Point-
    colours = np.repeat(AXIS_COLOURS, tips.shape[1], axis=0)                   # [axes * vertices, 3]
    marks = [(solids.rods(dashes[..., 0], dashes[..., 1], NET_WIDTH / 2 * pixel).reshape(-1, 3), CONTROL_COLOUR),
             (shafts.reshape(-1, 3), colours), (balls, colours)]
    return solids.draw(marks, view, background, inks=[], translucent=[], meshes=[(sheet, 1.0)])


def animate(control_frames: core.Point, surfaces: core.Point, control_topology: core.Topology, refined_topology: core.Topology
            ) -> Iterator[np.ndarray]:
    """Yield side-by-side panels of the point-subdivided and motor-subdivided surfaces with their control net and
    frames, `[frames, axes + 1] Point-[vertices]` and `[frames, schemes] Point-[refined]`, for each animation frame."""
    surfaces = surfaces.batch()                                                # [frames, schemes, refined] Point-
    view = solids.view([control_frames.batch(), surfaces], 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)
    for index in range(len(surfaces)):
        yield np.concatenate([panel(control_frames[index], surface, control_topology, refined_topology, view)
                              for surface in surfaces[index]], axis=1)
