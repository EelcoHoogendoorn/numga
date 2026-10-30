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
# The dashes along each edge of the control net; the widths of the net and of the frames' axes, and the length and
# width of the axes' heads, in pixels.
N_DASHES = 4
NET_WIDTH, AXIS_WIDTH, HEAD_LENGTH, HEAD_WIDTH = 2.4, 3.0, 9.0, 7.0


# --- plumbing -------------------------------------------------------------------------
def panel(frames: core.Point, surface: core.Point, control_mesh: core.Mesh, refined_mesh: core.Mesh,
          view: solids.View) -> np.ndarray:
    """Render a smooth-shaded surface `[refined] Point-` with its dashed control net and control-frame arrows
    `[axes + 1] Point-[vertices]`, each frame's origin followed by its axes' tips."""
    pixel = view.pixel
    faces = refined_mesh.faces
    sheet = meshes.samples(surface, faces, SURFACE_COLOUR, meshes.smooth(surface, refined_mesh), view, face_values=np.zeros(4),
                           colour=meshes.plain, gloss=solids.gloss)
    origins, tips = frames[0], frames[1:].batch()                              # Point-[vertices], [axes, vertices] Point-
    # Cut each edge of the control net into equal pieces, one fewer than twice the dash count, so every other piece is a dash.
    starts, ends = (control_mesh.incidence[1, 0] * origins).batch()            # [edges] Point-, Point-
    share = np.linspace(0, 1, 2 * N_DASHES).reshape(N_DASHES, 2)               # [dashes, ends]
    dashes = starts[:, None, None] * (1 - share) + ends[:, None, None] * share  # [edges, dashes, ends] Point-
    arrows = solids.arrows(origins.batch()[None], tips, AXIS_WIDTH / 2 * pixel, HEAD_LENGTH * pixel, HEAD_WIDTH / 2 * pixel)
    shafts, heads = arrows                              # [axes, vertices, 3] Plane+ <- Point-, [axes, vertices, 3] Plane+ <- Point-
    colours = np.repeat(AXIS_COLOURS, tips.shape[1], axis=0)                   # [axes * vertices, 3]
    marks = [(solids.rods(dashes[..., 0], dashes[..., 1], NET_WIDTH / 2 * pixel).reshape(-1, 3), CONTROL_COLOUR),
             (shafts.reshape(-1, 3), colours), (heads.reshape(-1, 3), colours)]
    return solids.draw(marks, view, background, inks=[], translucent=[], meshes=[(sheet, 1.0)])


def animate(control_frames: core.Point, surfaces: core.Point, control_mesh: core.Mesh, refined_mesh: core.Mesh
            ) -> Iterator[np.ndarray]:
    """Yield side-by-side panels of the point-subdivided and motor-subdivided surfaces with their control net and
    frames, `[frames, axes + 1] Point-[vertices]` and `[frames, schemes] Point-[refined]`, for each animation frame."""
    surfaces = surfaces.batch()                                                # [frames, schemes, refined] Point-
    view = solids.view([control_frames.batch(), surfaces], 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)
    for index in range(len(surfaces)):
        yield np.concatenate([panel(control_frames[index], surface, control_mesh, refined_mesh, view)
                              for surface in surfaces[index]], axis=1)
