"""Refined surfaces over their control nets, with the frames of the motors, drawn on the GPU: each surface a
smooth-shaded mesh, the control net dashed rods, the frames' axes arrows."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.surfaces.motor_loop import core
from examples.surfaces.projective_loop.render import SURFACE_COLOUR, edges
from examples.util.wgsl import meshes, solids

sm = solids.mv
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
def coordinates(points: core.Point) -> np.ndarray:
    """x, y and z of each point at the sites of a field, on the last axis."""
    xyzw = points.batch().cast(core.ga.subspace("wzy wxz wyx xyz")).kernel
    return xyzw[..., :3] / xyzw[..., 3:]


def reach(points: core.Point) -> np.ndarray:
    """The mean distance from the z axis of the points at the sites of a field, per case."""
    xyz = coordinates(points)
    return np.hypot(xyz[..., 0], xyz[..., 1]).mean(axis=-1)


def separation(first: core.Point, second: core.Point) -> float:
    """The largest distance between where two fields put the same site."""
    return float(np.linalg.norm(coordinates(first) - coordinates(second), axis=-1).max())


def panel(frames: np.ndarray, surface: np.ndarray, control_faces: np.ndarray, faces: np.ndarray,
          view: solids.View) -> np.ndarray:
    """A surface `[refined, 3]`, smooth shaded, with the control net, dashed, and the control frames `[axes + 1,
    vertices, 3]`, their origins first: each axis an arrow from the origin, x red, y green, z blue."""
    pixel = view.pixel
    places = sm.point(surface)
    sheet = meshes.samples(places, faces, SURFACE_COLOUR, meshes.smooth(places, faces), view)
    # Each edge of the net cut into equal pieces, one fewer than twice the dashes, every other piece a dash.
    ends = frames[0][edges(control_faces)]                                     # [edges, 2, 3]
    share = np.linspace(0, 1, 2 * N_DASHES).reshape(N_DASHES, 2)[..., None]    # [dashes, 2, 1]
    dashes = ends[:, None, None, 0] * (1 - share) + ends[:, None, None, 1] * share   # [edges, dashes, 2, 3]
    shafts, heads = solids.arrows(sm.point(np.broadcast_to(frames[0], frames[1:].shape)), sm.point(frames[1:]),
                                  AXIS_WIDTH / 2 * pixel, HEAD_LENGTH * pixel, HEAD_WIDTH / 2 * pixel)   # [axes, vertices, 3] each
    colours = np.repeat(AXIS_COLOURS, frames.shape[1], axis=0)                 # [axes * vertices, 3]
    marks = [(solids.rods(sm.point(dashes[..., 0, :]), sm.point(dashes[..., 1, :]), NET_WIDTH / 2 * pixel).reshape(-1, 3),
              CONTROL_COLOUR), (shafts.reshape(-1, 3), colours), (heads.reshape(-1, 3), colours)]
    return solids.draw(marks, view, background, meshes=[(sheet, 1.0)])


def animate(control: core.Point, surfaces: core.Point, control_faces: np.ndarray, faces: np.ndarray
            ) -> Iterator[np.ndarray]:
    """Per frame, the surfaces of each scheme side by side, each with the control net, dashed, and the
    control frames, seen by one camera framing every frame."""
    frames, surfaces = coordinates(control), coordinates(surfaces)           # [frames, axes + 1, vertices, 3], [frames, schemes, refined, 3]
    view = solids.view([sm.point(frames), sm.point(surfaces)], 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)
    for index in range(len(frames)):
        yield np.concatenate([panel(frames[index], surface, control_faces, faces, view) for surface in surfaces[index]],
                             axis=1)
