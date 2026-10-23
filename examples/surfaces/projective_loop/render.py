"""Shaded surfaces over their control meshes, and the gap between two refinements, drawn on the GPU: each surface a
smooth-shaded mesh, its control mesh a dotted line of beads along every edge."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np

from examples.surfaces.projective_loop import core
from examples.wgsl import image_figure, meshes, solids

sm = solids.mv
SURFACE_COLOUR = np.array([0.16, 0.5, 0.63])
GAP_COLOUR = np.array([0.82, 0.42, 0.21])
CONTROL_COLOUR = np.full(3, 0.45)
background = np.ones(3)
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the margin around the control mesh as
# a share of its larger side, and the pixels across each panel.
ELEVATION, AZIMUTH = np.radians(18.0), np.radians(30.0)
MARGIN, N_PIXELS = 0.25, 360
# The beads along each edge of the control mesh, and their radius in pixels.
N_DOTS, DOT_RADIUS = 28, 1.4


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """x, y and z of each point at the sites of a field, on the last axis."""
    xyzw = points.batch().cast(core.ga.subspace("wzy wxz wyx xyz")).kernel
    return xyzw[..., :3] / xyzw[..., 3:]


def separation(orders: core.Point) -> float:
    """The largest distance between where two refinements, on the first axis, put the same vertex."""
    xyz = coordinates(orders)
    return float(np.linalg.norm(xyz[0] - xyz[1], axis=-1).max())


def edges(faces: np.ndarray) -> np.ndarray:
    return np.unique(np.sort(np.stack([faces, np.roll(faces, -1, axis=1)], axis=-1), axis=-1).reshape(-1, 2), axis=0)


def view(control: np.ndarray) -> solids.View:
    """A view framing a control mesh `[vertices, 3]`."""
    return solids.view([sm.point(control)], 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)


def picture(control: np.ndarray, control_faces: np.ndarray, surface: np.ndarray, faces: np.ndarray, colours: np.ndarray,
            frame: solids.View) -> np.ndarray:
    """A surface `[refined, 3]` in the given face colours `[faces, 3]`, smooth shaded, over its control mesh
    `[vertices, 3]`, dotted."""
    places = sm.point(surface)
    sheet = meshes.samples(places, faces, np.zeros(3), meshes.smooth(places, faces), frame, face_values=colours,
                           colour=meshes.by_face)
    ends = control[edges(control_faces)]                                       # [edges, 2, 3]
    share = np.linspace(0, 1, N_DOTS)[:, None]                                 # [dots, 1]
    dots = ends[:, None, 0] * (1 - share) + ends[:, None, 1] * share           # [edges, dots, 3]
    beads = [(solids.ball(sm.point(dots.reshape(-1, 3)), DOT_RADIUS * frame.pixel)[:, None], CONTROL_COLOUR)]
    return solids.draw(beads, frame, background, meshes=[(sheet, 1.0)])


def draw_weights(controls: core.Point, control_faces: np.ndarray, surfaces: core.Point, faces: np.ndarray) -> plt.Figure:
    """Each case's control mesh, dotted, and the surface it refines to, one camera framing every control mesh."""
    controls, surfaces = coordinates(controls), coordinates(surfaces)         # [cases, vertices, 3], [cases, refined, 3]
    frame = view(controls.reshape(-1, 3))
    colours = np.broadcast_to(SURFACE_COLOUR, (len(faces), 3))
    return image_figure(np.concatenate([picture(control, control_faces, surface, faces, colours, frame)
                                        for control, surface in zip(controls, surfaces)], axis=1))


def draw_orders(control: core.Point, control_faces: np.ndarray, weighted_orders: core.Point, positional_orders: core.Point,
                faces: np.ndarray) -> plt.Figure:
    """For points with their weights and for positions: the surface refined after the perspective, each
    face coloured by how far the surface refined before it lies, on one scale for both, from blue where
    they agree to orange at the largest gap."""
    xyz = coordinates(control)
    frame = view(xyz)
    gaps = [np.linalg.norm(np.subtract(*coordinates(orders)), axis=-1)[faces].mean(axis=-1)
            for orders in (weighted_orders, positional_orders)]                # [faces] each
    largest = max(gap.max() for gap in gaps)
    return image_figure(np.concatenate([picture(xyz, control_faces, coordinates(orders)[0], faces,
                                                SURFACE_COLOUR * (1 - share) + GAP_COLOUR * share, frame)
                                        for orders, share in zip((weighted_orders, positional_orders),
                                                                 (gap[:, None] / largest for gap in gaps))], axis=1))


def animate_weights(frames: Iterable[tuple[core.Point, core.Point]], control_faces: np.ndarray, faces: np.ndarray) -> list[np.ndarray]:
    """One frame per weighting: the control mesh `Point[vertices]`, dotted, and the surface it refines to
    `Point[refined]`, all framed on the first control mesh."""
    frames = [(coordinates(controls), coordinates(surface)) for controls, surface in frames]
    frame = view(frames[0][0])
    colours = np.broadcast_to(SURFACE_COLOUR, (len(faces), 3))
    return [picture(control, control_faces, surface, faces, colours, frame) for control, surface in frames]
