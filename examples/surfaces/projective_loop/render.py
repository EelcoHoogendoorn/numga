"""Render subdivided surfaces over their dotted control meshes and shade the discrepancy between perspective-first and subdivision-first refinement on the GPU."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.surfaces.projective_loop import core
from examples.util.wgsl import image_figure, meshes, solids

sm = solids.mv
SURFACE_COLOUR = np.array([0.16, 0.5, 0.63])
GAP_COLOUR = np.array([0.82, 0.42, 0.21])
CONTROL_COLOUR = np.full(3, 0.45)
background = np.ones(3)
# Camera elevation and azimuth in radians (with azimuth zero looking along +y), the viewport margin as a
# share of the control mesh's larger side, and the pixel width of each panel.
ELEVATION, AZIMUTH = np.radians(18.0), np.radians(30.0)
MARGIN, N_PIXELS = 0.25, 360
# Number of beads along each control-mesh edge and their radius in pixels.
N_DOTS, DOT_RADIUS = 28, 1.4


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """The Euclidean coordinates of finite points at the sites of a field."""
    mv = core.mv
    batch = points.batch()
    return ((stack((mv.x, mv.y, mv.z)) & batch[..., None]) / (mv.w & batch)[..., None]).to_array()


def picture(control: np.ndarray, control_mesh: core.Mesh, surface: np.ndarray, refined_mesh: core.Mesh,
            colours: np.ndarray, frame: solids.View) -> np.ndarray:
    """Render a smooth-shaded surface `[refined, 3]` in the given face colours `[faces, 3]` inside its dotted control mesh `[vertices, 3]`."""
    places = sm.point(surface)
    sheet = meshes.samples(places, refined_mesh.faces, np.zeros(3), meshes.smooth(places, refined_mesh.faces), frame,
                           face_values=colours, colour=meshes.by_face, gloss=solids.gloss)
    ends = control[control_mesh.edges]                                         # [edges, 2, 3]
    share = np.linspace(0, 1, N_DOTS)[:, None]                                 # [dots, 1]
    dots = ends[:, None, 0] * (1 - share) + ends[:, None, 1] * share           # [edges, dots, 3]
    beads = [(solids.ball(sm.point(dots.reshape(-1, 3)), DOT_RADIUS * frame.pixel)[:, None], CONTROL_COLOUR)]
    return solids.draw(beads, frame, background, inks=[], translucent=[], meshes=[(sheet, 1.0)])


def draw_weights(controls: core.Point, control_mesh: core.Mesh, surfaces: core.Point, refined_mesh: core.Mesh) -> plt.Figure:
    """Render each weight case's dotted control mesh and subdivided surface side by side from a shared camera."""
    control_coords, surface_coords = coordinates(controls), coordinates(surfaces)  # [cases, vertices, 3], [cases, refined, 3]
    frame = solids.view([sm.point(control_coords.reshape(-1, 3))], 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)
    colours = np.broadcast_to(SURFACE_COLOUR, (len(refined_mesh.faces), 3))
    return image_figure(np.concatenate([picture(control, control_mesh, surface, refined_mesh, colours, frame)
                                        for control, surface in zip(control_coords, surface_coords)], axis=1))


def draw_orders(control: core.Point, control_mesh: core.Mesh, weighted_orders: core.Point, positional_orders: core.Point,
                refined_mesh: core.Mesh) -> plt.Figure:
    """Render the surfaces subdivided after the perspective collineation for weighted points and for Euclidean positions, colouring each face from blue where the two refinement orders agree to orange at the largest discrepancy."""
    xyz = coordinates(control)
    frame = solids.view([sm.point(xyz)], 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)
    # Each face's gap, the distance between the two orders' points at its corners, the join of the points at unit
    # weight, averaged.
    corner_mean = refined_mesh.incidence[2, 0].mean(axis=0)                     # Scalar+[faces <- refined]
    gaps = [(corner_mean * (unit[0] & unit[1]).norm()).to_array()
            for unit in (weighted_orders.normalized(), positional_orders.normalized())]
    largest = max(gap.max() for gap in gaps)
    return image_figure(np.concatenate([picture(xyz, control_mesh, coordinates(orders)[0], refined_mesh,
                                                SURFACE_COLOUR * (1 - share) + GAP_COLOUR * share, frame)
                                        for orders, share in zip((weighted_orders, positional_orders),
                                                                 (gap[:, None] / largest for gap in gaps))], axis=1))


def animate_weights(frames: Iterable[tuple[core.Point, core.Point]], control_mesh: core.Mesh, refined_mesh: core.Mesh
                    ) -> Iterator[np.ndarray]:
    """Yield rendered frames of the dotted control mesh and its subdivided surface from a camera framed on the first control mesh."""
    coord_frames = ((coordinates(control), coordinates(surface)) for control, surface in frames)
    first = next(coord_frames)
    frame = solids.view([sm.point(first[0])], 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)
    colours = np.broadcast_to(SURFACE_COLOUR, (len(refined_mesh.faces), 3))
    for control, surface in chain([first], coord_frames):
        yield picture(control, control_mesh, surface, refined_mesh, colours, frame)
