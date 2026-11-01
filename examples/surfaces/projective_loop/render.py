"""Render subdivided surfaces over their dotted control meshes and shade the discrepancy between perspective-first and subdivision-first refinement on the GPU."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

from examples.surfaces.projective_loop import core
from examples.util.wgsl import image_figure, meshes, solids

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
def picture(control: core.Point, control_topology: core.Topology, surface: core.Point, refined_topology: core.Topology,
            colours: np.ndarray, frame: solids.View) -> np.ndarray:
    """Render a smooth-shaded surface `Point-[refined]` in the given face colours `[faces, 3]` inside its dotted
    control mesh `Point-[vertices]`."""
    points = surface.normalized().batch()                                      # [refined] Point-
    sheet = meshes.samples(points, refined_topology.elements[2], np.zeros(3), meshes.smooth(points, refined_topology), frame,
                           face_values=colours, colour=meshes.by_face, gloss=solids.gloss)
    # Beads at even shares along each edge of the control mesh, between its ends at unit weight.
    tails, heads = (control_topology.incidence[1, 0] * control.normalized()).batch()  # [edges] Point-, Point-
    share = np.linspace(0, 1, N_DOTS)                                          # [dots]
    dots = tails[:, None] * (1 - share) + heads[:, None] * share               # [edges, dots] Point-
    beads = [(solids.ball(dots.reshape(-1), DOT_RADIUS * frame.pixel)[:, None], CONTROL_COLOUR)]
    return solids.draw(beads, frame, background, inks=[], translucent=[], meshes=[(sheet, 1.0)])


def draw_weights(controls: core.Point, control_topology: core.Topology, surfaces: core.Point, refined_topology: core.Topology) -> plt.Figure:
    """Render each weight case's dotted control mesh and subdivided surface side by side from a shared camera."""
    frame = solids.view([controls.batch()], 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)
    colours = np.broadcast_to(SURFACE_COLOUR, (refined_topology.n_elements[2], 3))
    return image_figure(np.concatenate([picture(control, control_topology, surface, refined_topology, colours, frame)
                                        for control, surface in zip(controls, surfaces)], axis=1))


def draw_orders(control: core.Point, control_topology: core.Topology, weighted_orders: core.Point, positional_orders: core.Point,
                refined_topology: core.Topology) -> plt.Figure:
    """Render the surfaces subdivided after the perspective collineation for weighted points and for Euclidean positions, colouring each face from blue where the two refinement orders agree to orange at the largest discrepancy."""
    frame = solids.view([control.batch()], 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)
    # Each face's gap, the distance between the two orders' points at its corners, the join of the points at unit
    # weight, averaged.
    corner_mean = refined_topology.incidence[2, 0].mean(axis=0)                 # Scalar+[faces <- refined]
    gaps = [(corner_mean * (unit[0] & unit[1]).norm()).to_array()
            for unit in (weighted_orders.normalized(), positional_orders.normalized())]
    largest = max(gap.max() for gap in gaps)
    return image_figure(np.concatenate([picture(control, control_topology, orders[0], refined_topology,
                                                SURFACE_COLOUR * (1 - share) + GAP_COLOUR * share, frame)
                                        for orders, share in zip((weighted_orders, positional_orders),
                                                                 (gap[:, None] / largest for gap in gaps))], axis=1))


def animate_weights(frames: Iterable[tuple[core.Point, core.Point]], control_topology: core.Topology, refined_topology: core.Topology
                    ) -> Iterator[np.ndarray]:
    """Yield rendered frames of the dotted control mesh and its subdivided surface from a camera framed on the first control mesh."""
    frames = iter(frames)
    first = next(frames)
    frame = solids.view([first[0].batch()], 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)
    colours = np.broadcast_to(SURFACE_COLOUR, (refined_topology.n_elements[2], 3))
    for control, surface in chain([first], frames):
        yield picture(control, control_topology, surface, refined_topology, colours, frame)
