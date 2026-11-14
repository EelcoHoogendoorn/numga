"""Renders subdivided surfaces over their dotted control cages and shades the discrepancy between perspective-first and subdivision-first refinement on the GPU."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

from examples.surfaces.projective_loop import core
from numga.render import meshes, solids
from examples.util.panels import image_figure

surface_colour = np.array([0.16, 0.5, 0.63])
gap_colour = np.array([0.82, 0.42, 0.21])
control_colour = np.full(3, 0.45)
background = np.ones(3)
# Camera elevation and azimuth in radians (with azimuth zero looking along +y), the viewport margin as a
# share of the control cage's larger side, and the pixel width of each panel.
elevation, azimuth = np.radians(18.0), np.radians(30.0)
margin, n_pixels = 0.25, 360
# Number of beads along each control-cage edge and their radius in pixels.
n_dots, dot_radius = 28, 1.4


# --- plumbing -------------------------------------------------------------------------
def surface_panel(control: core.Point, control_topology: core.Topology, surface: core.Point, refined_topology: core.Topology,
                  colours: np.ndarray, view: solids.View) -> np.ndarray:
    """Renders a smooth-shaded surface (`Point-[V_refined]`) in the given face colours (`[F_refined, 3]`) inside its dotted control cage (`Point-[V]`)."""
    V, E, F = control_topology.n_elements
    V_refined, E_refined, F_refined = refined_topology.n_elements
    points = surface.normalized().batch()                                                   # [V_refined] Point-
    sheet = meshes.samples(points, refined_topology.elements[2], np.zeros(3), meshes.smooth(points, refined_topology), view,
                           face_values=colours, colour=meshes.by_face, gloss=solids.gloss)
    # Beads at even shares along each edge of the control cage, between its ends at unit weight.
    tails, heads = (control_topology.incidence[1, 0] * control.normalized()).batch()        # [E] Point-, Point-
    share = np.linspace(0, 1, n_dots)                                                       # [n_dots]
    dots = tails[:, None] * (1 - share) + heads[:, None] * share                            # [E, n_dots] Point-
    beads = [(solids.ball(dots.reshape(-1), dot_radius * view.pixel)[:, None], control_colour)]
    return solids.draw(beads, view, background, inks=[], translucent=[], meshes=[(sheet, 1.0)])


def weights_figure(controls: core.Point, control_topology: core.Topology, surfaces: core.Point, refined_topology: core.Topology) -> plt.Figure:
    """Renders each weight case's dotted control cage and subdivided surface side by side from a shared camera."""
    V_refined, E_refined, F_refined = refined_topology.n_elements
    view = solids.view([controls.batch()], 0.0, elevation, azimuth, margin, n_pixels)
    colours = np.broadcast_to(surface_colour, (F_refined, 3))
    return image_figure(np.concatenate([surface_panel(control, control_topology, surface, refined_topology, colours, view)
                                        for control, surface in zip(controls, surfaces)], axis=1))


def orders_figure(control: core.Point, control_topology: core.Topology, weighted_orders: core.Point, positional_orders: core.Point,
                  refined_topology: core.Topology) -> plt.Figure:
    """Renders the surfaces subdivided after the perspective collineation for weighted points and for Euclidean positions, colouring each face from blue where the two refinement orders agree to orange at the largest discrepancy."""
    view = solids.view([control.batch()], 0.0, elevation, azimuth, margin, n_pixels)
    # Each face's gap averages the Euclidean distance between the two orders' unit-weight points across its three corners.
    corner_mean = refined_topology.incidence[2, 0].mean(axis=0)                             # Scalar+[F_refined <- V_refined]
    gaps = [(corner_mean * (unit[0] & unit[1]).norm()).to_array()
            for unit in (weighted_orders.normalized(), positional_orders.normalized())]
    largest = max(gap.max() for gap in gaps)
    return image_figure(np.concatenate([surface_panel(control, control_topology, orders[0], refined_topology,
                                                      surface_colour * (1 - share) + gap_colour * share, view)
                                        for orders, share in zip((weighted_orders, positional_orders),
                                                                 (gap[:, None] / largest for gap in gaps))], axis=1))


def weights_animation(frames: Iterable[tuple[core.Point, core.Point]], control_topology: core.Topology, refined_topology: core.Topology
                      ) -> Iterator[np.ndarray]:
    """Yields rendered frames of the dotted control cage and its subdivided surface from a camera framed on the first control cage."""
    V_refined, E_refined, F_refined = refined_topology.n_elements
    frames = iter(frames)
    first = next(frames)
    view = solids.view([first[0].batch()], 0.0, elevation, azimuth, margin, n_pixels)
    colours = np.broadcast_to(surface_colour, (F_refined, 3))
    for control, surface in chain([first], frames):
        yield surface_panel(control, control_topology, surface, refined_topology, colours, view)
