"""Quadrics and their shared contacts in an affine view of projective space.

The quadrics are drawn where their forms vanish, sample by sample on the GPU under a camera of the plane, and their
contact planes and points of contact as strokes and marks over them."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.util.animation import capture
from examples.util.wgsl import simplices
from examples.util.wgsl.quadrics import curves
from examples.util.wgsl.camera import Camera, flat_panel
from examples.util.wgsl.simplices import Cap, Topology
from examples.math.eight_conics import core

# The eighth quadric, its three neighbours, and the other four; the contact points of a single pair.
completion_colour = np.array(to_rgb("#ea580c"))
neighbour_colours = np.array([to_rgb(colour) for colour in ("#2563eb", "#16a34a", "#db2777")])
context_colour = np.array(to_rgb("#cbd5e1"))
contact_colour = np.array(to_rgb("#0f172a"))
white = np.ones(3)
# The grid the view is framed on, its half width, and the margin around what is drawn.
n_samples = 240
half_width = 2.5
margin = 0.12
# In pixels: the widths of the curves, the eighth's, a neighbour's, the base's, the touching sphere's and the others',
# and of the contact planes, with their dashes and gaps; the radii of the points of contact and of their white rims.
eighth_width, neighbour_width, base_width, touching_width, context_width = 4.2, 2.5, 3.0, 3.6, 1.7
plane_width, plane_dashes = 1.25, (5.0, 3.0)
contact_radius, rim_radius = 5.5, 7.5
# How much the contact planes cover: the double contact's, and the neighbours' in the animation.
contact_opacity, plane_opacity = 0.45, 0.6


# --- plumbing -------------------------------------------------------------------------
def framing(shapes: core.Quadric) -> np.ndarray:
    """A square view around every curve where a batch of quadrics vanishes, with a margin `[2, 2]`: the curves found on
    a grid as its samples whose form's sign differs from a neighbour's."""
    samples = np.linspace(-half_width, half_width, n_samples)
    grid = core.mv.point(np.stack(np.meshgrid(samples, samples), axis=-1))     # [samples, samples] Point-
    crossing = np.zeros(grid.shape, dtype=bool)
    # One quadric at a time: every frame's quadrics on the grid at once would hold gigabytes.
    for shape in shapes:
        inside = (shape(grid) & grid) > 0
        crossing[:-1] |= inside[:-1] != inside[1:]
        crossing[:, :-1] |= inside[:, :-1] != inside[:, 1:]
    # The samples on the curves, read by the planes x and y over their weights.
    on_curves = grid[crossing]                                                 # [crossings] Point-
    readings = (stack((core.mv.x, core.mv.y))[:, None] & on_curves) / (core.mv.w & on_curves)   # [2, crossings] Scalar+
    low, high = readings.min(axis=-1), readings.max(axis=-1)                   # [2] Scalar+, [2] Scalar+
    centre = (low + high) / 2                                                  # [2] Scalar+
    half = (high - low).max() / 2 * (1 + margin)                               # [] Scalar+
    return stack((centre - half, centre + half)).to_array()


def plane_strokes(contacts: core.Point, reach: float) -> core.Point:
    """Each contact plane as a segment through its two points of contact `[..., 2]`, `reach` either way of their
    middle, beyond the view: `[..., ends]`."""
    middle = contacts.mean(axis=-1)                                            # [...] Point-
    # The direction from one contact to the other, at unit length: a direction's length is the size of its dual.
    run = contacts[..., 1] - contacts[..., 0]                                  # [...] Point-
    along = run / run.dual().norm()                                            # [...] Point-
    return stack((middle - reach * along, middle + reach * along), axis=-1)


def marks(points: core.Point, colours: np.ndarray, camera: Camera) -> list[np.ndarray]:
    """Points of contact, each a disc in its colour `[n, 3]` rimmed in white."""
    return [simplices.markers(points, white, rim_radius, camera),
            simplices.markers(points, colours, contact_radius, camera)]


def contact_strokes(ends: core.Point, colour: np.ndarray, opacity: float, camera: Camera) -> np.ndarray:
    """A contact plane as a dashed segment between its ends `[ends]`."""
    return simplices.strokes(ends, Topology.chains([2]), colour, plane_width, camera, dashes=plane_dashes, cap=Cap.BUTT,
                             opacity=opacity)


def canvas(bounds: np.ndarray) -> tuple[plt.Figure, plt.Axes, Camera]:
    """The figure, its axes on the view and the camera its picture is read by."""
    figure, axes = plt.subplots(figsize=(5.8, 5.8))
    axes.set(xlim=bounds[:, 0], ylim=bounds[:, 1], aspect="equal")
    axes.set_axis_off()
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return figure, axes, flat_panel(axes, core.ga)


def draw_double_contact(base: core.Quadric, touching: core.Quadric, contact_plane: core.Plane, contacts: core.Point) -> plt.Figure:
    """A quadric in double contact with a sphere: both through the same two points of the contact plane, with the
    same tangents there."""
    bounds = framing(stack((base, touching)))
    figure, axes, camera = canvas(bounds)
    reach = 2 * np.ptp(bounds, axis=0).max()
    simplices.overlay(axes, [contact_strokes(plane_strokes(contacts, reach), contact_colour, contact_opacity, camera),
                             curves(base[None], base_width, neighbour_colours[0], camera),
                             curves(touching[None], touching_width, completion_colour, camera),
                             *marks(contacts, contact_colour, camera)])
    return figure


def animate_eighth(quadrics: core.Quadric, contact_planes: core.Plane, contacts: core.Point, edges: np.ndarray,
                   closing_edges: np.ndarray) -> Iterator[np.ndarray]:
    """The cube deforming `[frames, 8]`: the eighth quadric bold, its three neighbours in their colours, each
    with their common contact plane and their two points of contact, and the other four quadrics faint."""
    n_frames, n_quadrics = quadrics.shape
    # The view framed on the quadrics of the first frame.
    bounds = framing(quadrics[0])
    neighbours = edges[closing_edges, 0]                                       # [closing edges] vertices
    others = np.setdiff1d(np.arange(n_quadrics - 1), neighbours)
    figure, axes, camera = canvas(bounds)
    reach = 2 * np.ptp(bounds, axis=0).max()
    planes = plane_strokes(contacts[:, closing_edges], reach)                  # [frames, closing edges, ends] Point-
    image = simplices.overlay(axes, [np.zeros(camera.shape + (4,))])
    for frame in range(n_frames):
        pictures = [curves(quadrics[frame, others], context_width, context_colour, camera)]
        for side, vertex in enumerate(neighbours):
            pictures += [contact_strokes(planes[frame, side], neighbour_colours[side], plane_opacity, camera),
                         curves(quadrics[frame, vertex][None], neighbour_width, neighbour_colours[side], camera)]
        pictures += [curves(quadrics[frame, n_quadrics - 1][None], eighth_width, completion_colour, camera),
                     *marks(contacts[frame, closing_edges].reshape(-1), np.repeat(neighbour_colours, 2, axis=0),
                            camera)]
        image.set_data(simplices.composited(pictures))
        yield capture(figure)
    plt.close(figure)
