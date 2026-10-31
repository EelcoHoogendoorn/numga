"""Quadrics and their shared contacts in an affine view of projective space.

The quadrics are drawn where their forms vanish, pixel by pixel on the GPU, and their contact planes and points of
contact as strokes and marks over them."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.util.animation import capture
from examples.util.wgsl import pinned, planar, room
from examples.util.wgsl.simplices import Cap, Marker
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
    places = np.stack(np.meshgrid(samples, samples), axis=-1)                 # [samples, samples, 2]
    grid = core.mv.point(places)                                               # [samples, samples] Point-
    crossing = np.zeros(places.shape[:2], dtype=bool)
    # One quadric at a time: every frame's quadrics on the grid at once would hold gigabytes.
    for shape in shapes:
        inside = (shape(grid) & grid) > 0
        crossing[:-1] |= inside[:-1] != inside[1:]
        crossing[:, :-1] |= inside[:, :-1] != inside[:, 1:]
    points = places[crossing]
    centre = (points.min(axis=0) + points.max(axis=0)) / 2
    half = (points.max(axis=0) - points.min(axis=0)).max() / 2 * (1 + margin)
    return np.stack([centre - half, centre + half])


def plane_strokes(contacts: core.Point, reach: float) -> core.Point:
    """Each contact plane as a segment through its two points of contact `[..., 2]`, `reach` either way of their
    middle, beyond the view: `[..., ends]`."""
    middle = contacts.mean(axis=-1)                                            # [...] Point-
    # The direction from one contact to the other, at unit length: a direction's length is the size of its dual.
    run = contacts[..., 1] - contacts[..., 0]                                  # [...] Point-
    along = run / run.dual().norm()                                            # [...] Point-
    return stack((middle - reach * along, middle + reach * along), axis=-1)


def marks(points: core.Point, colours: np.ndarray) -> list[planar.Marks]:
    """Points of contact, each a disc in its colour `[n, 3]` rimmed in white."""
    return [planar.Marks(points, Marker.DISC, 0.0, rim_radius, white, 1.0),
            planar.Marks(points, Marker.DISC, 0.0, contact_radius, colours, 1.0)]


def canvas(bounds: np.ndarray) -> tuple[plt.Figure, plt.Axes, planar.Window]:
    """The figure, its axes on the view and the window its picture fills."""
    figure, axes = plt.subplots(figsize=(5.8, 5.8))
    axes.set(xlim=bounds[:, 0], ylim=bounds[:, 1], aspect="equal")
    axes.set_axis_off()
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return figure, axes, planar.filling(tuple(bounds[:, 0]), tuple(bounds[:, 1]), room(axes))


def draw_double_contact(base: core.Quadric, touching: core.Quadric, contact_plane: core.Plane, contacts: core.Point) -> plt.Figure:
    """A quadric in double contact with a sphere: both through the same two points of the contact plane, with the
    same tangents there."""
    bounds = framing(stack((base, touching)))
    figure, axes, window = canvas(bounds)
    reach = 2 * np.ptp(bounds, axis=0).max()
    layers = [planar.Strokes(plane_strokes(contacts, reach)[None], plane_width, contact_colour, contact_opacity,
                             plane_dashes, Cap.BUTT),
              planar.Curves(base[None], base_width, neighbour_colours[0], 1.0),
              planar.Curves(touching[None], touching_width, completion_colour, 1.0),
              *marks(contacts, contact_colour)]
    pinned(axes, planar.draw(window, layers, white), window.extent)
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
    figure, axes, window = canvas(bounds)
    reach = 2 * np.ptp(bounds, axis=0).max()
    planes = plane_strokes(contacts[:, closing_edges], reach)                  # [frames, closing edges, ends] Point-
    image = pinned(axes, np.ones(window.shape + (3,)), window.extent)
    for frame in range(n_frames):
        layers = [planar.Curves(quadrics[frame, others], context_width, context_colour, 1.0)]
        for side, vertex in enumerate(neighbours):
            layers += [planar.Strokes(planes[frame, side][None], plane_width, neighbour_colours[side], plane_opacity,
                                      plane_dashes, Cap.BUTT),
                       planar.Curves(quadrics[frame, vertex][None], neighbour_width, neighbour_colours[side], 1.0)]
        layers += [planar.Curves(quadrics[frame, n_quadrics - 1][None], eighth_width, completion_colour, 1.0),
                   *marks(contacts[frame, closing_edges].reshape(-1), np.repeat(neighbour_colours, 2, axis=0))]
        image.set_data(planar.draw(window, layers, white))
        yield capture(figure)
    plt.close(figure)
