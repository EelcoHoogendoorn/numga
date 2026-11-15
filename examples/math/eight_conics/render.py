"""Affine views render projective quadrics, their shared contact planes, and their points of contact.

Each quadric is drawn on the GPU where its polarity form vanishes under a planar camera, with contact planes and contact points composited as dashed strokes and rimmed discs above the curves.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from numga.render import Camera, Cap, drawing, simplices
from numga.render.quadrics import curves
from numga.topology import Topology
from examples.math.eight_conics import core
from examples.util.animation import capture
from examples.util.panels import overlay, panel

completion_colour = np.array(to_rgb("#ea580c"))
neighbour_colours = np.array([to_rgb(colour) for colour in ("#2563eb", "#16a34a", "#db2777")])
context_colour = np.array(to_rgb("#cbd5e1"))
contact_colour = np.array(to_rgb("#0f172a"))
white = np.ones(3)

n_samples = 240
half_width = 2.5
margin = 0.12

eighth_width, neighbour_width, base_width, second_width, context_width = 4.2, 2.5, 3.0, 3.6, 1.7
plane_width, plane_dashes = 1.25, (5.0, 3.0)
contact_radius, rim_radius = 5.5, 7.5
contact_opacity, plane_opacity = 0.45, 0.6


def view_bounds(shapes: core.Quadric) -> np.ndarray:
    """Zero-crossing samples on a planar grid determine square `[2, 2]` view bounds around `shapes` (`[...] Plane+ <- Point-`)."""
    samples = np.linspace(-half_width, half_width, n_samples)
    grid = core.mv.point(np.stack(np.meshgrid(samples, samples), axis=-1))  # [n_samples, n_samples] Point-
    crossing_mask = np.zeros(grid.shape, dtype=bool)
    # Evaluating one quadric at a time keeps peak grid memory small.
    for shape in shapes:
        sign_mask = (shape(grid) & grid) > 0
        crossing_mask[:-1] |= sign_mask[:-1] != sign_mask[1:]
        crossing_mask[:, :-1] |= sign_mask[:, :-1] != sign_mask[:, 1:]
    curve_points = grid[crossing_mask]                                      # [...] Point-
    coordinates = (stack((core.mv.x, core.mv.y))[:, None] & curve_points) / (core.mv.w & curve_points)  # [2, ...] Scalar+
    low, high = coordinates.min(axis=-1), coordinates.max(axis=-1)          # [2] Scalar+, [2] Scalar+
    centre = (low + high) / 2                                               # [2] Scalar+
    half_span = (high - low).max() / 2 * (1 + margin)                       # [] Scalar+
    return stack((centre - half_span, centre + half_span)).to_array()


def plane_endpoints(contacts: core.Point, reach: float) -> core.Point:
    """Each pair of contact points (`[..., 2] Point-`) extends by `reach` past its midpoint to segment endpoints (`[..., 2] Point-`)."""
    midpoints = contacts.mean(axis=-1)                                      # [...] Point-
    chord = contacts[..., 1] - contacts[..., 0]                             # [...] Point-
    tangent_direction = chord / chord.antinorm()                            # [...] Point-
    return stack((midpoints - reach * tangent_direction, midpoints + reach * tangent_direction), axis=-1)  # [..., 2] Point-


def contact_markers(points: core.Point, colours: np.ndarray, camera: Camera) -> list[np.ndarray]:
    """White-rimmed coloured discs mark each contact point (`[...] Point-`) under `camera`."""
    return [
        simplices.markers(points, white, rim_radius, camera),
        simplices.markers(points, colours, contact_radius, camera),
    ]


def contact_strokes(endpoints: core.Point, colour: np.ndarray, opacity: float, camera: Camera) -> np.ndarray:
    """Dashed stroke layer connects the two `endpoints` (`[2] Point-`) of a contact plane."""
    return simplices.strokes(
        endpoints, Topology.chains([2]), colour, plane_width, camera,
        dashes=plane_dashes, cap=Cap.BUTT, opacity=opacity,
    )


def scene_canvas(bounds: np.ndarray) -> tuple[plt.Figure, plt.Axes, Camera]:
    """Square borderless matplotlib figure, axes, and planar camera span `bounds` (`[2, 2]`)."""
    figure, axes = plt.subplots(figsize=(5.8, 5.8))
    axes.set(xlim=bounds[:, 0], ylim=bounds[:, 1], aspect="equal")
    axes.set_axis_off()
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return figure, axes, panel(axes, core.mv.x, core.mv.y, core.mv.w)


def double_contact_figure(base: core.Quadric, contact_quadric: core.Quadric, contacts: core.Point) -> plt.Figure:
    """Composite figure shows `base` and `contact_quadric` sharing both crossings `contacts` (`[2] Point-`) along their contact plane."""
    bounds = view_bounds(stack((base, contact_quadric)))
    figure, axes, camera = scene_canvas(bounds)
    reach = 2 * np.ptp(bounds, axis=0).max()
    overlay(axes, [
        contact_strokes(plane_endpoints(contacts, reach), contact_colour, contact_opacity, camera),
        curves(base[None], base_width, neighbour_colours[0], camera),
        curves(contact_quadric[None], second_width, completion_colour, camera),
        *contact_markers(contacts, contact_colour, camera),
    ])
    return figure


def eighth_conic_animation(
    edges: np.ndarray,
    closing_edges: np.ndarray,
    states: Iterable[tuple[core.Quadric, core.Plane, core.Point, core.Quadric]],
) -> Iterator[np.ndarray]:
    """Yields RGB frames of the deforming cube of eight quadrics, highlighting the eighth quadric and its three neighbours."""
    state_iterator = iter(states)
    first_state = next(state_iterator)
    first_quadrics = first_state[0]
    n_quadrics = first_quadrics.shape[0]
    bounds = view_bounds(first_quadrics)
    neighbour_indices = edges[closing_edges, 0]
    context_indices = np.setdiff1d(np.arange(n_quadrics - 1), neighbour_indices)
    figure, axes, camera = scene_canvas(bounds)
    reach = 2 * np.ptp(bounds, axis=0).max()
    marker_colours = np.repeat(neighbour_colours, 2, axis=0)
    image = overlay(axes, [np.zeros(camera.shape + (4,))])
    for state in itertools.chain([first_state], state_iterator):
        quadrics, contacts = state[0], state[2]
        endpoints = plane_endpoints(contacts[closing_edges], reach)         # [3, 2] Point-
        layers = [curves(quadrics[context_indices], context_width, context_colour, camera)]
        for side, vertex in enumerate(neighbour_indices):
            layers += [
                contact_strokes(endpoints[side], neighbour_colours[side], plane_opacity, camera),
                curves(quadrics[vertex][None], neighbour_width, neighbour_colours[side], camera),
            ]
        layers += [
            curves(quadrics[n_quadrics - 1][None], eighth_width, completion_colour, camera),
            *contact_markers(contacts[closing_edges].reshape(-1), marker_colours, camera),
        ]
        image.set_data(drawing.composited(layers))
        yield capture(figure)
    plt.close(figure)
