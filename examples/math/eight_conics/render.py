"""Quadrics and their shared contacts in an affine view of projective space."""

from __future__ import annotations

from collections.abc import Iterator

import contourpy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from numga import stack
from examples.util.animation import capture
from examples.math.eight_conics import core

# The eighth quadric, its three neighbours, and the other four; the contact points of a single pair.
COMPLETION = "#ea580c"
NEIGHBOURS = ("#2563eb", "#16a34a", "#db2777")
CONTEXT = "#cbd5e1"
CONTACT = "#0f172a"
# The grid the quadrics are contoured on, its half width, and the margin around what is drawn.
N_SAMPLES = 480
REACH = 2.5
MARGIN = 0.12


# --- plumbing -------------------------------------------------------------------------
def xy(points: core.Point) -> np.ndarray:
    """Coordinates in the affine chart whose plane at infinity is w."""
    weight = core.mv.w & points
    return np.stack(((core.mv.x & points / weight).to_array(),
                     (core.mv.y & points / weight).to_array()), axis=-1)


def curves(shapes: core.Quadric) -> list[list[np.ndarray]]:
    """Zero contours of a batch of quadrics: each point of a grid paired with its polar plane."""
    samples = np.linspace(-REACH, REACH, N_SAMPLES)
    grid_x, grid_y = np.meshgrid(samples, samples)
    grid = core.mv.point(np.stack((grid_x, grid_y), axis=-1))
    # One quadric at a time: every frame's quadrics on the grid at once would hold gigabytes.
    return [contourpy.contour_generator(samples, samples, (shape(grid) & grid).to_array()).lines(0) for shape in shapes]


def plane_segments(planes: core.Plane, bounds: np.ndarray) -> list[list[np.ndarray]]:
    """Clip a batch of planes to the view, as the segments they are drawn as."""
    horizontal, vertical = bounds.T
    grid_x, grid_y = np.meshgrid(horizontal, vertical)
    grid = core.mv.point(np.stack((grid_x, grid_y), axis=-1))
    values = (planes[:, None, None] & grid).to_array()
    return [contourpy.contour_generator(horizontal, vertical, value).lines(0) for value in values]


def framing(paths: list[list[np.ndarray]]) -> np.ndarray:
    """A square view around every drawn curve, with a margin `[2, 2]`."""
    points = np.concatenate([segment for path in paths for segment in path])
    centre = (points.min(axis=0) + points.max(axis=0)) / 2
    half = (points.max(axis=0) - points.min(axis=0)).max() / 2 * (1 + MARGIN)
    return np.stack([centre - half, centre + half])


def canvas(bounds: np.ndarray) -> tuple[plt.Figure, plt.Axes]:
    figure, axes = plt.subplots(figsize=(5.8, 5.8))
    axes.set(xlim=bounds[:, 0], ylim=bounds[:, 1], aspect="equal")
    axes.set_axis_off()
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return figure, axes


def draw_double_contact(base: core.Quadric, touching: core.Quadric, contact_plane: core.Plane, contacts: core.Point) -> plt.Figure:
    """A quadric in double contact with a sphere: both through the same two points of the contact plane, with the
    same tangents there."""
    paths = curves(stack((base, touching)))
    bounds = framing(paths)
    figure, axes = canvas(bounds)
    axes.add_collection(LineCollection(plane_segments(contact_plane[None], bounds)[0], colors=CONTACT, linewidths=0.9,
                                       linestyles="dashed", alpha=0.45))
    axes.add_collection(LineCollection(paths[0], colors=NEIGHBOURS[0], linewidths=2.2))
    axes.add_collection(LineCollection(paths[1], colors=COMPLETION, linewidths=2.6))
    axes.scatter(*xy(contacts).T, s=90, color=CONTACT, edgecolors="white", linewidths=1.5, zorder=4)
    return figure


def animate_eighth(quadrics: core.Quadric, contact_planes: core.Plane, contacts: core.Point, edges: np.ndarray,
                   closing_edges: np.ndarray) -> Iterator[np.ndarray]:
    """The cube deforming `[frames, 8]`: the eighth quadric bold, its three neighbours in their colours, each
    with their common contact plane and their two points of contact, and the other four quadrics faint."""
    n_frames, n_quadrics = quadrics.shape
    # The view framed on the quadrics of the first frame.
    bounds = framing(curves(quadrics[0]))
    neighbours = edges[closing_edges, 0]                                       # [3] vertices
    others = np.setdiff1d(np.arange(n_quadrics - 1), neighbours)
    plane_paths = plane_segments(contact_planes[:, closing_edges].reshape(-1), bounds)
    contact_points = xy(contacts[:, closing_edges])                            # [frames, 3, 2, 2]
    # The figure and its curves laid out once; each frame only moves them.
    figure, axes = canvas(bounds)
    other_curves = [axes.add_collection(LineCollection([], colors=CONTEXT, linewidths=1.2)) for _ in others]
    plane_strokes, neighbour_curves = [], []
    for colour in NEIGHBOURS:
        plane_strokes.append(axes.add_collection(LineCollection([], colors=colour, linewidths=0.9, linestyles="dashed",
                                                              alpha=0.6)))
        neighbour_curves.append(axes.add_collection(LineCollection([], colors=colour, linewidths=1.8)))
    eighth_curve = axes.add_collection(LineCollection([], colors=COMPLETION, linewidths=3.0))
    markers = [axes.scatter(*contact_points[0, side].T, s=90, color=colour, edgecolors="white", linewidths=1.5,
                            zorder=4) for side, colour in enumerate(NEIGHBOURS)]
    for frame in range(n_frames):
        paths = curves(quadrics[frame])
        for curve, vertex in zip(other_curves, others):
            curve.set_segments(paths[vertex])
        for side, vertex in enumerate(neighbours):
            plane_strokes[side].set_segments(plane_paths[frame * 3 + side])
            neighbour_curves[side].set_segments(paths[vertex])
        eighth_curve.set_segments(paths[n_quadrics - 1])
        for side, marker in enumerate(markers):
            marker.set_offsets(contact_points[frame, side])
        yield capture(figure)
    plt.close(figure)
