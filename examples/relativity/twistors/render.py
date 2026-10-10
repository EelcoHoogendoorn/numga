"""Light-ray incidence, the Robinson congruence, and linked electromagnetic field lines."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

from examples.animation import capture
from examples.relativity.twistors import core

INK, BLUE, ORANGE = "#263b51", "#267fa9", "#d17c24"
# Fibres near the projection point extend beyond the displayed region.
LIMIT = 3.8
# The field lines' view: how far it zooms in on the box, and the frame's size in inches and dots per inch.
FIELD_ZOOM = 1.9
FIELD_SIZE, FIELD_DPI = 4.0, 72


# --- plumbing -----------------------------------------------------------------------
def spatial(events: core.Event) -> np.ndarray:
    """The x, y and z components of events."""
    return events.cast(events.algebra.subspace("x y z")).kernel


def spacetime(events: core.Event) -> np.ndarray:
    """The x, z and t components of events."""
    return events.cast(events.algebra.subspace("x z t")).kernel


def clipped(points: np.ndarray) -> np.ndarray:
    """Break each curve where it leaves the displayed region."""
    return np.where(np.linalg.norm(points, axis=-1, keepdims=True) > LIMIT, np.nan, points)


def colours(count: int) -> np.ndarray:
    """One colour per curve, kept through every frame."""
    hue = np.arange(count) / count
    return hsv_to_rgb(np.stack([hue, np.full_like(hue, 0.75), np.full_like(hue, 0.8)], axis=-1))


def spatial_axes(figure: plt.Figure, position: tuple[float, float, float, float], zoom: float = 1.2) -> plt.Axes:
    """One fixed spatial viewport shared by the fields and animation."""
    ax = figure.add_axes(position, projection="3d")
    ax.set(xlim=(-LIMIT, LIMIT), ylim=(-LIMIT, LIMIT), zlim=(-LIMIT, LIMIT))
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    ax.set_axis_off()
    ax.view_init(elev=22, azim=-55)
    return ax


def draw_curves(ax: plt.Axes, curves: core.Event) -> None:
    """Each spatial curve in its own colour."""
    points = clipped(spatial(curves))
    points = points.reshape((-1,) + points.shape[-2:])
    for curve, rgb in zip(points, colours(len(points))):
        ax.plot(*curve.T, color=rgb, linewidth=1.1)


def draw_incidence(events: core.Event, rays: core.Event) -> plt.Figure:
    """Events on a light ray, one panel per case: [cases, events] and [cases, ray_samples] Event."""
    positions, lines = spacetime(events), spacetime(rays)
    extent = max(np.max(np.abs(positions)), np.max(np.abs(lines))) * 1.12
    figure = plt.figure(figsize=(10, 5), facecolor="white")
    for column, (points, line) in enumerate(zip(positions, lines)):
        ax = figure.add_subplot(1, len(positions), column + 1, projection="3d")
        ax.plot(*line.T, color=BLUE, linewidth=1.6)
        ax.scatter(*points.T, color=ORANGE, s=34, depthshade=False)
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), zlim=(-extent, extent),
               xlabel="x", ylabel="z", zlabel="t", xticks=[], yticks=[], zticks=[])
        ax.set_box_aspect((1, 1, 1))
        ax.set_proj_type("ortho")
        ax.view_init(elev=15, azim=-68)
        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.line.set_color("0.8")
    figure.tight_layout()
    return figure


def draw_congruence(curves: core.Event) -> plt.Figure:
    """Spatial curves tangent to the light-ray directions at one instant."""
    figure = plt.figure(figsize=(7, 6), facecolor="white")
    ax = spatial_axes(figure, (0, 0, 1, 1))
    draw_curves(ax, curves)
    return figure


def draw_fields(curves: core.Event) -> plt.Figure:
    """Field lines at one instant."""
    figure = plt.figure(figsize=(FIELD_SIZE, FIELD_SIZE), dpi=FIELD_DPI, facecolor="white")
    draw_curves(spatial_axes(figure, (0, 0, 1, 1), FIELD_ZOOM), curves)
    return figure


def animate_fields(stream: Iterable[core.Event]) -> list[np.ndarray]:
    """Field lines carried through time in a fixed spatial viewport."""
    frames = []
    for curves in stream:
        figure = draw_fields(curves)
        frames.append(capture(figure))
        plt.close(figure)
    return frames

