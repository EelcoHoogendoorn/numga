"""Light-ray incidence, the Robinson congruence, and linked electromagnetic field lines, drawn on the GPU: curves
as tubes, events as balls, and the spacetime diagram's axes as labelled arrows."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, to_rgb

from examples.animation import capture
from examples.relativity.twistors import core
from examples.wgsl import compose, image_figure, labels, meshes, show, solids

sm = solids.mv
INK, BLUE, ORANGE = "#263b51", "#267fa9", "#d17c24"
AXIS_COLOUR = np.array([0.6, 0.62, 0.66])
BACKGROUND = np.ones(3)
# The views' elevations and azimuths in radians, azimuth zero looking along +y: space, and the spacetime diagram.
ELEVATION, AZIMUTH = np.radians(22.0), np.radians(35.0)
DIAGRAM_ELEVATION, DIAGRAM_AZIMUTH = np.radians(15.0), np.radians(22.0)
# The margins around what is drawn in space and in the diagram, as shares of the larger side; the pixels across the
# congruence, across the field lines' frame and across a panel of the diagram.
MARGIN, DIAGRAM_MARGIN = 0.1, 0.15
CONGRUENCE_PIXELS, FIELD_PIXELS, DIAGRAM_PIXELS = 600, 288, 480
# The points around a tube; the widths in pixels of the curves, the ray and the axes, the axes' arrowheads' length
# and width, and the events' radius; how far the axes reach past what they hold, and their names past their tips.
N_AROUND = 8
CURVE_WIDTH, FIELD_WIDTH, RAY_WIDTH, AXIS_WIDTH, HEAD_LENGTH, HEAD_WIDTH, EVENT_RADIUS = 2.4, 2.0, 3.4, 1.8, 14.0, 8.0, 5.0
AXIS_REACH, LABEL_DISTANCE = 1.12, 1.24


# --- plumbing -----------------------------------------------------------------------
def spatial(events: core.Event) -> np.ndarray:
    """The x, y and z components of events."""
    return events.cast(events.algebra.subspace("x y z")).kernel


def spacetime(events: core.Event) -> np.ndarray:
    """The x, z and t components of events."""
    return events.cast(events.algebra.subspace("x z t")).kernel


def colours(count: int) -> np.ndarray:
    """One colour per curve, kept through every frame."""
    hue = np.arange(count) / count
    return hsv_to_rgb(np.stack([hue, np.full_like(hue, 0.75), np.full_like(hue, 0.8)], axis=-1))


def space_view(pixels: int, *curves: core.Event) -> solids.View:
    """A view framing every spatial curve, `pixels` across."""
    return solids.view([sm.point(spatial(batch)) for batch in curves], 0.0, ELEVATION, AZIMUTH, MARGIN, pixels)


def draw_curves(curves: core.Event, width: float, view: solids.View) -> np.ndarray:
    """Each spatial curve as a tube of a width in pixels, in its own colour."""
    points = spatial(curves)
    points = points.reshape((-1,) + points.shape[-2:])                        # [curves, samples, 3]
    tubes = meshes.tubes(list(zip(points, colours(len(points)))), width, view, N_AROUND)
    return compose([(*tubes, 1.0)], BACKGROUND)


def draw_incidence(events: core.Event, rays: core.Event) -> plt.Figure:
    """Events on a light ray, one panel per case: [cases, events] and [cases, ray_samples] Event. Each panel is a
    spacetime diagram, x, z and t drawn along the scene's x, y and z, its axes arrows named at their tips."""
    positions, lines = spacetime(events), spacetime(rays)
    extent = AXIS_REACH * max(np.max(np.abs(positions)), np.max(np.abs(lines)))
    names = sm.point(LABEL_DISTANCE / AXIS_REACH * extent * np.eye(3))      # [3] Point
    # One view for both panels, framing the axes, their names and every ray.
    view = solids.view([solids.origin[None], names, sm.point(lines)], 0.0, DIAGRAM_ELEVATION, DIAGRAM_AZIMUTH,
                       DIAGRAM_MARGIN, DIAGRAM_PIXELS)
    shafts, heads = solids.frame(extent, AXIS_WIDTH / 2 * view.pixel, HEAD_LENGTH * view.pixel, HEAD_WIDTH / 2 * view.pixel)
    figure = plt.figure(figsize=(10, 5), facecolor="white")
    for column, (points, line) in enumerate(zip(positions, lines)):
        ray = meshes.tubes([(line, to_rgb(BLUE))], RAY_WIDTH, view, N_AROUND)
        marks = solids.layer([(shafts, AXIS_COLOUR), (heads, AXIS_COLOUR),
                              (solids.ball(sm.point(points), EVENT_RADIUS * view.pixel)[:, None], to_rgb(ORANGE))],
                             view)
        ax = figure.add_axes((column / len(positions), 0.0, 1 / len(positions), 1.0))
        show(ax, compose([(*ray, 1.0), (*marks, 1.0)], BACKGROUND), view)
        # Each axis named just past its arrow's tip.
        labels(ax, view, names, list("xzt"), ha="center", va="center", fontsize=12, color=AXIS_COLOUR)
    return figure


def draw_congruence(curves: core.Event) -> plt.Figure:
    """Spatial curves tangent to the light-ray directions at one instant."""
    return image_figure(draw_curves(curves, CURVE_WIDTH, space_view(CONGRUENCE_PIXELS, curves)))


def draw_fields(curves: core.Event) -> plt.Figure:
    """Field lines at one instant."""
    return image_figure(draw_curves(curves, FIELD_WIDTH, space_view(FIELD_PIXELS, curves)))


def animate_fields(stream: Iterable[core.Event]) -> list[np.ndarray]:
    """Field lines carried through time, seen by one camera framing every instant."""
    instants = list(stream)
    view = space_view(FIELD_PIXELS, *instants)
    frames = []
    for curves in instants:
        figure = image_figure(draw_curves(curves, FIELD_WIDTH, view))
        frames.append(capture(figure))
        plt.close(figure)
    return frames
