"""Light-ray incidence, the Robinson congruence, and linked electromagnetic field lines, drawn on the GPU: curves
as strokes, events as balls, rays as rods, and the spacetime diagram's axes as labelled arrows."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, to_rgb

from numga import concatenate, stack
from examples.util.animation import capture
from examples.relativity.twistors import core
from examples.util.wgsl import fitted, image_figure, labels, show, solids, strokes

INK, BLUE, ORANGE = "#263b51", "#267fa9", "#d17c24"
AXIS_COLOUR = np.array([0.6, 0.62, 0.66])
BACKGROUND = np.ones(3)
# The views' elevations and azimuths in radians, azimuth zero looking along +y: space, and the spacetime diagram.
ELEVATION, AZIMUTH = np.radians(22.0), np.radians(35.0)
DIAGRAM_ELEVATION, DIAGRAM_AZIMUTH = np.radians(15.0), np.radians(22.0)
# The margins around what is drawn in space and in the diagram, as shares of the larger side; the pixels across the
# congruence and across the field lines' frame.
MARGIN, DIAGRAM_MARGIN = 0.1, 0.15
CONGRUENCE_PIXELS, FIELD_PIXELS = 300, 330
# The view the field lines are carried through, as a multiple of the one framing them at the first instant, holding
# them at every instant shown.
FIELD_ROOM = 1.35
# The widths in pixels of the curves, the ray and the axes, the axes' arrowheads' length and width, and the events'
# radius; how far the axes reach past what they hold, and their names past their tips.
CURVE_WIDTH, FIELD_WIDTH, RAY_WIDTH, AXIS_WIDTH, HEAD_LENGTH, HEAD_WIDTH, EVENT_RADIUS = 1.2, 2.3, 1.7, 0.9, 7.0, 4.0, 2.5
AXIS_REACH, LABEL_DISTANCE = 1.12, 1.24


# --- plumbing -----------------------------------------------------------------------
def diagram_points(events: core.Event) -> solids.Point:
    """Draw x, z and time along the scene's x, y and z: an explicit diagram, with Euclidean picture distances."""
    mv = events.context.multivector
    readings = stack((mv.x, mv.z, mv.t)) & events[..., None]
    return solids.origin + (solids.axes.dual() * solids.context.include_from_algebra(readings)).sum(axis=-1)



def colours(count: int) -> np.ndarray:
    """One colour per curve, kept through every frame."""
    hue = np.arange(count) / count
    return hsv_to_rgb(np.stack([hue, np.full_like(hue, 0.75), np.full_like(hue, 0.8)], axis=-1))


def space_view(pixels: int, *curves: core.Event) -> solids.View:
    """A view framing every spatial curve, `pixels` across."""
    mv = curves[0].context.multivector
    Event = curves[0].algebra.gatype.antivector().derive.containing("u v")     # Antivector-
    space_slice = mv.t ^ (Event & -mv.t.dual())
    points = [solids.origin + solids.context.include_from_algebra(space_slice(batch)) for batch in curves]
    return solids.view(points, 0.0, ELEVATION, AZIMUTH, MARGIN, pixels)


def draw_curves(curves: core.Event, width: float, view: solids.View) -> np.ndarray:
    """Each spatial curve as a stroke of a width in pixels, in its own colour."""
    mv = curves.context.multivector
    Event = curves.algebra.gatype.antivector().derive.containing("u v")        # Antivector-
    space_slice = mv.t ^ (Event & -mv.t.dual())
    points = solids.origin + solids.context.include_from_algebra(space_slice(curves))
    points = points.reshape((-1, points.shape[-1]))                           # [curves, samples] Point-
    lines = strokes.polylines(list(zip(points, colours(len(points)))), width, view)
    return solids.draw([], view, BACKGROUND, inks=[], translucent=[], meshes=[(lines, 1.0)])


def draw_incidence(events: core.Event, rays: core.Event) -> plt.Figure:
    """Events on a light ray, one panel per case: [cases, events] and [cases, ray_samples] Event. Each panel is a
    spacetime diagram, x, z and t drawn along the scene's x, y and z, its axes arrows named at their tips."""
    positions, lines = diagram_points(events), diagram_points(rays)           # [cases, events] Point-, [cases, ray_samples] Point-
    # The axes reach a little past the farthest coordinate drawn, and their names a little past their tips.
    framed = concatenate((positions.reshape(-1), lines.reshape(-1)))
    widths = (solids.axes & framed[..., None]).abs().reshape(-1)
    extent = AXIS_REACH * widths.max()
    names = solids.origin + solids.axes.dual() * (LABEL_DISTANCE / AXIS_REACH * extent)   # [3] Point-
    figure = plt.figure(figsize=(5, 2.5), facecolor="white")
    panels = [figure.add_axes((column / len(positions), 0.0, 1 / len(positions), 1.0))
              for column in range(len(positions))]
    # One view for both panels, framing the axes, their names and every ray, as many pixels across as a panel holds.
    view = fitted(solids.view([solids.origin[None], names, lines], 0.0, DIAGRAM_ELEVATION, DIAGRAM_AZIMUTH,
                              DIAGRAM_MARGIN, 1), panels[0])
    shafts, heads = solids.frame(extent, AXIS_WIDTH / 2 * view.pixel, HEAD_LENGTH * view.pixel, HEAD_WIDTH / 2 * view.pixel)
    for ax, points, line in zip(panels, positions, lines):
        # The ray, straight, as a rod from its first sample to its last.
        ray = solids.rods(line[:1], line[-1:], RAY_WIDTH / 2 * view.pixel)   # [1, 3] Plane+ <- Point-
        marks = [(shafts, AXIS_COLOUR), (heads, AXIS_COLOUR), (ray, to_rgb(BLUE)),
                 (solids.ball(points, EVENT_RADIUS * view.pixel)[:, None], to_rgb(ORANGE))]
        show(ax, solids.draw(marks, view, BACKGROUND, inks=[], translucent=[], meshes=[]), view)
        # Each axis named just past its arrow's tip.
        labels(ax, view, names, list("xzt"), ha="center", va="center", fontsize=8, color=AXIS_COLOUR)
    return figure


def draw_congruence(curves: core.Event) -> plt.Figure:
    """Spatial curves tangent to the light-ray directions at one instant."""
    return image_figure(draw_curves(curves, CURVE_WIDTH, space_view(CONGRUENCE_PIXELS, curves)))


def draw_fields(curves: core.Event) -> plt.Figure:
    """Field lines at one instant."""
    return image_figure(draw_curves(curves, FIELD_WIDTH, space_view(FIELD_PIXELS, curves)))


# README picture: this makes plots/twistor_hopfion.gif, tuned as a visual for the README. Don't change it; to
# experiment, copy it.
def animate_fields(stream: Iterable[core.Event], room: float = FIELD_ROOM) -> Iterator[np.ndarray]:
    """Field lines carried through time, seen by one camera framing the first instant's lines, `room` times over."""
    stream = iter(stream)
    first = next(stream)
    framed = space_view(FIELD_PIXELS, first)
    view = framed._replace(half=framed.half * room)
    # The figure set up once, with the first instant; only its picture changes between frames.
    figure = image_figure(draw_curves(first, FIELD_WIDTH, view))
    shown = figure.axes[0].images[0]
    yield capture(figure)
    for curves in stream:
        shown.set_data(draw_curves(curves, FIELD_WIDTH, view))
        yield capture(figure)
    plt.close(figure)
