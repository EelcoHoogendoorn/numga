"""GPU rendering for twistors and linked light draws spatial curves as coloured tubes, spacetime events as balls, light rays as rods, and spacetime diagram axes as labelled arrows."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb, to_rgb

from numga import concatenate, stack
from examples.util.animation import capture
from examples.relativity.twistors import core
from numga.topology import Topology
from numga.render import simplices, solids
from examples.util.panels import fitted, image_figure, labels, show

mv = core.mv
ink_colour, blue_colour, orange_colour = "#263b51", "#267fa9", "#d17c24"
axis_colour = np.array([0.6, 0.62, 0.66])
background = np.ones(3)
# Camera elevations and azimuths in radians (azimuth zero looks along +y) for 3D space and the spacetime diagram.
elevation, azimuth = np.radians(22.0), np.radians(35.0)
diagram_elevation, diagram_azimuth = np.radians(15.0), np.radians(22.0)
# Fractional margins around the drawn objects and image widths in pixels for the congruence and field lines.
margin, diagram_margin = 0.1, 0.15
n_congruence_pixels, n_field_pixels = 300, 330
# Camera half-width multiplier around the initial field lines so the expanding pulse stays framed throughout the animation.
field_room = 1.35
# Stroke widths and ball radii in pixels, plus fractional axis reach and label offset past axis tips.
curve_width, field_width, ray_width, axis_width, tip_radius, event_radius = 1.2, 2.3, 1.7, 0.9, 1.6, 2.5
axis_reach, label_distance = 1.12, 1.24


# --- plumbing -----------------------------------------------------------------------
def diagram_points(events: core.Event) -> solids.Point:
    """Places spacetime `events`'s `x`, `z`, and `t` coordinates along the scene's Euclidean `x`, `y`, and `z` axes (`[...] Point-`)."""
    readings = stack((mv.x, mv.z, mv.t)) & events[..., None]                  # [..., 3] Scalar+
    return solids.origin + (solids.axes.dual() * solids.context.include_from_algebra(readings)).sum(axis=-1)  # [...] Point-


def curve_colours(n_curves: int) -> np.ndarray:
    """Evenly spaced HSV hues assign one persistent RGB colour (`[n_curves, 3]`) to each closed curve."""
    hue = np.arange(n_curves) / n_curves
    return hsv_to_rgb(np.stack([hue, np.full_like(hue, 0.75), np.full_like(hue, 0.8)], axis=-1))


def space_view(n_pixels: int, *curves: core.Event) -> solids.View:
    """Camera view of width `n_pixels` framing every spatial curve batch in `curves`."""
    space_slice = mv.t ^ (core.Event & -mv.t.dual())                          # [] Event- <- Event-
    points = [solids.origin + solids.context.include_from_algebra(space_slice(batch)) for batch in curves]
    return solids.view(points, 0.0, elevation, azimuth, margin, n_pixels)


def curves_image(curves: core.Event, width: float, view: solids.View) -> np.ndarray:
    """Renders each closed spatial curve in `curves` as a coloured tube of pixel radius `width / 2`."""
    space_slice = mv.t ^ (core.Event & -mv.t.dual())                          # [] Event- <- Event-
    points = solids.origin + solids.context.include_from_algebra(space_slice(curves))  # [...] Point-
    n_curves, n_samples = int(np.prod(points.shape[:-1])), points.shape[-1]
    lines = simplices.tubes(
        points.reshape(-1),
        Topology.loops([n_samples] * n_curves),
        np.repeat(curve_colours(n_curves), n_samples, axis=0),
        width,
        view,
        solids.gloss,
    )
    return solids.draw([], view, background, inks=[], translucent=[], meshes=[(lines, 1.0)])


def incidence_figure(events: core.Event, rays: core.Event) -> plt.Figure:
    """Two-panel spacetime diagram showing incident `events` (`[n_cases, 2] Event-`) on their reconstructed `rays` (`[n_cases, n_samples] Event-`)."""
    n_cases = len(events)
    positions, lines = diagram_points(events), diagram_points(rays)           # [n_cases, 2] Point-, [n_cases, ...] Point-
    framed = concatenate((positions.reshape(-1), lines.reshape(-1)))
    widths = (solids.axes & framed[..., None]).abs().reshape(-1)
    extent = axis_reach * widths.max()
    names = solids.origin + solids.axes.dual() * (label_distance / axis_reach * extent)  # [3] Point-
    figure = plt.figure(figsize=(5, 2.5), facecolor="white")
    panels = [figure.add_axes((column / n_cases, 0.0, 1 / n_cases, 1.0)) for column in range(n_cases)]
    view = fitted(
        solids.view([solids.origin[None], names, lines], 0.0, diagram_elevation, diagram_azimuth, diagram_margin, 1),
        panels[0],
    )
    axis_tips = solids.origin + solids.axes.dual() * extent                   # [3] Point-
    shafts = solids.rods(solids.origin, axis_tips, axis_width / 2 * view.pixel)
    heads = solids.ball(axis_tips, tip_radius * view.pixel)[:, None]
    for ax, points, line in zip(panels, positions, lines):
        ray = solids.rods(line[:1], line[-1:], ray_width / 2 * view.pixel)    # [1, 3] Plane+ <- Point-
        marks = [
            (shafts, axis_colour),
            (heads, axis_colour),
            (ray, to_rgb(blue_colour)),
            (solids.ball(points, event_radius * view.pixel)[:, None], to_rgb(orange_colour)),
        ]
        show(ax, solids.draw(marks, view, background, inks=[], translucent=[], meshes=[]), view)
        labels(ax, view, names, list("xzt"), ha="center", va="center", fontsize=8, color=axis_colour)
    return figure


def congruence_figure(curves: core.Event) -> plt.Figure:
    """Renders the linked spatial curves (`[...] Event-`) tangent to the Robinson congruence at time zero."""
    return image_figure(curves_image(curves, curve_width, space_view(n_congruence_pixels, curves)))


def fields_figure(curves: core.Event) -> plt.Figure:
    """Renders the linked electric field lines (`[...] Event-`) at one instant."""
    return image_figure(curves_image(curves, field_width, space_view(n_field_pixels, curves)))


def fields_animation(states: Iterable[core.Event]) -> Iterator[np.ndarray]:
    """Streams RGB frames of the linked field lines (`[...] Event-`) through a fixed camera scaled by `field_room` around the initial frame."""
    states = iter(states)
    first = next(states)
    framed = space_view(n_field_pixels, first)
    view = framed._replace(half=framed.half * field_room)
    figure = image_figure(curves_image(first, field_width, view))
    shown = figure.axes[0].images[0]
    yield capture(figure)
    for curves in states:
        shown.set_data(curves_image(curves, field_width, view))
        yield capture(figure)
    plt.close(figure)
