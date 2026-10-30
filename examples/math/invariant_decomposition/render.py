"""The orbit of a point under a six-dimensional rotation is drawn as a tube through its first three directions and as
a circle in each of the rotation's three planes."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.math.invariant_decomposition import core
from examples.util.animation import capture
from examples.util.wgsl import compose, fitted, show, solids, strokes

# The figure's size in inches, at a hundred pixels to the inch, and the share of its width the tangle takes, the planes
# sharing the rest; the share of a plane's panel left free around its view, and how many times its circle's radius
# the view spans to either side of the middle, room for the marks' width.
figure_size = (6.0, 2.0)
tangle_width = 1 / 3
plane_padding = 0.08
circle_room = 1.1
# The tangle's view, elevation and azimuth in radians, azimuth zero looking along +y; its margin, as a share of its
# larger side; the radii of the orbit's tube and of the moving point, in pixels; the orbit's colour up to the moment
# and after it.
tangle_elevation, tangle_azimuth = np.radians([30.0, 30.0])
tangle_margin = 0.16
tube_radius, marker_radius = 0.9, 2.3
trail_colour, orbit_colour = np.array(to_rgb("#7d3c98")), np.array(to_rgb("0.8"))
# Each plane's colour; the widths of the whole circle and of its trace up to the moment, in points, and the area of
# the moving point, in square points.
plane_colours = ("#c0392b", "#2e86c1", "#27ae60")
circle_width, trace_width, marker_area = 0.5, 0.9, 8.0


def gpu_points(points: core.Antivector) -> solids.Point:
    """View the source along its remaining directions, with its x, y and z slices along the scene's axes."""
    mv = points.context.multivector
    readings = stack((mv.x, mv.y, mv.z)) & points[..., None]
    return solids.origin + (solids.axes.dual() * solids.context.include_from_algebra(readings)).sum(axis=-1)



def plane_axes(circles: core.Antivector, parts: core.Bivector) -> tuple[core.Antivector, core.Antivector]:
    """In each plane, the direction of the first projected point and the direction a quarter turn on from it."""
    first = circles[:, 0].normalized()                                         # [3] Antivector-
    return first, first.commutator(parts).normalized()                         # [3] Antivector-, [3] Antivector-


def tangle_image(places: solids.Point, index: int, view: solids.View) -> np.ndarray:
    """The orbit through its first three directions as a tube, purple up to one moment and grey after it, with the
    point at that moment as a ball."""
    # Each segment runs from one time of the orbit to the next: purple up to the moment, grey after it.
    passed = np.arange(1, len(places)) <= index                                # [times - 1]
    colours = np.where(passed[:, None], trail_colour, orbit_colour)            # [times - 1, 3]
    tube = strokes.segments(places[:-1], places[1:], colours, 2 * tube_radius, view)
    orbit = solids.layer([], view, inks=[], translucent=[], meshes=[(tube, 1.0)])
    moment = solids.layer([(solids.ball(places[index], marker_radius * view.pixel)[None, None], trail_colour)], view,
                          inks=[], translucent=[], meshes=[])
    return compose([(*orbit, 1.0), (*moment, 1.0)], np.ones(3))


def layout(points: core.Antivector, circles: core.Antivector, parts: core.Bivector,
           index: int) -> tuple[plt.Figure, Callable[[int], None]]:
    """The orbit in the first three directions, and in each of the three planes, up to one moment: the figure, and
    what moves it on to another moment."""
    places = gpu_points(points)                                                # [times] Point-
    figure = plt.figure(figsize=figure_size)
    tangle = figure.add_axes((0.0, 0.0, tangle_width, 1.0))
    # One view framing the whole orbit, as many pixels across as the tangle's room holds.
    view = fitted(solids.view([places], 0.0, tangle_elevation, tangle_azimuth, tangle_margin, 1), tangle)
    shown = show(tangle, tangle_image(places, index, view), view)
    across, along = plane_axes(circles, parts)
    flat = np.stack([circles.dot(across[:, None]).to_array(), circles.dot(along[:, None]).to_array()], axis=-1)   # [3, times, 2]
    panel = (1 - tangle_width) / len(plane_colours)
    traces, markers = [], []
    for plane, colour in enumerate(plane_colours):
        extent = circle_room * np.abs(flat[plane]).max()
        ax = figure.add_axes((tangle_width + (plane + plane_padding) * panel, plane_padding,
                              (1 - 2 * plane_padding) * panel, 1 - 2 * plane_padding))
        ax.plot(*flat[plane].T, color=orbit_colour, linewidth=circle_width)
        traces += ax.plot(*flat[plane, : index + 1].T, color=colour, linewidth=trace_width)
        markers.append(ax.scatter(*flat[plane, index], color=colour, s=marker_area, zorder=3))
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
        ax.set_axis_off()

    def move(moment: int) -> None:
        shown.set_data(tangle_image(places, moment, view))
        for plane, (trace, marker) in enumerate(zip(traces, markers)):
            trace.set_data(*flat[plane, : moment + 1].T)
            marker.set_offsets(flat[plane, moment])

    return figure, move


def draw(points: core.Antivector, circles: core.Antivector, parts: core.Bivector, index: int) -> plt.Figure:
    """The orbit in the first three directions, and in each of the three planes, up to one moment."""
    figure, _ = layout(points, circles, parts, index)
    return figure


def animate(points: core.Antivector, circles: core.Antivector, parts: core.Bivector, frames: int) -> Iterator[np.ndarray]:
    """The orbit up to moments evenly spread over it, in a figure laid out once."""
    first, *rest = np.linspace(0, len(points) - 1, frames).astype(int)
    figure, move = layout(points, circles, parts, first)
    yield capture(figure)
    for index in rest:
        move(index)
        yield capture(figure)
    plt.close(figure)
