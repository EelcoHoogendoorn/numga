"""The orbit of a point under a six-dimensional rotation is drawn as a tube through its first three directions and as
a circle in each of the rotation's three planes."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from examples.math.invariant_decomposition import core
from examples.util.animation import capture
from examples.util.panels import fitted, offset_panel, overlay, show
from numga.render import Camera, drawing, simplices, solids
from numga.render.drawing import compose
from numga.topology import Topology

# The planes that read a point along a direction of space.
Plane = core.ga.gatype.vector()                                                # Plane+
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
# Each plane's colour; the widths of the whole circle and of its trace up to the moment, and the radius of the moving
# point, in pixels.
plane_colours = np.array([to_rgb(colour) for colour in ("#c0392b", "#2e86c1", "#27ae60")])
circle_width, trace_width, point_radius = 0.7, 1.25, 2.0


def plane_axes(circles: core.Direction, parts: core.Bivector) -> tuple[Plane, Plane]:
    """Each plane's initial radial direction and its quarter-turn companion dualize into the two pairing planes (`[3] Plane+`)."""
    first_directions = circles[:, 0].normalized()                              # [3] Antivector-
    return first_directions.dual_inverse(), first_directions.commutator(parts).normalized().dual_inverse()   # [3] Plane+, [3] Plane+


def plane_image(circle: core.Direction, colour: np.ndarray, index: int, camera: Camera) -> np.ndarray:
    """Renders one invariant plane's grey reference circle, coloured trace up to `index`, and current point marker."""
    return drawing.composited([
        simplices.strokes(circle, Topology.chains([len(circle)]), orbit_colour, circle_width, camera),
        simplices.strokes(circle[:index + 1], Topology.chains([index + 1]), colour, trace_width, camera),
        simplices.markers(circle[index][None], colour, point_radius, camera)])


def tangle_image(places: solids.Point, index: int, view: solids.View) -> np.ndarray:
    """Renders the 3D shadow orbit as a purple-and-grey tube with a ball marker at `index`."""
    # Each segment runs from one sample of the orbit to the next, coloured purple up to the current moment and grey after.
    passed_mask = np.arange(1, len(places)) <= index                           # [...]
    colours = np.where(passed_mask[:, None], trail_colour, orbit_colour)       # [..., 3]
    tube = simplices.tubes(places, Topology.chains([len(places)]), colours, 2 * tube_radius, view,
                           gloss=solids.gloss)
    orbit = solids.layer([], view, inks=[], translucent=[], meshes=[(tube, 1.0)])
    moment = solids.layer([(solids.ball(places[index], marker_radius * view.pixel)[None, None], trail_colour)], view,
                          inks=[], translucent=[], meshes=[])
    return compose([(*orbit, 1.0), (*moment, 1.0)], np.ones(3))


def layout(points: core.Direction, circles: core.Direction, parts: core.Bivector,
           index: int) -> tuple[plt.Figure, Callable[[int], None]]:
    """Builds the figure showing the 3D orbit tangle beside the three invariant-plane circles, and returns an updater."""
    # Projecting out w, v and u leaves the orbit's 3D shadow in x, y and z, placed around the scene's origin.
    remaining = points.context.multivector.wvu                                 # Trivector+
    shadow = remaining ^ (points & remaining.dual())                           # [...] Antivector-
    places = solids.origin + solids.context.include_from_algebra(shadow)       # [...] Point-
    figure = plt.figure(figsize=figure_size)
    tangle = figure.add_axes((0.0, 0.0, tangle_width, 1.0))
    # One view frames the whole orbit at pixel resolution matching the panel.
    view = fitted(solids.view([places], 0.0, tangle_elevation, tangle_azimuth, tangle_margin, 1), tangle)
    shown = show(tangle, tangle_image(places, index, view), view)
    across_planes, along_planes = plane_axes(circles, parts)                   # [3] Plane+, [3] Plane+
    panel_width = (1 - tangle_width) / len(plane_colours)
    images, cameras = [], []
    for plane, colour in enumerate(plane_colours):
        extent = circle_room * float(circles[plane].norm().max().to_array())
        ax = figure.add_axes((tangle_width + (plane + plane_padding) * panel_width, plane_padding,
                              (1 - 2 * plane_padding) * panel_width, 1 - 2 * plane_padding))
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
        ax.set_axis_off()
        # The plane's camera reads the circle by pairing with the two orthogonal planes spanning that turn.
        cameras.append(offset_panel(ax, across_planes[plane], along_planes[plane]))
        images.append(overlay(ax, [plane_image(circles[plane], colour, index, cameras[plane])]))

    def move(moment: int) -> None:
        shown.set_data(tangle_image(places, moment, view))
        for plane, (image, camera) in enumerate(zip(images, cameras)):
            image.set_data(plane_image(circles[plane], plane_colours[plane], moment, camera))

    return figure, move


def orbit_figure(points: core.Direction, circles: core.Direction, parts: core.Bivector, index: int) -> plt.Figure:
    """Draws the 3D orbit shadow and its three invariant-plane circles up to sample `index`."""
    figure, _ = layout(points, circles, parts, index)
    return figure


def orbit_animation(points: core.Direction, circles: core.Direction, parts: core.Bivector, n_frames: int) -> Iterator[np.ndarray]:
    """Yields `n_frames` captured frames evenly spaced along the orbit in a single figure layout."""
    first, *rest = np.linspace(0, len(points) - 1, n_frames).astype(int)
    figure, move = layout(points, circles, parts, first)
    yield capture(figure)
    for index in rest:
        move(index)
        yield capture(figure)
    plt.close(figure)
