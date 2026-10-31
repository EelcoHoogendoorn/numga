"""Pressure, streamlines and lift around a wing."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm, to_rgb, to_rgba

from numga import Extensor, concatenate, stack

from examples import instantiate
from examples.util.animation import capture
from examples.util.wgsl import contours, simplices
from examples.util.wgsl.camera import Camera, flat_panel, panel
from examples.fluids.wing import core

Mesh = instantiate("examples.util.mesh", core.ga).Mesh

window = (-3.2, 3.2, -1.8, 1.8)
pressure_range = (-3.0, 1.0)
# The streamlines: the stream function's levels a step apart, one through zero, and their ink and width in pixels.
stream_spacing = 0.3
stream_ink = np.array(to_rgba("#334155"))
stream_width = 0.5
lift_scale = 0.35
wing_colour = "#0f172a"
lift_colour = "#16a34a"
mark_colour = "#facc15"
centre_colour = "#f8fafc"
# The animations' dots per inch: the README's wing.gif 400 pixels across.
frame_dpi = 50
# Widths, lengths and radii in pixels at the frames' dots per inch: the lift's arrow and the strokes of the chevron at
# its tip; the marks and their rims.
lift_width, lift_chevron = 1.7, 6.8
mark_radius, mark_rim = 2.2, 0.56
# The map's figure, at its own dots per inch: the grid's curves, the wing's circle and its image, the marks.
map_dpi = 40
grid_width, outline_width, map_mark_radius, map_mark_rim = 0.5, 1.2, 1.9, 0.56


# --- plumbing -------------------------------------------------------------------------
def rgb(colour: str) -> np.ndarray:
    """A named colour's red, green and blue."""
    return np.array(to_rgb(colour))


def marked(marks: core.Point, colour: str, radius: float, rim: float, camera: Camera) -> list[np.ndarray]:
    """Marks at points `[...] Point-`, in a colour inside a rim of the wing's."""
    places = marks.reshape(-1)
    return [simplices.markers(places, rgb(wing_colour), radius, camera),
            simplices.markers(places, rgb(colour), radius - rim, camera)]


def draw(ax, points: core.Point, velocity: core.Point, stream: core.Bivector, lift: core.Force, speed: float,
         marks: list = ()) -> None:
    """The pressure coefficient around the wing at the points, the streamlines, the wing, its lift, and the marks
    `[(points, colour)]`."""
    # Each ring closed, its first angle repeated at the end, so no seam is left unpainted.
    def closed(values: Extensor) -> Extensor:
        return concatenate([values, values[:, :1]], axis=1)

    ring_points = closed(points)                                             # [rings, angles + 1] Point-
    speed_squared = closed(velocity.dot(velocity)).to_array()
    # The pressure coefficient: low where the flow is fast.
    pressure = 1 - speed_squared / speed**2
    colours = plt.get_cmap("RdBu")(TwoSlopeNorm(0.0, *pressure_range)(pressure))[..., :3]
    # The stream function's values, its pairings with the unit scalar.
    stream_values = closed(stream & core.mv.scalar([1])).to_array()
    ax.set(xlim=window[:2], ylim=window[2:], xticks=[], yticks=[])
    ax.set_aspect("equal")
    # The rings, numbered ring by ring, a grid around the wing.
    mesh = contours.Mesh.grid(*ring_points.shape)
    places = ring_points.reshape(-1)                                         # [rings * (angles + 1)] Point-
    streamlines = contours.Lines(stream_spacing, 0.0, stream_ink, stream_width)
    camera = flat_panel(ax, core.ga)
    # The wing filled by the triangles of its outline's hull, under the flow, whose rings start at the wing's surface
    # and cover all of the hull outside it.
    outline = points[0]                                                      # [angles] Point-
    # The lift from the origin, as it is, read along the directions x and y: `lift & mv.x.dual()` is `mv.x | lift`, its
    # share along x; a stroke ending in a chevron.
    push = lift * lift_scale                                                 # [] Force+
    forces = panel(ax, core.mv.x.dual(), core.mv.y.dual())
    simplices.overlay(ax, [
        contours.picture(outline, Mesh.triangulate(outline), np.zeros(len(outline)), rgb(wing_colour), camera),
        contours.picture(places, mesh, stream_values.reshape(-1), colours.reshape(-1, 3), camera,
                         lines=(streamlines,)),
        simplices.strokes(stack((0 * push, push)), Mesh.chains([2]), rgb(lift_colour), lift_width, forces,
                          chevrons=np.ones(1), chevron=lift_chevron),
        *[picture for spots, colour in marks
          for picture in marked(spots, colour, mark_radius, mark_rim, camera)]])
    # The panel keeps its frame around the picture.
    ax.set_axis_on()


def figure(points: core.Point, velocity: core.Point, stream: core.Bivector, lift: core.Force, speed: float) -> plt.Figure:
    result = plt.figure(figsize=(8, 4.6), dpi=frame_dpi, layout="constrained")
    draw(result.subplots(), points, velocity, stream, lift, speed)
    return result


def frame(points: core.Point, velocity: core.Point, stream: core.Bivector, lift: core.Force, speed: float,
          critical: core.Point, centre: core.Point) -> np.ndarray:
    """One frame, with the images of the map's critical points and the cylinder's centre marked."""
    result = plt.figure(figsize=(8, 4.6), dpi=frame_dpi, layout="constrained")
    draw(result.subplots(), points, velocity, stream, lift, speed, [(critical, mark_colour), (centre, centre_colour)])
    pixels = capture(result)
    plt.close(result)
    return pixels


def animate_marked(frames: Iterable[tuple[core.Point, core.Point, core.Bivector, core.Force, core.Point, core.Point]],
                   speed: float) -> Iterator[np.ndarray]:
    """Each frame of points, velocity, stream function and lift, with the critical points' images and the
    cylinder's centre marked."""
    for points, velocity, stream, lift, critical, centre in frames:
        yield frame(points, velocity, stream, lift, speed, critical, centre)


# README picture: this makes plots/wing.gif, tuned as a visual for the README. Don't change it; to
# experiment, copy it.
def animate(frames: Iterable[tuple[core.Point, core.Point, core.Bivector, core.Force]], speed: float) -> Iterator[np.ndarray]:
    """Each frame of points, velocity, stream function and lift."""
    for points, velocity, stream, lift in frames:
        yield plain_frame(points, velocity, stream, lift, speed)


def plain_frame(points: core.Point, velocity: core.Point, stream: core.Bivector, lift: core.Force, speed: float) -> np.ndarray:
    result = plt.figure(figsize=(8, 4.6), dpi=frame_dpi, layout="constrained")
    draw(result.subplots(), points, velocity, stream, lift, speed)
    pixels = capture(result)
    plt.close(result)
    return pixels


def bipolar(grid: core.Point, grid_images: core.Point, circle: core.Point, circle_image: core.Point,
            critical: core.Point) -> plt.Figure:
    """The grid of circles through and around the critical points with the wing's circle, before and
    after the map, with the critical points and where they move."""
    result = plt.figure(figsize=(10, 4.2), dpi=map_dpi, layout="constrained")
    for ax, curves, outline, marks in zip(
            result.subplots(1, 2), (grid, grid_images), (circle, circle_image), (critical, 2 * critical)):
        ax.set(xlim=window[:2], ylim=window[2:], xticks=[], yticks=[])
        ax.set_aspect("equal")
        camera = flat_panel(ax, core.ga)
        n_curves, n_samples = curves.shape
        # The marks and their reflections through the origin.
        simplices.overlay(ax, [
            simplices.strokes(curves.reshape(-1), Mesh.chains([n_samples] * n_curves), rgb("#94a3b8"), grid_width,
                              camera),
            simplices.strokes(outline, Mesh.chains([len(outline)]), rgb(wing_colour), outline_width, camera),
            *marked(stack([marks, -marks]), mark_colour, map_mark_radius, map_mark_rim, camera)])
        # The panel keeps its frame around the picture.
        ax.set_axis_on()
    return result
