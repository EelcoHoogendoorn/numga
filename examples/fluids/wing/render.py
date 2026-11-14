"""Pressure, streamlines, and lift around a Joukowski wing."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm, to_rgb, to_rgba

from numga import stack

from examples.util import shapes
from examples.util.animation import capture
from numga.topology import Topology
from numga.render import Camera, contours, simplices
from examples.util.panels import offset_panel, overlay
from examples.fluids.wing import core

window = (-3.2, 3.2, -1.8, 1.8)
pressure_range = (-3.0, 1.0)
pressure_map = plt.get_cmap("RdBu")
# Flat pressure bands centred on evenly spaced levels with white at free-stream pressure.
pressure_step = 0.25
pressure_levels = np.arange(pressure_range[0], pressure_range[1] + pressure_step / 2, pressure_step)
pressure_bands = contours.Bands(pressure_range[0] - pressure_step / 2, pressure_step,
                                pressure_map(TwoSlopeNorm(0.0, *pressure_range)(pressure_levels)))
# Streamlines spaced evenly around zero, with stroke ink and pixel width.
stream_spacing = 0.3
stream_ink = np.array(to_rgba("#334155"))
stream_width = 0.5
lift_scale = 0.35
wing_colour = "#0f172a"
lift_colour = "#16a34a"
mark_colour = "#facc15"
centre_colour = "#f8fafc"
# Frame resolution giving a 400-pixel-wide gallery animation.
frame_dpi = 50
# Stroke widths, chevron size, and marker radii in pixels at `frame_dpi`.
lift_width, lift_chevron = 1.7, 6.8
mark_radius, mark_rim = 2.2, 0.56
# Resolution and stroke sizes for the two-panel Joukowski map figure.
map_dpi = 40
grid_width, outline_width, map_mark_radius, map_mark_rim = 0.5, 1.2, 1.9, 0.56


# --- plumbing -------------------------------------------------------------------------
def rgb(colour: str) -> np.ndarray:
    """Red, green, and blue channels of a named colour."""
    return np.array(to_rgb(colour))


def rimmed_markers(marks: core.Point, colour: str, radius: float, rim: float, camera: Camera) -> list[np.ndarray]:
    """Circular markers at `marks` filled with `colour` inside a dark rim."""
    places = marks.reshape(-1)                                               # [...] Point-
    return [simplices.markers(places, rgb(wing_colour), radius, camera),
            simplices.markers(places, rgb(colour), radius - rim, camera)]


def draw(ax, points: core.Point, velocity: core.Point, stream: core.Bivector, lift: core.Force, speed: float,
         marks: list = (), bands: tuple[contours.Bands, ...] = ()) -> None:
    """Draw the pressure coefficient, streamlines, wing body, lift arrow, and optional markers on `ax`."""
    speed_squared = velocity.squared().to_array()
    # Bernoulli pressure coefficient, negative where the flow accelerates over the upper surface.
    pressure = 1 - speed_squared / speed**2
    colours = pressure_map(TwoSlopeNorm(0.0, *pressure_range)(pressure))[..., :3]
    # Scalar stream-function values paired against the intrinsic unit antiscalar.
    stream_values = (stream & core.mv.xy.dual()).to_array()
    ax.set(xlim=window[:2], ylim=window[2:], xticks=[], yticks=[])
    ax.set_aspect("equal")
    n_rings, n_angles = points.shape
    topology = Topology.loops([n_angles]).product(Topology.chains([n_rings]))
    places = points.reshape(-1)                                              # [V] Point-
    streamlines = contours.Lines(stream_spacing, 0.0, stream_ink, stream_width)
    camera = offset_panel(ax, core.mv.x, core.mv.y)
    # Fill the wing interior with the Delaunay triangulation of its surface loop beneath the outer flow rings.
    outline = points[0]                                                      # [n_angles] Point-
    hull, _ = shapes.delaunay((stack((core.mv.x, core.mv.y)) & outline.batch()[..., None]).to_array())
    # Read the extrinsic force directly by pairing with the unit x and y displacements.
    push = lift * lift_scale                                                 # [] Force+
    forces = offset_panel(ax, core.mv.x.dual(), core.mv.y.dual())
    overlay(ax, [
        contours.picture(outline, hull, np.zeros(n_angles), rgb(wing_colour), camera),
        contours.picture(places, topology, pressure.reshape(-1), colours.reshape(-1, 3), camera, bands=bands),
        contours.picture(places, topology, stream_values.reshape(-1), np.ones(3), camera, lines=(streamlines,),
                         opacity=0.0),
        simplices.strokes(stack((0 * push, push)), Topology.chains([2]), rgb(lift_colour), lift_width, forces,
                          chevrons=np.ones(1), chevron=lift_chevron),
        *[picture for spots, colour in marks
          for picture in rimmed_markers(spots, colour, mark_radius, mark_rim, camera)]])
    ax.set_axis_on()


def figure(points: core.Point, velocity: core.Point, stream: core.Bivector, lift: core.Force, speed: float) -> plt.Figure:
    """Single-panel figure of the steady pressure field, streamlines, and lift around the wing."""
    result = plt.figure(figsize=(8, 4.6), dpi=frame_dpi, layout="constrained")
    draw(result.subplots(), points, velocity, stream, lift, speed)
    return result


def marked_frame(points: core.Point, velocity: core.Point, stream: core.Bivector, lift: core.Force, speed: float,
                 critical_images: core.Point, centre: core.Point) -> np.ndarray:
    """Rendered RGB frame with the critical-point images and cylinder centre marked."""
    result = plt.figure(figsize=(8, 4.6), dpi=frame_dpi, layout="constrained")
    draw(result.subplots(), points, velocity, stream, lift, speed,
         [(critical_images, mark_colour), (centre, centre_colour)])
    pixels = capture(result)
    plt.close(result)
    return pixels


def animate_marked(speed: float,
                   frames: Iterable[tuple[core.Point, core.Point, core.Bivector, core.Force, core.Point, core.Point]],
                   ) -> Iterator[np.ndarray]:
    """Yield RGB frames of the morphing flow with critical-point images and cylinder centre marked."""
    for points, velocity, stream, lift, critical_images, centre in frames:
        yield marked_frame(points, velocity, stream, lift, speed, critical_images, centre)


def animate(speed: float, frames: Iterable[tuple[core.Point, core.Point, core.Bivector, core.Force]]) -> Iterator[np.ndarray]:
    """Yield RGB frames of the pitching wing with flat pressure bands."""
    for points, velocity, stream, lift in frames:
        yield plain_frame(points, velocity, stream, lift, speed)


def plain_frame(points: core.Point, velocity: core.Point, stream: core.Bivector, lift: core.Force, speed: float) -> np.ndarray:
    """Rendered RGB frame of the pitching wing with flat pressure bands."""
    result = plt.figure(figsize=(8, 4.6), dpi=frame_dpi, layout="constrained")
    draw(result.subplots(), points, velocity, stream, lift, speed, bands=(pressure_bands,))
    pixels = capture(result)
    plt.close(result)
    return pixels


def bipolar(grid: core.Point, grid_images: core.Point, circle: core.Point, circle_image: core.Point,
            critical: core.Point) -> plt.Figure:
    """Two-panel figure of bipolar circle families and the cylinder circle before and after the Joukowski map."""
    result = plt.figure(figsize=(10, 4.2), dpi=map_dpi, layout="constrained")
    for ax, curves, outline, marks in zip(
            result.subplots(1, 2), (grid, grid_images), (circle, circle_image), (critical, 2 * critical)):
        ax.set(xlim=window[:2], ylim=window[2:], xticks=[], yticks=[])
        ax.set_aspect("equal")
        camera = offset_panel(ax, core.mv.x, core.mv.y)
        n_curves, n_samples = curves.shape
        overlay(ax, [
            simplices.strokes(curves.reshape(-1), Topology.loops([n_samples] * n_curves), rgb("#94a3b8"), grid_width,
                              camera),
            simplices.strokes(outline, Topology.loops([len(outline)]), rgb(wing_colour), outline_width, camera),
            *rimmed_markers(stack([marks, -marks]), mark_colour, map_mark_radius, map_mark_rim, camera)])
        ax.set_axis_on()
    return result
