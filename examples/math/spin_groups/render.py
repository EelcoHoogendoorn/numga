"""Drawing for the spin groups: the table of signatures, and the orbits of the two isoclinic flows on
the three-sphere, seen through the stereographic projection: strokes and dots drawn on the GPU."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

from numga import stack

from examples.math.spin_groups import core
from examples.util.wgsl import fitted, meshes, show, solids, strokes
from examples.util.wgsl.scene import Scene

sm = solids.mv
background = np.ones(3)
# One view for all three families: the elevation and azimuth in radians, azimuth zero looking along +y, the margin
# around the orbits as a share of the larger side, and the pixels across a panel of a frame.
elevation, azimuth = np.radians(24.0), np.radians(35.0)
margin = 0.14
frame_pixels = 300
# The widths in pixels of the circles and the knots in the still, and of the faint orbits in the animation, and their
# opacity; the riders' radius in pixels, and the count of riders along each knot.
circle_width, knot_width, faint_width, faint_opacity = 1.1, 1.8, 1.1, 0.45
rider_radius = 2.2
n_riders = 6


def coordinates(points: core.Euclidean) -> np.ndarray:
    """The x, y and z coordinates of points in space, `[..., 3]`."""
    mv = points.context.multivector
    return (stack((mv.x, mv.y, mv.z)) & points[..., None]).to_array()


def gpu_points(points: core.Euclidean) -> solids.Point:
    """The GPU library's own points at the places of points in space: their coordinates, read by pairing, are the one
    hand-off from this algebra to the renderer's."""
    return sm.point(coordinates(points))


def colours(shape: tuple[int, int]) -> np.ndarray:
    """A colour per orbit: the start's position around the xy plane as the hue, its height as the value."""
    hue = np.broadcast_to(np.linspace(0.0, 1.0, shape[1], endpoint=False), shape)
    value = np.broadcast_to(np.linspace(0.95, 0.55, shape[0])[:, None], shape)
    return hsv_to_rgb(np.stack([hue, np.full(shape, 0.75), value], axis=-1))


def view(pixels: int, drawn: list[core.Euclidean]) -> solids.View:
    """One view for every panel, `pixels` across, framing everything drawn."""
    return solids.view([gpu_points(points) for points in drawn], 0.0, elevation, azimuth, margin, pixels)


def orbit_samples(orbits: core.Euclidean, width: float, view: solids.View) -> meshes.Samples:
    """Each orbit `[heights, azimuths, steps + 1]` as a stroke of a width in pixels, in the colour of its start, at
    every sample."""
    curves = zip(gpu_points(orbits.reshape((-1, orbits.shape[-1]))), colours(orbits.shape[:2]).reshape(-1, 3))
    return strokes.polylines(list(curves), width, view)


def dots(positions: core.Euclidean, rgb: np.ndarray, view: solids.View) -> tuple[solids.Quadric, np.ndarray]:
    """A dot at each position `[n]`, in its colour `[n, 3]`, as an opaque part."""
    return solids.ball(gpu_points(positions), rider_radius * view.pixel)[:, None], rgb


def draw_flows(left: core.Euclidean, right: core.Euclidean, knotted: core.Euclidean) -> plt.Figure:
    """The orbits of the two isoclinic flows, each a family of circles filling nested tori and linked
    in opposite senses, beside the orbits of a rotation combining them at different rates: trefoil
    knots, one on each torus."""
    drawn = [left, right, knotted]
    figure = plt.figure(figsize=(9, 3.1))
    axes = [figure.add_axes((index / len(drawn), 0.0, 1 / len(drawn), 1.0)) for index in range(len(drawn))]
    # The panels' rooms alike, one view fitted to the first holds them all.
    panel = fitted(view(1, drawn), axes[0])
    for ax, orbits, width in zip(axes, drawn, (circle_width, circle_width, knot_width)):
        show(ax, solids.draw([], panel, background, inks=[], translucent=[],
                             meshes=[(orbit_samples(orbits, width, panel), 1.0)]), panel)
    return figure


def animate_flows(left: core.Euclidean, right: core.Euclidean, knotted: core.Euclidean,
                  n_frames: int) -> Iterator[np.ndarray]:
    """The points of the three-sphere carried along every flow at once, their orbits drawn faintly;
    a few points ride along each knot. The three panels side by side, seen by one camera."""
    drawn = [left, right, knotted]
    frame = view(frame_pixels, drawn)
    # Each panel's faint orbits, set up once.
    panels = [Scene(frame, background, [], [(orbit_samples(orbits, faint_width, frame), faint_opacity)], inks=[],
                    translucent=[])
              for orbits in drawn]
    length = drawn[0].shape[2] - 1
    steps = np.linspace(0, length, n_frames, endpoint=False).astype(int)
    riders = np.arange(n_riders) * length // n_riders
    knots = drawn[2]
    for step in steps:
        # Every start carried along the two flows, and the riders along each knot.
        riding = [(orbits[:, :, step].reshape(-1), colours(orbits.shape[:2]).reshape(-1, 3)) for orbits in drawn[:2]]
        riding.append((knots[:, :, (step + riders) % length].reshape(-1),
                       np.repeat(colours(knots.shape[:2]).reshape(-1, 3), len(riders), axis=0)))
        yield np.concatenate([panel.draw([dots(*points, frame)], inks=[], translucent=[],
                                         meshes=[]) for panel, points in zip(panels, riding)], axis=1)
