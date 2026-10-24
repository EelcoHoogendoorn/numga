"""Drawing for the spin groups: the table of signatures, and the orbits of the two isoclinic flows on
the three-sphere, seen through the stereographic projection: strokes and dots drawn on the GPU."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

from examples.math.spin_groups import core
from examples.util.wgsl import fitted, meshes, show, solids, strokes
from examples.util.wgsl.scene import Scene

sm = solids.mv
BACKGROUND = np.ones(3)
# One view for all three families: the elevation and azimuth in radians, azimuth zero looking along +y, the margin
# around the orbits as a share of the larger side, and the pixels across a panel of a frame.
ELEVATION, AZIMUTH = np.radians(24.0), np.radians(35.0)
MARGIN = 0.14
FRAME_PIXELS = 440
# The widths in pixels of the circles and the knots in the still, and of the faint orbits in the animation, and their
# opacity; the riders' radius in pixels.
CIRCLE_WIDTH, KNOT_WIDTH, FAINT_WIDTH, FAINT_OPACITY = 2.0, 3.2, 1.6, 0.45
RIDER_RADIUS = 3.2


def components(points: core.Extensor) -> np.ndarray:
    """The x, y and z components of points in space."""
    return points.cast(points.algebra.subspace("x y z")).kernel


def colours(shape: tuple[int, int]) -> np.ndarray:
    """A colour per orbit: the start's position around the xy plane as the hue, its height as the value."""
    hue = np.broadcast_to(np.linspace(0.0, 1.0, shape[1], endpoint=False), shape)
    value = np.broadcast_to(np.linspace(0.95, 0.55, shape[0])[:, None], shape)
    return hsv_to_rgb(np.stack([hue, np.full(shape, 0.75), value], axis=-1))


def view(pixels: int, *drawn: np.ndarray) -> solids.View:
    """One view for every panel, `pixels` across, framing everything drawn `[..., 3]`."""
    return solids.view([sm.point(points) for points in drawn], 0.0, ELEVATION, AZIMUTH, MARGIN, pixels)


def orbit_samples(orbits: np.ndarray, width: float, view: solids.View) -> meshes.Samples:
    """Each orbit `[heights, azimuths, steps + 1, 3]` as a stroke of a width in pixels, in the colour of its start, at
    every sample."""
    curves = zip(sm.point(orbits.reshape(-1, *orbits.shape[-2:])), colours(orbits.shape[:2]).reshape(-1, 3))
    return strokes.polylines(list(curves), width, view)


def dots(positions: np.ndarray, rgb: np.ndarray, view: solids.View) -> tuple[solids.Quadric, np.ndarray]:
    """A dot at each position `[n, 3]`, in its colour `[n, 3]`, as an opaque part."""
    return solids.ball(sm.point(positions), RIDER_RADIUS * view.pixel)[:, None], rgb


def draw_flows(left: core.Extensor, right: core.Extensor, knotted: core.Extensor) -> plt.Figure:
    """The orbits of the two isoclinic flows, each a family of circles filling nested tori and linked
    in opposite senses, beside the orbits of a rotation combining them at different rates: trefoil
    knots, one on each torus."""
    drawn = [components(orbits) for orbits in (left, right, knotted)]
    figure = plt.figure(figsize=(16, 5.5))
    axes = [figure.add_axes((index / len(drawn), 0.0, 1 / len(drawn), 1.0)) for index in range(len(drawn))]
    # The panels' rooms alike, one view fitted to the first holds them all.
    panel = fitted(view(1, *drawn), axes[0])
    for ax, orbits, width in zip(axes, drawn, (CIRCLE_WIDTH, CIRCLE_WIDTH, KNOT_WIDTH)):
        show(ax, solids.draw([], panel, BACKGROUND, meshes=[(orbit_samples(orbits, width, panel), 1.0)]), panel)
    return figure


def animate_flows(left: core.Extensor, right: core.Extensor, knotted: core.Extensor,
                  frames: int) -> Iterator[np.ndarray]:
    """The points of the three-sphere carried along every flow at once, their orbits drawn faintly;
    six points ride along each knot. The three panels side by side, seen by one camera."""
    drawn = [components(orbits) for orbits in (left, right, knotted)]
    frame = view(FRAME_PIXELS, *drawn)
    # Each panel's faint orbits, set up once.
    panels = [Scene(frame, BACKGROUND, [], [(orbit_samples(orbits, FAINT_WIDTH, frame), FAINT_OPACITY)])
              for orbits in drawn]
    length = drawn[0].shape[2] - 1
    steps = np.linspace(0, length, frames, endpoint=False).astype(int)
    riders = np.arange(6) * length // 6
    knots = drawn[2]
    for step in steps:
        # Every start carried along the two flows, and six points along each knot.
        riding = [(orbits[:, :, step].reshape(-1, 3), colours(orbits.shape[:2]).reshape(-1, 3)) for orbits in drawn[:2]]
        riding.append((knots[:, :, (step + riders) % length].reshape(-1, 3),
                       np.repeat(colours(knots.shape[:2]).reshape(-1, 3), len(riders), axis=0)))
        yield np.concatenate([panel.draw([dots(*points, frame)]) for panel, points in zip(panels, riding)], axis=1)
