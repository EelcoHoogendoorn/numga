"""Drawing for the articulated robot, ray traced on the GPU: its capsules and spheres on level ground; and a chart
of how fast a batch of robots steps."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from examples.mechanics.articulated import core
from examples.util import charts
from examples.util.wgsl import image_figure, solids

mv, origin = solids.mv, solids.origin
part_colour = np.array([0.30, 0.45, 0.72])
ground_colour = np.array([0.86, 0.85, 0.82])
background = np.ones(3)
# The ground under the robot, a round slab whose top is the plane through the origin: its radius and thickness, in
# metres.
ground_radius, ground_thickness = 1.3, 0.04
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the pixels across the picture; and the
# space the robot falls through, which it frames, in metres: rings about the vertical through the origin, each by its
# radius and height, the ground's rim and a ring above it as high as the robot is dropped from.
elevation, azimuth = np.radians([18.0, 30.0])
n_pixels = 320
around = np.linspace(0, 2 * np.pi, 24, endpoint=False)
reach_rings = [(ground_radius, 0.0), (ground_radius, -ground_thickness), (0.4, 2.1)]
# The throughput chart's pixels, its curve's colour, and the real-time line's.
chart_px = (420, 300)
throughput_colour, real_time_colour = "tab:blue", "0.55"


# --- plumbing -------------------------------------------------------------------------
def picture(centres: solids.Point, radii: np.ndarray, n_capsules: int) -> np.ndarray:
    """The robots' parts on the ground, from the centres of their balls `[robots, balls]` and the balls' radii
    `[balls]`: the first n_capsules balls the capsules' starts, the next their ends, each pair a rod with balls at its
    ends, and the rest spheres."""
    # The rings the robot falls within, a point on each turned about the vertical.
    on_rings = mv.point([[radius, 0.0, height] for radius, height in reach_rings])   # [rings] Point
    reach = (mv.xy * (around / 2)).exp()[:, None] >> on_rings[None, :]          # [around, rings] Point
    view = solids.view([reach], 0.0, elevation, azimuth, 0.04, n_pixels)
    ground = solids.rods(mv.point([[0.0, 0.0, -ground_thickness]]), origin[None], ground_radius)   # [1, 3] Quadric
    rods = solids.rods(centres[..., :n_capsules], centres[..., n_capsules:2 * n_capsules], radii[:n_capsules])
    balls = solids.ball(centres, radii)                                         # [robots, balls] Quadric
    return solids.draw([(rods.reshape((-1, 3)), part_colour), (balls.reshape(-1)[:, None], part_colour),
                        (ground, ground_colour)], view, background, [], [], [])


def pose(centres: core.Point, radii: np.ndarray, n_capsules: int) -> plt.Figure:
    """One robot, from the centres of its balls `[balls]`, as `picture` draws it."""
    return image_figure(picture(centres.materialize(mv.context)[None], radii, n_capsules))


def animate(centres: Iterable[core.Point], radii: np.ndarray, n_capsules: int) -> Iterator[np.ndarray]:
    """One frame for each placement of the robots' balls, their centres `[robots, balls]`, as `picture` draws it."""
    for placed in centres:
        yield picture(placed.materialize(mv.context), radii, n_capsules)


@charts.style()
def throughput(counts: np.ndarray, steps_per_second: np.ndarray, real_time: float) -> plt.Figure:
    """Robot steps a second against the robots in a batch, `[counts]` each, log against log, with the line where every
    robot runs in real time, `real_time` steps a second each."""
    figure, axes = charts.subplots(*chart_px)
    axes.loglog(counts, counts * real_time, color=real_time_colour, linestyle="--")
    axes.loglog(counts, steps_per_second, color=throughput_colour, marker="o")
    axes.text(counts[1], counts[1] * real_time, " real time", color=real_time_colour, ha="left", va="top")
    axes.set_xlabel("humanoids")
    axes.set_ylabel("humanoid steps per second")
    axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure
