"""GPU ray-traced rendering draws the articulated robot's capsules and spheres above a circular ground slab alongside a log-log batch throughput chart."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from examples.mechanics.articulated import core
from examples.util import charts
from examples.util.panels import image_figure
from numga.render import solids
from numga.sparse import Sparse

mv, origin = solids.mv, solids.origin
part_colour = np.array([0.30, 0.45, 0.72])
ground_colour = np.array([0.86, 0.85, 0.82])
background = np.ones(3)
ground_radius, ground_thickness = 1.3, 0.04
elevation, azimuth = np.radians([18.0, 30.0])
n_pixels = 320
n_azimuth_samples = 24
azimuth_angles = np.linspace(0, 2 * np.pi, n_azimuth_samples, endpoint=False)
reach_rings = [(ground_radius, 0.0), (ground_radius, -ground_thickness), (0.4, 2.1)]
chart_px = (420, 300)
throughput_colour, real_time_colour = "tab:blue", "0.55"


# --- plumbing -------------------------------------------------------------------------
def picture(centres: solids.Point, radii: np.ndarray, n_capsules: int) -> np.ndarray:
    """Ray-traced RGB image depicts the ground slab and the robots' capsules and spheres from ball `centres` (`[n_robots, n_balls] Point-`) and `radii` (`[n_balls]`)."""
    ring_points = mv.point([[radius, 0.0, height] for radius, height in reach_rings])   # [3] Point-
    azimuth_turns = (mv.xy * (azimuth_angles / 2)).exp()                                # [n_azimuth_samples] Motor+
    reach = azimuth_turns[:, None] >> ring_points[None, :]                              # [n_azimuth_samples, 3] Point-
    view = solids.view([reach], 0.0, elevation, azimuth, 0.04, n_pixels)
    ground_slab = solids.rods(mv.point([[0.0, 0.0, -ground_thickness]]), origin[None], ground_radius)   # [1, 3] Plane+ <- Point-
    rods = solids.rods(centres[..., :n_capsules], centres[..., n_capsules : 2 * n_capsules], radii[:n_capsules])
    balls = solids.ball(centres, radii)                                                 # [...] Plane+ <- Point-
    return solids.draw(
        [
            (rods.reshape((-1, 3)), part_colour),
            (balls.reshape(-1)[:, None], part_colour),
            (ground_slab, ground_colour),
        ],
        view,
        background,
        [],
        [],
        [],
    )


def pose(centres: core.Point, radii: np.ndarray, n_capsules: int) -> plt.Figure:
    """Single-robot figure renders the static home or initial pose from ball `centres` (`[n_balls] Point-`)."""
    scene_centres = solids.context.include_from_algebra(centres)[None]                  # [1, ...] Point-
    return image_figure(picture(scene_centres, radii, n_capsules))


def animate_states(
    forward_kinematics_fn: Callable[[core.Motor, core.Scalar], core.Motor],
    balls: Sparse,
    centres: core.Point,
    radii: np.ndarray,
    n_capsules: int,
    states: Iterable[core.State],
) -> Iterator[np.ndarray]:
    """Yields one ray-traced RGB frame per robot state in `states` after carrying home-pose ball `centres` to their world positions."""
    for motor, _, angles, _ in states:
        ball_motors = (balls * forward_kinematics_fn(motor, angles)).batch()            # [...] Motor+
        placed_centres = ball_motors >> centres                                         # [...] Point-
        yield picture(solids.context.include_from_algebra(placed_centres), radii, n_capsules)


@charts.style()
def throughput(counts: np.ndarray, steps_per_second: np.ndarray, real_time: float) -> plt.Figure:
    """Log-log figure plots `steps_per_second` against batch size `counts` alongside the `real_time` dashed reference line."""
    figure, axes = charts.subplots(*chart_px)
    axes.loglog(counts, counts * real_time, color=real_time_colour, linestyle="--")
    axes.loglog(counts, steps_per_second, color=throughput_colour, marker="o")
    axes.text(counts[1], counts[1] * real_time, " real time", color=real_time_colour, ha="left", va="top")
    axes.set_xlabel("humanoids")
    axes.set_ylabel("humanoid steps per second")
    axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    return figure
