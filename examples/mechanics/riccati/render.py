"""Docking paths, thruster commands and fixed-heading slices of the remaining cost."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np

from examples.animation import capture
from examples.mechanics.riccati import core

BODY_COLOUR = "#3275a8"
FORCE_COLOUR = "#d16a30"
SITE_COLOUR = "#7b62a3"
GHOST_COUNT = 5
COST_GRID_EXTENT = 0.55
COST_GRID_COUNT = 101
COST_LEVEL = 0.15
COST_HORIZONS = 3
# The point each pose carries: the origin, dual to the weight direction w.
ORIGIN = core.mv.w.dual()


# --- plumbing -------------------------------------------------------------------------
def xy(points: core.Point) -> np.ndarray:
    """Read finite point coordinates by pairing them with the coordinate planes."""
    return np.stack([((plane & points) / (core.mv.w & points)).to_array()
                     for plane in (core.mv.x, core.mv.y)], axis=-1)         # [..., 2] float


def _limits(points: np.ndarray) -> np.ndarray:
    """A square viewport around display coordinates, with room for arrowheads."""
    low, high = points.min(axis=0), points.max(axis=0)                     # [2] float
    center = (low + high) / 2                                             # [2] float
    half_width = np.max(high - low) * 0.58
    return np.stack([center - half_width, center + half_width])           # [corners, 2] float


def _axes(axes: plt.Axes, limits: np.ndarray, title: str) -> None:
    axes.set(xlim=limits[:, 0], ylim=limits[:, 1], aspect="equal", title=title)
    axes.set_axis_off()


def _target(axes: plt.Axes, hull: np.ndarray) -> None:
    axes.add_patch(Polygon(hull, closed=True, fill=False, edgecolor="0.5",
                           linestyle="--", linewidth=1.1, zorder=1))


def draw_setup(
    hull: core.Point, mounts: core.Point, directions: core.Point, tracking_points: core.Point,
) -> plt.Figure:
    """The hull, the points used to measure docking error, and each positive thrust direction."""
    body = xy(hull)                                                       # [vertices, 2] float
    sites = xy(tracking_points)                                                   # [sites, 2] float
    bases = xy(mounts)                                                    # [thrusters, 2] float
    # These arrows show direction; their common display length is tied to the hull's size.
    display_length = np.ptp(body, axis=0).max() * 0.3
    ends = xy(mounts + directions * display_length)                       # [thrusters, 2] float
    arrows = ends - bases                                                # [thrusters, 2] float
    limits = _limits(np.concatenate([body, sites, bases, ends]))          # [corners, 2] float
    figure, axes = plt.subplots(figsize=(4.5, 4), layout="constrained")
    axes.add_patch(Polygon(body, closed=True, facecolor=BODY_COLOUR,
                           edgecolor=BODY_COLOUR, alpha=0.16, linewidth=1.3))
    axes.scatter(*bases.T, marker="s", s=25, color="0.25", label="Thrusters", zorder=4)
    axes.scatter(*sites.T, s=26, color=SITE_COLOUR, label="Tracking points", zorder=5)
    axes.quiver(*bases.T, *arrows.T, angles="xy", scale_units="xy", scale=1,
                color=FORCE_COLOUR, width=0.007, zorder=3)
    _axes(axes, limits, "Docking geometry")
    axes.legend(loc="upper right", frameon=False, fontsize=8)
    return figure


def draw_approaches(hull: core.Point, poses: core.Motor, labels: tuple[str, ...]) -> plt.Figure:
    """The center paths and a few hull positions, with the target pose shown dashed."""
    carried = poses[..., None] >> hull                                   # [times, cases, vertices] Point
    centers = poses >> ORIGIN                                            # [times, cases] Point
    bodies = xy(carried)                                                 # [times, cases, vertices, 2] float
    paths = xy(centers)                                                  # [times, cases, 2] float
    target = xy(hull)                                                    # [vertices, 2] float
    limits = _limits(np.concatenate([bodies.reshape(-1, 2), target]))     # [corners, 2] float
    figure = plt.figure(figsize=(4.6 * len(labels), 4.3), layout="constrained")
    for case, label in enumerate(labels):
        axes = figure.add_subplot(1, len(labels), case + 1)
        _target(axes, target)
        axes.plot(*paths[:, case].T, color=BODY_COLOUR, linewidth=1.6)
        # Equal distances along the path keep the ghost hulls apart near the final pose.
        distance = np.concatenate([[0], np.cumsum(np.linalg.norm(
            np.diff(paths[:, case], axis=0), axis=-1))])                  # [times] float
        selected = np.unique(np.searchsorted(distance, np.linspace(0, distance[-1], GHOST_COUNT)))  # [ghosts] int
        ghosts = bodies[selected, case]                                 # [ghosts, vertices, 2] float
        outlines = np.concatenate([ghosts, ghosts[:, :1]], axis=1)       # [ghosts, vertices + 1, 2] float
        axes.add_collection(LineCollection(outlines, colors=BODY_COLOUR, linewidths=1, alpha=0.35))
        axes.add_patch(Polygon(bodies[-1, case], closed=True, facecolor=BODY_COLOUR,
                               edgecolor=BODY_COLOUR, alpha=0.25, linewidth=1.2))
        axes.plot(*paths[0, case], "o", color=BODY_COLOUR, markersize=3)
        _axes(axes, limits, label)
    return figure


def draw_costs(values: core.StateCost, labels: tuple[str, ...]) -> plt.Figure:
    """One shared cost contour at several horizons, in the zero-heading translation slice."""
    remaining = np.geomspace(1, len(values), COST_HORIZONS).astype(int)   # [remaining] int
    offsets = np.linspace(-COST_GRID_EXTENT, COST_GRID_EXTENT, COST_GRID_COUNT)
    samples = -core.mv.yw * offsets[:, None] - core.mv.xw * offsets[None, :]  # [samples_y, samples_x] Twist
    costs = values[remaining - 1, :, None, None](samples, samples)        # [remaining, cases, samples_y, samples_x] Scalar
    points = (samples * -0.5).exp() >> ORIGIN                             # [samples_y, samples_x] Point
    coordinates = xy(points)                                             # [samples_y, samples_x, 2] float
    cost_values = costs.to_array()                                       # [remaining, cases, samples_y, samples_x] float
    colours = plt.get_cmap("viridis")(np.linspace(0.15, 0.8, len(remaining)))  # [remaining, 4] float
    limits = np.stack([coordinates.min(axis=(0, 1)), coordinates.max(axis=(0, 1))])  # [corners, 2] float
    figure = plt.figure(figsize=(4.6 * len(labels), 4.3), layout="constrained")
    for case, label in enumerate(labels):
        axes = figure.add_subplot(1, len(labels), case + 1)
        # The same contour value makes the accumulated cost comparable at different horizons.
        for index, colour in enumerate(colours):
            axes.contour(coordinates[..., 0], coordinates[..., 1], cost_values[index, case],
                         levels=[COST_LEVEL], colors=[colour], linewidths=1.6)
        _axes(axes, limits, label)
    handles = [Line2D([], [], color=colour, linewidth=1.6, label=str(steps))
               for colour, steps in zip(colours, remaining)]
    figure.legend(handles=handles, loc="outside lower center", ncol=len(remaining),
                  frameon=False, fontsize=9, title="Steps remaining")
    return figure


def print_costs(
    tracking: core.Scalar, effort: core.Scalar, landing: core.Scalar, predicted: core.Scalar,
    labels: tuple[str, ...],
) -> None:
    """Compare the accumulated tracking, weighted effort and landing costs with the predicted total."""
    total = tracking + effort + landing                                 # [cases] Scalar
    values = np.stack([cost.to_array() for cost in (tracking, effort, landing, total, predicted)], axis=-1)   # [cases, columns] float
    columns = ("Tracking cost", "Effort cost", "Landing cost", "Total", "Predicted")
    label_width = max(map(len, labels))
    column_width = max(map(len, columns)) + 1
    print(f"{'Case':<{label_width}}" + "".join(f"{column:>{column_width}}" for column in columns))
    for label, row in zip(labels, values):
        print(f"{label:<{label_width}}" + "".join(f"{value:{column_width}.6f}" for value in row))


def animate(
    hull: core.Point, mounts: core.Point, directions: core.Point, poses: core.Motor, commands: core.Scalar,
    labels: tuple[str, ...], arrow_scale: float,
) -> list[np.ndarray]:
    """Docking motion with signed thruster arrows, a frame for each pose, on fixed limits shared by every
    case."""
    carried = poses[..., None] >> hull                                   # [times, cases, vertices] Point
    carried_mounts = poses[..., None] >> mounts                          # [times, cases, thrusters] Point
    # A signed command reverses its ideal force direction before the pose carries it to the world.
    forces = poses[..., None] >> (directions * commands * arrow_scale)   # [times, cases, thrusters] Point
    ends = xy(carried_mounts + forces)                                   # [times, cases, thrusters, 2] float
    bases = xy(carried_mounts)                                           # [times, cases, thrusters, 2] float
    arrows = ends - bases                                                # [times, cases, thrusters, 2] float
    bodies = xy(carried)                                                 # [times, cases, vertices, 2] float
    trajectories = xy(poses >> ORIGIN)                                   # [times, cases, 2] float
    target = xy(hull)                                                    # [vertices, 2] float
    limits = _limits(np.concatenate([bodies.reshape(-1, 2), bases.reshape(-1, 2),
                                     ends.reshape(-1, 2), trajectories.reshape(-1, 2), target]))
    figure = plt.figure(figsize=(4.6 * len(labels), 4.3), dpi=100, layout="constrained")
    patches, quivers = [], []
    for case, label in enumerate(labels):
        axes = figure.add_subplot(1, len(labels), case + 1)
        _target(axes, target)
        axes.plot(*trajectories[:, case].T, color=BODY_COLOUR, linewidth=1, alpha=0.3)
        body = Polygon(bodies[0, case], closed=True, facecolor=BODY_COLOUR,
                       edgecolor=BODY_COLOUR, alpha=0.25, linewidth=1.3)
        axes.add_patch(body)
        thrust = axes.quiver(*bases[0, case].T, *arrows[0, case].T, angles="xy",
                             scale_units="xy", scale=1, color=FORCE_COLOUR, width=0.007, minlength=0)
        patches.append(body)
        quivers.append(thrust)
        _axes(axes, limits, label)

    figure.canvas.draw()
    figure.set_layout_engine("none")
    images = []
    for index in range(len(poses)):
        for case, (body, thrust) in enumerate(zip(patches, quivers)):
            body.set_xy(bodies[index, case])
            thrust.set_offsets(bases[index, case])
            thrust.set_UVC(*arrows[index, case].T)
        images.append(capture(figure))
    plt.close(figure)
    return images
