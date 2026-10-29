"""Docking paths, thruster commands and fixed-heading slices of the remaining cost."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np

from numga import stack
from examples.util import charts
from examples.util.animation import capture
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
origin = core.mv.w.dual()


# --- plumbing -------------------------------------------------------------------------
def xy(points: core.Point) -> np.ndarray:
    """Coordinates of finite points, by their pairings with the coordinate planes over their weight."""
    return ((stack((core.mv.x, core.mv.y)) & points[..., None]) / (core.mv.w & points)[..., None]).to_array()   # [..., 2] float


def along(directions: core.Point) -> np.ndarray:
    """Components of directions, points at infinity, by their pairings with the coordinate planes."""
    return (stack((core.mv.x, core.mv.y)) & directions[..., None]).to_array()   # [..., 2] float


def _limits(points: np.ndarray) -> np.ndarray:
    """A square viewport around display coordinates, with room for arrowheads."""
    low, high = points.min(axis=0), points.max(axis=0)                     # [2] float
    center = (low + high) / 2                                             # [2] float
    half_width = np.max(high - low) * 0.58
    return np.stack([center - half_width, center + half_width])           # [corners, 2] float


def _axes(axes: plt.Axes, limits: np.ndarray, title: str) -> None:
    axes.set(xlim=limits[:, 0], ylim=limits[:, 1], aspect="equal")
    axes.set_title(title)
    axes.set_axis_off()


def _target(axes: plt.Axes, hull: np.ndarray) -> None:
    axes.add_patch(Polygon(hull, closed=True, fill=False, edgecolor="0.5",
                           linestyle="--", zorder=1))


@charts.style()
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
    arrows = along(directions * display_length)                          # [thrusters, 2] float
    limits = _limits(np.concatenate([body, sites, bases, ends]))          # [corners, 2] float
    # A third of the column: one hull, five marks and three arrows, under a title and a legend.
    figure, axes = charts.subplots(280, 270, layout="constrained")
    axes.add_patch(Polygon(body, closed=True, facecolor=BODY_COLOUR, edgecolor=BODY_COLOUR, alpha=0.16))
    axes.scatter(*bases.T, marker="s", color="0.25", label="Thrusters", zorder=4)
    axes.scatter(*sites.T, color=SITE_COLOUR, label="Tracking points", zorder=5)
    axes.quiver(*bases.T, *arrows.T, angles="xy", scale_units="xy", scale=1,
                color=FORCE_COLOUR, width=0.007, zorder=3)
    axes.set(xlim=limits[:, 0], ylim=limits[:, 1], aspect="equal", title="Docking geometry")
    axes.set_axis_off()
    axes.legend(loc="upper right", frameon=False)
    return figure


@charts.style()
def draw_approaches(hull: core.Point, poses: core.Motor, labels: tuple[str, ...]) -> plt.Figure:
    """The center paths and a few hull positions, with the target pose shown dashed."""
    carried = poses[..., None] >> hull                                   # [times, cases, vertices] Point-
    centers = poses >> origin                                            # [times, cases] Point-
    bodies = xy(carried)                                                 # [times, cases, vertices, 2] float
    paths = xy(centers)                                                  # [times, cases, 2] float
    target = xy(hull)                                                    # [vertices, 2] float
    limits = _limits(np.concatenate([bodies.reshape(-1, 2), target]))     # [corners, 2] float
    # A panel a case as large as the docking geometry's, under its title.
    figure, panels = charts.subplots(280 * len(labels), 260, ncols=len(labels), squeeze=False, layout="constrained")
    for case, (axes, label) in enumerate(zip(panels[0], labels)):
        _target(axes, target)
        axes.plot(*paths[:, case].T, color=BODY_COLOUR)
        # Equal distances along the path keep the ghost hulls apart near the final pose: each step's length, the
        # size of the line through the centres before and after it.
        steps = (centers[:-1, case] & centers[1:, case]).norm().to_array()   # [times - 1] float
        distance = np.concatenate([[0], np.cumsum(steps)])                 # [times] float
        selected = np.unique(np.searchsorted(distance, np.linspace(0, distance[-1], GHOST_COUNT)))  # [ghosts] int
        ghosts = bodies[selected, case]                                 # [ghosts, vertices, 2] float
        outlines = np.concatenate([ghosts, ghosts[:, :1]], axis=1)       # [ghosts, vertices + 1, 2] float
        axes.add_collection(LineCollection(outlines, colors=BODY_COLOUR, alpha=0.35))
        axes.add_patch(Polygon(bodies[-1, case], closed=True, facecolor=BODY_COLOUR, edgecolor=BODY_COLOUR, alpha=0.25))
        axes.plot(*paths[0, case], "o", color=BODY_COLOUR)
        _axes(axes, limits, label)
    return figure


@charts.style()
def draw_costs(values: core.StateCost, labels: tuple[str, ...]) -> plt.Figure:
    """One shared cost contour at several horizons, in the zero-heading translation slice."""
    remaining = np.geomspace(1, len(values), COST_HORIZONS).astype(int)   # [remaining] int
    offsets = np.linspace(-COST_GRID_EXTENT, COST_GRID_EXTENT, COST_GRID_COUNT)
    samples = -core.mv.yw * offsets[:, None] - core.mv.xw * offsets[None, :]  # [samples_y, samples_x] Twist+
    costs = values[remaining - 1, :, None, None](samples, samples)        # [remaining, cases, samples_y, samples_x] Scalar+
    points = (samples * -0.5).exp() >> origin                             # [samples_y, samples_x] Point-
    coordinates = xy(points)                                             # [samples_y, samples_x, 2] float
    cost_values = costs.to_array()                                       # [remaining, cases, samples_y, samples_x] float
    colours = plt.get_cmap("viridis")(np.linspace(0.15, 0.8, len(remaining)))  # [remaining, 4] float
    limits = np.stack([coordinates.min(axis=(0, 1)), coordinates.max(axis=(0, 1))])  # [corners, 2] float
    # A panel a case as large as the docking geometry's, under its title, over one legend.
    figure, panels = charts.subplots(280 * len(labels), 310, ncols=len(labels), squeeze=False, layout="constrained")
    for case, (axes, label) in enumerate(zip(panels[0], labels)):
        # The same contour value makes the accumulated cost comparable at different horizons.
        for index, colour in enumerate(colours):
            axes.contour(coordinates[..., 0], coordinates[..., 1], cost_values[index, case],
                         levels=[COST_LEVEL], colors=[colour])
        _axes(axes, limits, label)
    handles = [Line2D([], [], color=colour, label=str(steps)) for colour, steps in zip(colours, remaining)]
    figure.legend(handles=handles, loc="outside lower center", ncol=len(remaining), frameon=False,
                  title="Steps remaining")
    return figure


def print_costs(
    tracking: core.Scalar, effort: core.Scalar, landing: core.Scalar, predicted: core.Scalar,
    labels: tuple[str, ...],
) -> None:
    """Compare the accumulated tracking, weighted effort and landing costs with the predicted total."""
    total = tracking + effort + landing                                 # [cases] Scalar+
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
) -> Iterator[np.ndarray]:
    """Docking motion with signed thruster arrows, a frame for each pose, on fixed limits shared by every
    case."""
    carried = poses[..., None] >> hull                                   # [times, cases, vertices] Point-
    carried_mounts = poses[..., None] >> mounts                          # [times, cases, thrusters] Point-
    # A signed command reverses its ideal force direction before the pose carries it to the world.
    forces = poses[..., None] >> (directions * commands * arrow_scale)   # [times, cases, thrusters] Point-
    ends = xy(carried_mounts + forces)                                   # [times, cases, thrusters, 2] float
    bases = xy(carried_mounts)                                           # [times, cases, thrusters, 2] float
    arrows = along(forces)                                               # [times, cases, thrusters, 2] float
    bodies = xy(carried)                                                 # [times, cases, vertices, 2] float
    trajectories = xy(poses >> origin)                                   # [times, cases, 2] float
    target = xy(hull)                                                    # [vertices, 2] float
    limits = _limits(np.concatenate([bodies.reshape(-1, 2), bases.reshape(-1, 2),
                                     ends.reshape(-1, 2), trajectories.reshape(-1, 2), target]))
    # A panel a case as large as the docking geometry's, under its title; the figure laid out once under the charts'
    # style, and only the bodies and the arrows moved between frames.
    with charts.style():
        figure, panels = charts.subplots(280 * len(labels), 260, ncols=len(labels), squeeze=False,
                                         layout="constrained")
        patches, quivers = [], []
        for case, (axes, label) in enumerate(zip(panels[0], labels)):
            _target(axes, target)
            axes.plot(*trajectories[:, case].T, color=BODY_COLOUR, alpha=0.3)
            body = Polygon(bodies[0, case], closed=True, facecolor=BODY_COLOUR, edgecolor=BODY_COLOUR, alpha=0.25)
            axes.add_patch(body)
            thrust = axes.quiver(*bases[0, case].T, *arrows[0, case].T, angles="xy",
                                 scale_units="xy", scale=1, color=FORCE_COLOUR, width=0.007, minlength=0)
            patches.append(body)
            quivers.append(thrust)
            _axes(axes, limits, label)
        figure.canvas.draw()
        figure.set_layout_engine("none")
    for index in range(len(poses)):
        for case, (body, thrust) in enumerate(zip(patches, quivers)):
            body.set_xy(bodies[index, case])
            thrust.set_offsets(bases[index, case])
            thrust.set_UVC(*arrows[index, case].T)
        yield capture(figure)
    plt.close(figure)
