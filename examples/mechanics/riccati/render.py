"""Docking paths, thruster commands and fixed-heading slices of the remaining cost, drawn on the GPU in the plane's own
algebra."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import numpy as np

from numga import concatenate, stack
from examples.util import charts
from examples.util.animation import capture
from examples.util.wgsl import contours, simplices
from examples.util.wgsl.camera import Camera, panel
from examples.mechanics.riccati import core

BODY_COLOUR = "#3275a8"
FORCE_COLOUR = "#d16a30"
SITE_COLOUR = "#7b62a3"
THRUSTER_COLOUR = "0.25"
TARGET_COLOUR = "0.5"
GHOST_COUNT = 5
COST_GRID_EXTENT = 0.55
COST_GRID_COUNT = 101
COST_LEVEL = 0.15
COST_HORIZONS = 3
# Widths, lengths and radii in pixels: of the lines, of the arrows and the strokes of the chevrons at their tips, and
# of the marks; the target's dashes, a dash and a gap.
LINE_WIDTH = 1.5
ARROW_WIDTH, ARROW_CHEVRON = 2.0, 8.0
MARK_RADIUS = 3.0
TARGET_DASHES = (5.5, 2.5)
# The point each pose carries: the origin, dual to the weight direction w.
origin = core.mv.w.dual()


# --- plumbing -------------------------------------------------------------------------
def readings(batches: list[core.Point]) -> core.Scalar:
    """Batches of finite points read across and up, `[2, points]`: their pairings with the coordinate planes over their
    weight."""
    points = concatenate([batch.reshape(-1) for batch in batches])         # [points] Point-
    return (stack((core.mv.x, core.mv.y))[:, None] & points) / (core.mv.w & points)


def _limits(points: list[core.Point]) -> np.ndarray:
    """A square viewport around batches of finite points, with room for the chevrons at the arrows' tips."""
    drawn = readings(points)                                              # [2, points] Scalar+
    low, high = drawn.min(axis=-1), drawn.max(axis=-1)                    # [2] Scalar+, [2] Scalar+
    center = (low + high) / 2                                             # [2] Scalar+
    half_width = (high - low).max() * 0.58
    return stack([center - half_width, center + half_width]).to_array()   # [corners, 2] float


def _axes(axes: plt.Axes, limits: np.ndarray, title: str) -> None:
    axes.set(xlim=limits[:, 0], ylim=limits[:, 1], aspect="equal")
    axes.set_title(title)
    axes.set_axis_off()


def _hulls(n_hulls: int, n_corners: int) -> tuple[simplices.Topology, simplices.Topology]:
    """Convex hulls of `n_corners` corners each, numbered hull after hull: triangles fanning out from each hull's
    first corner, and their outlines, the fans' boundary edges."""
    corners = np.arange(n_corners)
    fan = np.stack([np.zeros(n_corners - 2, dtype=int), corners[1:-1], corners[2:]], axis=-1)   # [triangles, 3] int
    surface = simplices.Topology(n_hulls * n_corners, (fan + n_corners * np.arange(n_hulls)[:, None, None]).reshape(-1, 3))
    return surface, surface.border


def _arrows(n_arrows: int) -> simplices.Topology:
    """Arrows from tails to tips listed after them: an edge from each tail to its tip."""
    return simplices.Topology(2 * n_arrows, np.stack([np.arange(n_arrows), n_arrows + np.arange(n_arrows)], axis=-1))


def _target(camera: Camera, hull: core.Point) -> np.ndarray:
    """The target pose's outline, dashed."""
    _, outline = _hulls(1, len(hull))
    return simplices.strokes(hull, outline, to_rgb(TARGET_COLOUR), LINE_WIDTH, camera, dashes=TARGET_DASHES,
                             cap=simplices.Cap.BUTT)


@charts.style()
def draw_setup(
    hull: core.Point, mounts: core.Point, directions: core.Point, tracking_points: core.Point,
) -> plt.Figure:
    """The hull, the points used to measure docking error, and each positive thrust direction."""
    body = readings([hull])                                               # [2, vertices] Scalar+
    # These arrows show direction; their common display length is tied to the hull's size.
    display_length = (body.max(axis=-1) - body.min(axis=-1)).max() * 0.3
    tips = mounts + directions * display_length                           # [thrusters] Point-
    limits = _limits([hull, tracking_points, mounts, tips])               # [corners, 2] float
    # A third of the column: one hull, five marks and three arrows, under a title and a legend.
    figure, axes = charts.subplots(280, 270, layout="constrained")
    axes.set(xlim=limits[:, 0], ylim=limits[:, 1], aspect="equal", title="Docking geometry")
    axes.set_axis_off()
    handles = [Line2D([], [], marker="o", linestyle="none", color=THRUSTER_COLOUR, label="Thrusters"),
               Line2D([], [], marker="o", linestyle="none", color=SITE_COLOUR, label="Tracking points")]
    axes.legend(handles=handles, loc="upper right", frameon=False)
    camera = panel(axes, core.mv.x, core.mv.y)
    surface, _ = _hulls(1, len(hull))
    simplices.overlay(axes, [
        contours.picture(hull, surface, np.zeros(len(hull)), to_rgb(BODY_COLOUR), camera, opacity=0.16),
        simplices.strokes(stack([mounts, tips]).reshape(-1), _arrows(len(mounts)), to_rgb(FORCE_COLOUR), ARROW_WIDTH,
                          camera, chevrons=np.ones(len(mounts)), chevron=ARROW_CHEVRON),
        simplices.markers(mounts, to_rgb(THRUSTER_COLOUR), MARK_RADIUS, camera),
        simplices.markers(tracking_points, to_rgb(SITE_COLOUR), MARK_RADIUS, camera)])
    return figure


@charts.style()
def draw_approaches(hull: core.Point, poses: core.Motor, labels: tuple[str, ...]) -> plt.Figure:
    """The center paths and a few hull positions, with the target pose shown dashed."""
    carried = poses[..., None] >> hull                                   # [times, cases, vertices] Point-
    centers = poses >> origin                                            # [times, cases] Point-
    limits = _limits([carried, hull])                                     # [corners, 2] float
    final, _ = _hulls(1, len(hull))
    # A panel a case as large as the docking geometry's, under its title.
    figure, panels = charts.subplots(280 * len(labels), 260, ncols=len(labels), squeeze=False, layout="constrained")
    for axes, label in zip(panels[0], labels):
        _axes(axes, limits, label)
    for case, axes in enumerate(panels[0]):
        camera = panel(axes, core.mv.x, core.mv.y)
        # Equal distances along the path keep the ghost hulls apart near the final pose: each step's length, the
        # size of the line through the centres before and after it.
        steps = (centers[:-1, case] & centers[1:, case]).norm().to_array()   # [times - 1] float
        distance = np.concatenate([[0], np.cumsum(steps)])                 # [times] float
        selected = np.unique(np.searchsorted(distance, np.linspace(0, distance[-1], GHOST_COUNT)))  # [ghosts] int
        _, ghost_outlines = _hulls(len(selected), len(hull))
        ghosts = carried[selected, case].reshape(-1)                     # [ghosts * vertices] Point-
        simplices.overlay(axes, [
            _target(camera, hull),
            contours.picture(carried[-1, case], final, np.zeros(len(hull)), to_rgb(BODY_COLOUR), camera,
                             opacity=0.25),
            simplices.strokes(ghosts, ghost_outlines, to_rgb(BODY_COLOUR), LINE_WIDTH, camera, opacity=0.35),
            simplices.strokes(centers[:, case], simplices.Topology.chains([len(centers)]), to_rgb(BODY_COLOUR),
                              LINE_WIDTH, camera),
            simplices.markers(centers[0, case][None], to_rgb(BODY_COLOUR), MARK_RADIUS, camera)])
    return figure


@charts.style()
def draw_costs(values: core.StateCost, labels: tuple[str, ...]) -> plt.Figure:
    """One shared cost contour at several horizons, in the zero-heading translation slice."""
    remaining = np.geomspace(1, len(values), COST_HORIZONS).astype(int)   # [remaining] int
    offsets = np.linspace(-COST_GRID_EXTENT, COST_GRID_EXTENT, COST_GRID_COUNT)
    samples = -core.mv.yw * offsets[:, None] - core.mv.xw * offsets[None, :]  # [samples_y, samples_x] Twist+
    costs = values[remaining - 1, :, None, None](samples, samples)        # [remaining, cases, samples_y, samples_x] Scalar+
    points = (samples * -0.5).exp() >> origin                             # [samples_y, samples_x] Point-
    drawn = readings([points])                                           # [2, samples_y * samples_x] Scalar+
    grid = simplices.Topology.grid(COST_GRID_COUNT, COST_GRID_COUNT)
    colours = plt.get_cmap("viridis")(np.linspace(0.15, 0.8, len(remaining)))  # [remaining, 4] float
    limits = stack([drawn.min(axis=-1), drawn.max(axis=-1)]).to_array()   # [corners, 2] float
    # A panel a case as large as the docking geometry's, under its title, over one legend.
    figure, panels = charts.subplots(280 * len(labels), 310, ncols=len(labels), squeeze=False, layout="constrained")
    handles = [Line2D([], [], color=colour, label=str(steps)) for colour, steps in zip(colours, remaining)]
    figure.legend(handles=handles, loc="outside lower center", ncol=len(remaining), frameon=False,
                  title="Steps remaining")
    for axes, label in zip(panels[0], labels):
        _axes(axes, limits, label)
    for case, axes in enumerate(panels[0]):
        camera = panel(axes, core.mv.x, core.mv.y)
        # The same contour value makes the accumulated cost comparable at different horizons.
        simplices.overlay(axes, [
            contours.picture(points.reshape(-1), grid, costs[index, case].to_array().reshape(-1), np.zeros(3), camera,
                             levels=(contours.Level(COST_LEVEL, colour, LINE_WIDTH),), opacity=0.0)
            for index, colour in enumerate(colours)])
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
    tips = carried_mounts + forces                                       # [times, cases, thrusters] Point-
    trajectories = poses >> origin                                       # [times, cases] Point-
    limits = _limits([carried, carried_mounts, tips, trajectories, hull])
    surface, _ = _hulls(1, len(hull))
    thrust = _arrows(len(mounts))
    # A panel a case as large as the docking geometry's, under its title; the figure laid out once under the charts'
    # style, and only the bodies and the arrows moved between frames.
    with charts.style():
        figure, panels = charts.subplots(280 * len(labels), 260, ncols=len(labels), squeeze=False,
                                         layout="constrained")
        for axes, label in zip(panels[0], labels):
            _axes(axes, limits, label)
        figure.canvas.draw()
        figure.set_layout_engine("none")
    cameras = [panel(axes, core.mv.x, core.mv.y) for axes in panels[0]]
    guides = [[_target(camera, hull),
               simplices.strokes(trajectories[:, case], simplices.Topology.chains([len(trajectories)]), to_rgb(BODY_COLOUR),
                                 LINE_WIDTH, camera, opacity=0.3)]
              for case, camera in enumerate(cameras)]
    images = [simplices.overlay(axes, layers) for axes, layers in zip(panels[0], guides)]
    for index in range(len(poses)):
        for case, (image, camera) in enumerate(zip(images, cameras)):
            image.set_data(simplices.composited([
                *guides[case],
                contours.picture(carried[index, case], surface, np.zeros(len(hull)), to_rgb(BODY_COLOUR), camera,
                                 opacity=0.25),
                simplices.strokes(stack([carried_mounts[index, case], tips[index, case]]).reshape(-1), thrust,
                                  to_rgb(FORCE_COLOUR), ARROW_WIDTH, camera, chevrons=np.ones(len(mounts)),
                                  chevron=ARROW_CHEVRON),
                # Each thruster a point of its own, so that it shows whatever it pushes.
                simplices.markers(carried_mounts[index, case], to_rgb(THRUSTER_COLOUR), MARK_RADIUS, camera)]))
        yield capture(figure)
    plt.close(figure)
