"""Draws the docking geometry, remaining-cost contours, approach trajectories, and thruster-command animation."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

from numga import concatenate, stack
from examples.util import charts
from examples.util.animation import capture
from numga.topology import Topology
from numga.render import Camera, contours, drawing, simplices
from examples.util.panels import overlay, panel
from examples.mechanics.riccati import core

body_colour = "#3275a8"
force_colour = "#d16a30"
site_colour = "#7b62a3"
thruster_colour = "0.25"
target_colour = "0.5"
n_ghosts = 5
cost_grid_extent = 0.55
n_samples = 101
cost_level = 0.15
n_horizons = 3
line_width = 1.5
arrow_width, arrow_chevron = 2.0, 8.0
mark_radius = 3.0
target_dashes = (5.5, 2.5)
origin = core.mv.w.dual()                                                  # [] Point-


# --- plumbing -------------------------------------------------------------------------
def point_coordinates(batches: list[core.Point]) -> core.Scalar:
    """Reads planar coordinates (`[2, ...] Scalar+`) across batches of finite points by homogeneous plane pairing."""
    points = concatenate([batch.reshape(-1) for batch in batches])         # [...] Point-
    return (stack((core.mv.x, core.mv.y))[:, None] & points) / (core.mv.w & points)


def _viewport_limits(batches: list[core.Point]) -> np.ndarray:
    """Computes square viewport bounds (`[2, 2]`) enclosing all given point batches with margin for arrow chevrons."""
    coordinates = point_coordinates(batches)                               # [2, ...] Scalar+
    low, high = coordinates.min(axis=-1), coordinates.max(axis=-1)         # [2] Scalar+, [2] Scalar+
    center = (low + high) / 2                                              # [2] Scalar+
    half_width = (high - low).max() * 0.58
    return stack([center - half_width, center + half_width]).to_array()    # [2, 2]


def _panel_axes(axes: plt.Axes, limits: np.ndarray) -> None:
    """Configures equal-aspect viewport bounds with hidden axes."""
    axes.set(xlim=limits[:, 0], ylim=limits[:, 1], aspect="equal")
    axes.set_axis_off()


def _hull_topologies(n_hulls: int, n_corners: int) -> tuple[Topology, Topology]:
    """Builds fan triangulations and boundary outlines for `n_hulls` convex polygons of `n_corners` vertices."""
    corners = np.arange(n_corners)                                         # [n_corners]
    fan = np.stack([np.zeros(n_corners - 2, dtype=int), corners[1:-1], corners[2:]], axis=-1)  # [n_corners - 2, 3]
    surface = Topology(n_hulls * n_corners, (fan + n_corners * np.arange(n_hulls)[:, None, None]).reshape(-1, 3))
    return surface, surface.border


def _arrow_topology(n_arrows: int) -> Topology:
    """Connects `n_arrows` tail vertices to the `n_arrows` tip vertices listed immediately after them."""
    return Topology(2 * n_arrows, np.stack([np.arange(n_arrows), n_arrows + np.arange(n_arrows)], axis=-1))


def _target_outline(camera: Camera, hull: core.Point) -> np.ndarray:
    """Renders the dashed outline of the target hull pose."""
    _, outline = _hull_topologies(1, len(hull))
    return simplices.strokes(hull, outline, to_rgb(target_colour), line_width, camera, dashes=target_dashes,
                             cap=simplices.Cap.BUTT)


@charts.style()
def setup_figure(
    hull: core.Point, mounts: core.Point, directions: core.Point, tracking_points: core.Point,
) -> plt.Figure:
    """Draws the vessel hull, thruster mounts, positive thrust directions, and tracking points."""
    n_vertices = len(hull)
    n_thrusters = len(mounts)
    body = point_coordinates([hull])                                       # [2, n_vertices] Scalar+
    display_length = (body.max(axis=-1) - body.min(axis=-1)).max() * 0.3
    tips = mounts + directions * display_length                            # [n_thrusters] Point-
    limits = _viewport_limits([hull, tracking_points, mounts, tips])       # [2, 2]
    figure, axes = charts.subplots(280, 260, layout="constrained")
    _panel_axes(axes, limits)
    camera = panel(axes, core.mv.x, core.mv.y, core.mv.w)
    surface, _ = _hull_topologies(1, n_vertices)
    overlay(axes, [
        contours.picture(hull, surface, np.zeros(n_vertices), to_rgb(body_colour), camera, opacity=0.16),
        simplices.strokes(stack([mounts, tips]).reshape(-1), _arrow_topology(n_thrusters), to_rgb(force_colour),
                          arrow_width, camera, chevrons=np.ones(n_thrusters), chevron=arrow_chevron),
        simplices.markers(mounts, to_rgb(thruster_colour), mark_radius, camera),
        simplices.markers(tracking_points, to_rgb(site_colour), mark_radius, camera)])
    return figure


@charts.style()
def approaches_figure(hull: core.Point, poses: core.Motor) -> plt.Figure:
    """Draws the centre trajectories, arc-length-sampled hull outlines, and dashed target pose across cases."""
    n_poses, n_cases = poses.shape
    n_vertices = len(hull)
    posed_hulls = poses[..., None] >> hull                                 # [n_poses, n_cases, n_vertices] Point-
    centers = poses >> origin                                              # [n_poses, n_cases] Point-
    limits = _viewport_limits([posed_hulls, hull])                         # [2, 2]
    hull_surface, _ = _hull_topologies(1, n_vertices)
    figure, panels = charts.subplots(280 * n_cases, 260, ncols=n_cases, squeeze=False, layout="constrained")
    for axes in panels[0]:
        _panel_axes(axes, limits)
    for case, axes in enumerate(panels[0]):
        camera = panel(axes, core.mv.x, core.mv.y, core.mv.w)
        # Sampling at equal arc length keeps ghost hulls distinct near the target pose.
        step_lengths = (centers[:-1, case] & centers[1:, case]).norm().to_array()  # [n_poses - 1]
        distances = np.concatenate([[0], np.cumsum(step_lengths)])         # [n_poses]
        ghost_indices = np.unique(np.searchsorted(distances, np.linspace(0, distances[-1], n_ghosts)))
        _, ghost_outlines = _hull_topologies(len(ghost_indices), n_vertices)
        ghosts = posed_hulls[ghost_indices, case].reshape(-1)              # [...] Point-
        overlay(axes, [
            _target_outline(camera, hull),
            contours.picture(posed_hulls[-1, case], hull_surface, np.zeros(n_vertices), to_rgb(body_colour), camera,
                             opacity=0.25),
            simplices.strokes(ghosts, ghost_outlines, to_rgb(body_colour), line_width, camera, opacity=0.35),
            simplices.strokes(centers[:, case], Topology.chains([n_poses]), to_rgb(body_colour),
                              line_width, camera),
            simplices.markers(centers[0, case][None], to_rgb(body_colour), mark_radius, camera)])
    return figure


@charts.style()
def costs_figure(values: core.StateCost) -> plt.Figure:
    """Draws one shared remaining-cost contour across logarithmically spaced horizons in the zero-heading slice."""
    n_values, n_cases = values.shape
    horizon_steps = np.geomspace(1, n_values, n_horizons).astype(int)      # [n_horizons]
    offsets = np.linspace(-cost_grid_extent, cost_grid_extent, n_samples)  # [n_samples]
    samples = -core.mv.yw * offsets[:, None] - core.mv.xw * offsets[None, :]  # [n_samples, n_samples] Twist+
    costs = values[horizon_steps - 1, :, None, None](samples, samples)     # [n_horizons, n_cases, n_samples, n_samples] Scalar+
    sample_motors = (samples * -0.5).exp()                                 # [n_samples, n_samples] Motor+
    points = sample_motors >> origin                                       # [n_samples, n_samples] Point-
    coordinates = point_coordinates([points])                              # [2, ...] Scalar+
    grid = Topology.grid(n_samples, n_samples)
    colours = plt.get_cmap("viridis")(np.linspace(0.15, 0.8, n_horizons))  # [n_horizons, 4]
    limits = stack([coordinates.min(axis=-1), coordinates.max(axis=-1)]).to_array()  # [2, 2]
    figure, panels = charts.subplots(280 * n_cases, 260, ncols=n_cases, squeeze=False, layout="constrained")
    for axes in panels[0]:
        _panel_axes(axes, limits)
    for case, axes in enumerate(panels[0]):
        camera = panel(axes, core.mv.x, core.mv.y, core.mv.w)
        overlay(axes, [
            contours.picture(points.reshape(-1), grid, costs[index, case].to_array().reshape(-1), np.zeros(3), camera,
                             levels=(contours.Level(cost_level, colour, line_width),), opacity=0.0)
            for index, colour in enumerate(colours)])
    return figure


def cost_summary(
    tracking_cost: core.Scalar, effort_cost: core.Scalar, landing_cost: core.Scalar, predicted_cost: core.Scalar,
    labels: tuple[str, ...],
) -> None:
    """Prints the realized tracking, effort, and landing costs alongside their sum and Bellman prediction."""
    total_cost = tracking_cost + effort_cost + landing_cost                # [...] Scalar+
    table = np.stack(
        [cost.to_array() for cost in (tracking_cost, effort_cost, landing_cost, total_cost, predicted_cost)],
        axis=-1,
    )                                                                      # [..., 5]
    columns = ("Tracking cost", "Effort cost", "Landing cost", "Total", "Predicted")
    label_width = max(map(len, labels))
    column_width = max(map(len, columns)) + 1
    print(f"{'Case':<{label_width}}" + "".join(f"{column:>{column_width}}" for column in columns))
    for label, row in zip(labels, table):
        print(f"{label:<{label_width}}" + "".join(f"{value:{column_width}.6f}" for value in row))


def docking_frames(
    hull: core.Point, mounts: core.Point, directions: core.Point, poses: core.Motor, commands: core.Scalar,
    arrow_scale: float,
) -> Iterator[np.ndarray]:
    """Yields frames of the docking motion and signed thruster force arrows on shared viewport bounds."""
    n_poses, n_cases = poses.shape
    n_vertices = len(hull)
    n_thrusters = len(mounts)
    posed_hulls = poses[..., None] >> hull                                 # [n_poses, n_cases, n_vertices] Point-
    posed_mounts = poses[..., None] >> mounts                              # [n_poses, n_cases, n_thrusters] Point-
    # Scaling each body-fixed thrust direction by its signed command reverses negative pushes before posing.
    forces = poses[..., None] >> (directions * commands * arrow_scale)     # [n_poses, n_cases, n_thrusters] Point-
    tips = posed_mounts + forces                                           # [n_poses, n_cases, n_thrusters] Point-
    trajectories = poses >> origin                                         # [n_poses, n_cases] Point-
    limits = _viewport_limits([posed_hulls, posed_mounts, tips, trajectories, hull])
    hull_surface, _ = _hull_topologies(1, n_vertices)
    thrust_arrows = _arrow_topology(n_thrusters)
    with charts.style():
        figure, panels = charts.subplots(280 * n_cases, 260, ncols=n_cases, squeeze=False, layout="constrained")
        for axes in panels[0]:
            _panel_axes(axes, limits)
        figure.canvas.draw()
        figure.set_layout_engine("none")
    cameras = [panel(axes, core.mv.x, core.mv.y, core.mv.w) for axes in panels[0]]
    guides = [[_target_outline(camera, hull),
               simplices.strokes(trajectories[:, case], Topology.chains([n_poses]), to_rgb(body_colour),
                                 line_width, camera, opacity=0.3)]
              for case, camera in enumerate(cameras)]
    images = [overlay(axes, layers) for axes, layers in zip(panels[0], guides)]
    for index in range(n_poses):
        for case, (image, camera) in enumerate(zip(images, cameras)):
            image.set_data(drawing.composited([
                *guides[case],
                contours.picture(posed_hulls[index, case], hull_surface, np.zeros(n_vertices), to_rgb(body_colour),
                                 camera, opacity=0.25),
                simplices.strokes(stack([posed_mounts[index, case], tips[index, case]]).reshape(-1), thrust_arrows,
                                  to_rgb(force_colour), arrow_width, camera, chevrons=np.ones(n_thrusters),
                                  chevron=arrow_chevron),
                simplices.markers(posed_mounts[index, case], to_rgb(thruster_colour), mark_radius, camera)]))
        yield capture(figure)
    plt.close(figure)
