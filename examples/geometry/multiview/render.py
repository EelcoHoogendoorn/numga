"""Top-down drawing and convergence animation for multi-camera reconstruction in PGA2D.

Sight cones and fused splats are rasterized by evaluating each quadric on a grid of plane
points, `quadric(grid) & grid`.
"""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Ellipse
import numpy as np

from numga.algebras import PGA2D

from examples import instantiate
from examples.animation import capture

# The figures are top-down views of the plane: they draw PGA2D geometry.
core = instantiate("examples.geometry.multiview.core", PGA2D)
Motor, Point, Quadric, Information = core.Motor, core.Point, core.Quadric, core.Information
mv, point = core.mv, core.point

CAMERA_COLORS = ["#0284c7", "#ec4899", "#8b5cf6"]
SPLAT_COLOR = np.array([0.98, 0.48, 0.04])
X_RANGE = (-1.25, 1.25)
Y_RANGE = (-0.30, 2.95)
RESOLUTION = (750, 750)
# Angular half-width of a pixel, the cone radius per unit depth.
PIXEL_ANGLE = 0.035         # rad


# --- read-out -----------------------------------------------------------------------------
def coordinates(points: Point) -> np.ndarray:
    """Cartesian (x, y) of points, read in an explicit blade layout."""
    k = points.cast(core.ga.subspace("yw wx xy")).kernel
    return k[..., :2] / k[..., 2:]


def optical_axes(motors: Motor) -> np.ndarray:
    """Unit viewing directions (x, y): the normals of the cameras' focal lines."""
    normals = (motors >> mv.y).cast(core.ga.subspace("x y")).kernel
    return normals / np.linalg.norm(normals, axis=-1, keepdims=True)


def pose_covariance(information: Information) -> np.ndarray:
    """Pose covariance over twist coefficients (yw, wx, xy): the pseudoinverse of the curvature."""
    basis = mv("yw wx xy", np.eye(3))
    gram = information[..., None, None](basis[:, None], basis[None, :]).to_array()
    return np.linalg.pinv(gram, rcond=1e-4)


def pose_sigmas(information: Information) -> np.ndarray:
    """One-sigma pose uncertainties [..., 3] over twist coefficients, from the curvature form."""
    covariance = pose_covariance(information)
    return np.sqrt(np.maximum(np.diagonal(covariance, axis1=-2, axis2=-1), 0.0))


# --- rasterized quadrics ------------------------------------------------------------------
def plane_grid() -> tuple[Point, float]:
    """The viewport as a grid of plane points, and the width of one pixel."""
    xs = np.linspace(*X_RANGE, RESOLUTION[1])
    ys = np.linspace(*Y_RANGE, RESOLUTION[0])
    xx, yy = np.meshgrid(xs, ys)
    return point(np.stack([xx, yy], axis=-1)), (X_RANGE[1] - X_RANGE[0]) / RESOLUTION[1]


def depths(motors: Motor, grid: Point) -> np.ndarray:
    """[n_cams, h, w] depth of each grid point in front of each camera's focal line."""
    return (((motors >> mv.y)[:, None, None]) & grid).to_array()


def coverage(quadric_val: np.ndarray, level_set: np.ndarray, pixel_w: float) -> np.ndarray:
    """Anti-aliased inside-ness of the level set np.sqrt(quadric_val) <= level_set."""
    dist = level_set - np.sqrt(np.maximum(quadric_val, 0.0))
    return 1.0 / (1.0 + np.exp(np.clip(-dist / (0.75 * pixel_w), -30.0, 30.0)))


def blend(image: np.ndarray, coverages: np.ndarray, color: np.ndarray, alpha: float) -> np.ndarray:
    """Composite a stack of same-coloured coverages over the image, one after another."""
    keep = np.prod(1.0 - alpha * coverages, axis=0)
    return image * keep[..., None] + color * (1.0 - keep)[..., None]


def blend_cones(image: np.ndarray, grid: Point, pixel_w: float, motors: Motor, world_cones: Quadric) -> np.ndarray:
    """Each camera's sight cones, as wide as a pixel at each depth, in the camera's colour."""
    depth = depths(motors, grid)
    for c, color in zip(range(motors.shape[0]), CAMERA_COLORS):
        values = (world_cones[:, c][:, None, None](grid) & grid).to_array()      # [n_points, h, w]
        covered = coverage(values, PIXEL_ANGLE * np.maximum(depth[c], 0.05), pixel_w) * (depth[c] > 0.02)
        image = blend(image, covered, np.array(mcolors.to_rgb(color)), 0.30)
    return image


def blend_splats(image: np.ndarray, grid: Point, pixel_w: float, motors: Motor, fused: Quadric, points: Point) -> np.ndarray:
    """The fused quadrics around their points, at least a pixel wide at the points' depth."""
    in_front = np.any(depths(motors, grid) > 0.02, axis=0)
    floor = np.maximum((fused(points) & points).to_array(), 0.0)                  # [n_points]
    depth = ((motors >> mv.y)[None, :] & points[:, None]).to_array().mean(axis=1)  # [n_points]
    radius = np.sqrt((PIXEL_ANGLE * np.maximum(depth, 0.2)) ** 2 + floor)
    values = (fused[:, None, None](grid) & grid).to_array() - floor[:, None, None]
    covered = coverage(values, radius[:, None, None], pixel_w) * in_front
    return blend(image, covered, SPLAT_COLOR, 0.95)


def show_image(ax: plt.Axes, image: np.ndarray) -> None:
    ax.imshow(np.clip(image, 0.0, 1.0), extent=[*X_RANGE, *Y_RANGE], origin="lower")


# --- overlays -----------------------------------------------------------------------------
def plot_cameras(ax: plt.Axes, motors: Motor) -> None:
    """Each camera's field-of-view wedge, sensor line and optical axis."""
    centers = coordinates(motors >> point(np.zeros(2)))
    axes = optical_axes(motors)
    scale, half_fov = 0.28, np.radians(38.0)
    for center, axis, color in zip(centers, axes, CAMERA_COLORS):
        perp = np.array([-axis[1], axis[0]])
        p_left = center + axis * scale - perp * (scale * np.tan(half_fov))
        p_right = center + axis * scale + perp * (scale * np.tan(half_fov))
        tip = center + axis * (scale * 1.15)

        # Shaded FOV wedge, boundary rays, sensor line and dashed optical axis:
        triangle = np.stack([center, p_left, p_right], axis=0)
        ax.fill(triangle[:, 0], triangle[:, 1], color=color, alpha=0.18, zorder=3)
        ax.plot([center[0], p_left[0]], [center[1], p_left[1]], color=color, linewidth=1.1, alpha=0.6, zorder=3)
        ax.plot([center[0], p_right[0]], [center[1], p_right[1]], color=color, linewidth=1.1, alpha=0.6, zorder=3)
        ax.plot([p_left[0], p_right[0]], [p_left[1], p_right[1]], color=color, linewidth=2.0, zorder=4)
        ax.plot([center[0], tip[0]], [center[1], tip[1]], color=color, linewidth=1.2, linestyle="--", zorder=4)


def plot_pose_covariance(ax: plt.Axes, motors: Motor, information: Information) -> None:
    """Each observed camera's position covariance ellipse and orientation fan, from its information form.

    Anchored cameras carry no information and draw nothing.
    """
    covariance = pose_covariance(information)
    observed = np.abs(covariance).max(axis=(-2, -1)) > 0
    centers = coordinates(motors >> point(np.zeros(2)))[observed]
    axes = optical_axes(motors)[observed]
    colors = np.array(CAMERA_COLORS[:len(observed)])[observed]
    for center, axis, cov_twist, color in zip(centers, axes, covariance[observed], colors):
        # Local translation covariance: a twist with coefficients (yw, wx, xy) displaces the camera
        # by minus its wx coefficient along local x and by its yw coefficient along local y. Rotated
        # into the world by the camera's frame:
        cov_local = np.array([
            [ cov_twist[1, 1], -cov_twist[1, 0]],
            [-cov_twist[0, 1],  cov_twist[0, 0]],
        ])
        frame = np.column_stack([[axis[1], -axis[0]], axis])
        evals, evecs = np.linalg.eigh(frame @ cov_local @ frame.T)
        # One-sigma radii, scaled for display and kept inside the axes:
        radii = np.minimum(np.sqrt(np.maximum(evals, 1e-8)) * 0.08, 1.0)
        ax.add_patch(Ellipse(
            xy=center, width=2 * radii[0], height=2 * radii[1],
            angle=np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0])),
            edgecolor=color, facecolor=color, alpha=0.28, linewidth=1.6, linestyle="--", zorder=6,
        ))

        # Angular uncertainty fan (orientation standard deviation on xy generator):
        rot_std_deg = min(float(np.degrees(np.sqrt(max(cov_twist[2, 2], 0.0))) * 0.35), 180.0)
        arc_r = 0.38
        heading = float(np.degrees(np.arctan2(axis[1], axis[0])))
        ax.add_patch(Arc(
            xy=center, width=2 * arc_r, height=2 * arc_r, angle=0,
            theta1=heading - rot_std_deg, theta2=heading + rot_std_deg,
            color=color, linewidth=1.3, linestyle=":", zorder=6,
        ))
        for sign in (-1, 1):
            th = np.radians(heading + sign * rot_std_deg)
            p_end = center + arc_r * np.array([np.cos(th), np.sin(th)])
            ax.plot([center[0], p_end[0]], [center[1], p_end[1]], color=color, linestyle=":", linewidth=1.1, zorder=6)


def frame_axes(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(*X_RANGE)
    ax.set_ylim(*Y_RANGE)
    ax.axis("off")


def plot_reconstruction(ax: plt.Axes, motors: Motor, world_cones: Quadric, fused: Quadric, points: Point) -> None:
    """Sight cones, fused splats, reconstructed points and cameras."""
    ax.clear()
    grid, pixel_w = plane_grid()
    image = blend_cones(np.ones((*RESOLUTION, 3)), grid, pixel_w, motors, world_cones)
    show_image(ax, blend_splats(image, grid, pixel_w, motors, fused, points))
    plot_cameras(ax, motors)
    frame_axes(ax)


# --- figures ------------------------------------------------------------------------------
def draw_rig(motors: Motor) -> plt.Figure:
    """The cameras."""
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=140, layout="constrained")
    plot_cameras(ax, motors)
    frame_axes(ax)
    return fig


def draw_cones(motors: Motor, world_cones: Quadric) -> plt.Figure:
    """The cameras and their sight cones through the scene points."""
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=140, layout="constrained")
    grid, pixel_w = plane_grid()
    show_image(ax, blend_cones(np.ones((*RESOLUTION, 3)), grid, pixel_w, motors, world_cones))
    plot_cameras(ax, motors)
    frame_axes(ax)
    return fig


def draw_reconstruction(motors: Motor, world_cones: Quadric, fused: Quadric, points: Point) -> plt.Figure:
    """Sight cones fused into splats around the reconstructed points."""
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=75, layout="constrained")
    plot_reconstruction(ax, motors, world_cones, fused, points)
    return fig


def draw_reconstruction_with_covariance(
    motors: Motor, world_cones: Quadric, fused: Quadric, points: Point, information: Information,
) -> plt.Figure:
    """The reconstruction with each camera's pose covariance."""
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=140, layout="constrained")
    plot_reconstruction(ax, motors, world_cones, fused, points)
    plot_pose_covariance(ax, motors, information)
    return fig


def animate_convergence(states: Iterable) -> list[np.ndarray]:
    """RGB frames of the reconstruction, one per state."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=120, layout="constrained")
    frames = []
    for motors, world_cones, fused, points in states:
        plot_reconstruction(ax, motors, world_cones, fused, points)
        frames.append(capture(fig))
    plt.close(fig)
    return frames


def animate_convergence_with_covariance(
    states: Iterable,
) -> list[np.ndarray]:
    """RGB frames of the reconstruction and pose covariances, one per state."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=120, layout="constrained")
    frames = []
    for motors, world_cones, fused, points, information in states:
        plot_reconstruction(ax, motors, world_cones, fused, points)
        plot_pose_covariance(ax, motors, information)
        frames.append(capture(fig))
    plt.close(fig)
    return frames
