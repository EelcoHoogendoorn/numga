"""Drawing and coordinate read-out for epipolar geometry and two-view reconstruction.

Constructs viewport geometry only; the mathematics in core never imports this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np

from examples.geometry.epipolar.core import Line, Motor, Point, ga, mv, point

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def euclidean(points: Point) -> np.ndarray:
    """Read xyz coordinates using an explicit coordinate basis, independent of storage order."""
    k = points.cast(ga.subspace("yzw zxw xyw zyx")).kernel
    return k[..., :3] / k[..., 3:]


def screen_line_endpoints(lines: Line, half_width: float) -> np.ndarray:
    """Clip lines on the screen z = 1 to x = ±half_width, returning (..., 2, 2) screen coordinates."""
    ends = [euclidean(lines ^ (mv.x - mv.w * x))[..., :2] for x in (-half_width, half_width)]
    return np.stack(ends, axis=-2)


def draw_camera_frustum(
    ax: Axes,
    pose: Motor,
    scale: float,
    color: str,
    linestyle: str,
    linewidth: float,
    label: str,
) -> None:
    """Draw a 3D camera wireframe pyramid: its centre, sensor rectangle and optical axis."""
    # Camera local sensor corners at distance z = scale, and the optical axis tip, moved by the pose:
    w, h = 0.5 * scale, 0.38 * scale
    local = np.array([
        [-w, -h, scale],
        [ w, -h, scale],
        [ w,  h, scale],
        [-w,  h, scale],
        [0.0, 0.0, scale * 1.3],
        [0.0, 0.0, 0.0],
    ])
    placed = euclidean(pose >> point(local))
    corners, axis_end, center = placed[:4], placed[4], placed[5]

    # Sensor rectangle:
    loop = np.concatenate([corners, corners[:1]], axis=0)
    ax.plot(
        loop[:, 0], loop[:, 1], loop[:, 2],
        color=color, linestyle=linestyle, linewidth=linewidth, label=label,
    )

    # Rays from center to corners:
    for corner in corners:
        segment = np.stack([center, corner], axis=0)
        ax.plot(
            segment[:, 0], segment[:, 1], segment[:, 2],
            color=color, linestyle=linestyle, linewidth=linewidth * 0.8,
        )

    # Optical axis:
    ax.plot(
        [center[0], axis_end[0]], [center[1], axis_end[1]], [center[2], axis_end[2]],
        color=color, linestyle=linestyle, linewidth=linewidth * 1.2,
    )

    # Camera center dot:
    ax.scatter([center[0]], [center[1]], [center[2]], color=color, s=45, depthshade=False)


def draw_camera_view(
    ax: Axes,
    image: Point,
    measured: Point,
    title: str,
    point_colors: np.ndarray,
) -> None:
    """Draw a 2D camera sensor view with true and measured keypoints."""
    true_uv = euclidean(image)[:, :2]
    measured_uv = euclidean(measured)[:, :2]
    ax.scatter(
        true_uv[:, 0], true_uv[:, 1],
        color="#94a3b8", s=30, alpha=0.6, marker="o", label="True projection",
    )
    ax.scatter(measured_uv[:, 0], measured_uv[:, 1], c=point_colors, s=40, edgecolors="#1e293b", linewidths=0.6)
    ax.scatter([], [], color="#3b82f6", s=40, label="Measured keypoint (with noise)")

    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Sensor u", fontsize=8, labelpad=-2)
    ax.set_ylabel("Sensor v", fontsize=8, labelpad=-2)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.8)


def draw_epipolar_lines(ax: Axes, lines: Line, point_colors: np.ndarray) -> None:
    """Draw epipolar lines on a camera's screen, one per keypoint."""
    for (a, b), color in zip(screen_line_endpoints(lines, 0.6), point_colors):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, alpha=0.35, linewidth=1.0)


def draw_3d_reconstruction(
    ax: Axes,
    landmarks: Point,
    reconstructed: Point,
    true_motor: Motor,
    est_motor: Motor,
    title: str,
    point_colors: np.ndarray,
) -> None:
    """Draw the 3D world scene overlaying true vs estimated cameras and landmarks."""
    points_true = euclidean(landmarks)
    points_reconstructed = euclidean(reconstructed)
    ax.scatter(
        points_true[:, 0], points_true[:, 1], points_true[:, 2],
        color="#94a3b8", s=35, alpha=0.5, label="Ground truth 3D landmarks",
    )
    ax.scatter(
        points_reconstructed[:, 0], points_reconstructed[:, 1], points_reconstructed[:, 2],
        c=point_colors, marker="*", s=90, depthshade=False,
    )
    ax.scatter([], [], color="#10b981", marker="*", s=90, label="Triangulated 3D points")

    # Camera frustums:
    draw_camera_frustum(ax, mv.rotor(), 0.4, "#0284c7", "-", 1.5, "Camera 1 (World origin)")
    draw_camera_frustum(ax, true_motor, 0.4, "#94a3b8", "--", 1.2, "Camera 2 Ground Truth")
    draw_camera_frustum(ax, est_motor, 0.4, "#10b981", "-", 1.6, "Camera 2 Reconstructed")

    # Sight rays for a subset of landmarks to show intersection:
    c1 = euclidean(mv.zyx)
    c2 = euclidean(est_motor >> mv.zyx)
    for target in points_reconstructed[::5]:
        ray1_seg = np.stack([c1, target], axis=0)
        ray2_seg = np.stack([c2, target], axis=0)
        ax.plot(ray1_seg[:, 0], ray1_seg[:, 1], ray1_seg[:, 2], color="#0284c7", alpha=0.3, linewidth=0.8, linestyle=":")
        ax.plot(ray2_seg[:, 0], ray2_seg[:, 1], ray2_seg[:, 2], color="#10b981", alpha=0.3, linewidth=0.8, linestyle=":")

    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(-0.2, 4.8)
    ax.set_box_aspect([1.8, 1.5, 3.5])
    ax.set_xlabel("X (m)", fontsize=8, labelpad=-5)
    ax.set_ylabel("Y (m)", fontsize=8, labelpad=-5)
    ax.set_zlabel("Z (depth)", fontsize=8, labelpad=-5)
    ax.view_init(elev=18, azim=-55)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.8)


def draw_epipolar(
    landmarks: Point,
    image_1: Point,
    image_2: Point,
    noisy_1: Point,
    noisy_2: Point,
    epipolar_lines_2: Line,
    reconstructed: Point,
    true_motor: Motor,
    est_motor: Motor,
) -> plt.Figure:
    """Both camera images, epipolar lines on the second, and the 3D reconstruction."""
    # One colour per landmark, matching keypoints across the views and the reconstruction:
    point_colors = plt.colormaps["turbo"].resampled(landmarks.shape[0])(np.arange(landmarks.shape[0]))
    fig = plt.figure(figsize=(19.0, 5.8), dpi=140, layout="constrained")

    draw_camera_view(
        fig.add_subplot(1, 3, 1), image_1, noisy_1,
        f"1. Camera 1 Image ({landmarks.shape[0]} Keypoints)\nForward projection + Gaussian pixel noise",
        point_colors,
    )
    ax2 = fig.add_subplot(1, 3, 2)
    draw_epipolar_lines(ax2, epipolar_lines_2, point_colors)
    draw_camera_view(
        ax2, image_2, noisy_2,
        "2. Camera 2 Image (Epipolar Constraint)\nKeypoints lie along 1D epipolar search lines",
        point_colors,
    )
    draw_3d_reconstruction(
        fig.add_subplot(1, 3, 3, projection="3d"), landmarks, reconstructed, true_motor, est_motor,
        "3. 3D World Reconstruction (PGA3D Motor & Line Quadrics)",
        point_colors,
    )
    return fig
