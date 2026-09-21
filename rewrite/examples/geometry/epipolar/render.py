"""Drawing and visual helpers for epipolar geometry and two-view reconstruction.

Constructs viewport geometry only; the mathematics in core never imports this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def draw_camera_frustum(
    ax: Axes,
    center: np.ndarray,
    rotation: np.ndarray,
    scale: float,
    color: str,
    linestyle: str,
    linewidth: float,
    label: str,
) -> None:
    """Draw a 3D camera wireframe pyramid representing center and sensor plane."""
    # Camera local sensor corners at distance z = scale:
    w, h = 0.5 * scale, 0.38 * scale
    corners_local = np.array([
        [-w, -h, scale],
        [ w, -h, scale],
        [ w,  h, scale],
        [-w,  h, scale],
    ])
    # In Camera coordinate frame, X_cam = R * (X_world - center)
    # Thus X_world = center + R.T * X_cam:
    corners_world = center + (rotation.T @ corners_local.T).T

    # Sensor rectangle:
    loop = np.concatenate([corners_world, corners_world[:1]], axis=0)
    ax.plot(
        loop[:, 0], loop[:, 1], loop[:, 2],
        color=color, linestyle=linestyle, linewidth=linewidth, label=label,
    )

    # Rays from center to corners:
    for corner in corners_world:
        segment = np.stack([center, corner], axis=0)
        ax.plot(
            segment[:, 0], segment[:, 1], segment[:, 2],
            color=color, linestyle=linestyle, linewidth=linewidth * 0.8,
        )

    # Optical axis:
    axis_end = center + rotation.T @ np.array([0.0, 0.0, scale * 1.3])
    ax.plot(
        [center[0], axis_end[0]], [center[1], axis_end[1]], [center[2], axis_end[2]],
        color=color, linestyle=linestyle, linewidth=linewidth * 1.2,
    )

    # Camera center dot:
    ax.scatter([center[0]], [center[1]], [center[2]], color=color, s=45, depthshade=False)


def draw_camera_view(
    ax: Axes,
    points_true: np.ndarray,
    points_noisy: np.ndarray,
    lines: np.ndarray | None,
    title: str,
    point_colors: list[str],
) -> None:
    """Draw a 2D camera sensor view with projected keypoints and optional epipolar lines."""
    # Plot epipolar lines if supplied (in Camera 2):
    if lines is not None:
        u_vals = np.linspace(-0.6, 0.6, 100)
        for line, color in zip(lines, point_colors):
            a, b, c = line
            if np.abs(b) > 1e-6:
                v_vals = -(a * u_vals + c) / b
                valid = np.abs(v_vals) <= 0.65
                if np.any(valid):
                    ax.plot(u_vals[valid], v_vals[valid], color=color, alpha=0.35, linewidth=1.0)

    # True keypoint positions:
    ax.scatter(
        points_true[:, 0], points_true[:, 1],
        color="#94a3b8", s=30, alpha=0.6, marker="o", label="True projection",
    )

    # Noisy measured keypoints:
    for pt, color in zip(points_noisy, point_colors):
        ax.scatter([pt[0]], [pt[1]], color=color, s=40, edgecolors="#1e293b", linewidths=0.6)

    ax.scatter([], [], color="#3b82f6", s=40, label="Measured keypoint (with noise)")

    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Sensor u", fontsize=8, labelpad=-2)
    ax.set_ylabel("Sensor v", fontsize=8, labelpad=-2)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.8)


def draw_3d_reconstruction(
    ax: Axes,
    points_true: np.ndarray,
    points_reconstructed: np.ndarray,
    c1_world: np.ndarray,
    r1_world: np.ndarray,
    c2_true: np.ndarray,
    r2_true: np.ndarray,
    c2_est: np.ndarray,
    r2_est: np.ndarray,
    title: str,
    point_colors: list[str],
) -> None:
    """Draw the 3D world scene overlaying true vs estimated cameras and landmarks."""
    # True landmarks:
    ax.scatter(
        points_true[:, 0], points_true[:, 1], points_true[:, 2],
        color="#94a3b8", s=35, alpha=0.5, label="Ground truth 3D landmarks",
    )

    # Reconstructed landmarks:
    for pt, color in zip(points_reconstructed, point_colors):
        ax.scatter(
            [pt[0]], [pt[1]], [pt[2]],
            color=color, marker="*", s=90, depthshade=False,
        )
    ax.scatter([], [], color="#10b981", marker="*", s=90, label="Triangulated 3D points")

    # Camera frustums:
    identity_rot = np.eye(3)
    draw_camera_frustum(ax, c1_world, identity_rot, 0.4, "#0284c7", "-", 1.5, "Camera 1 (World origin)")
    draw_camera_frustum(ax, c2_true, r2_true, 0.4, "#94a3b8", "--", 1.2, "Camera 2 Ground Truth")
    draw_camera_frustum(ax, c2_est, r2_est, 0.4, "#10b981", "-", 1.6, "Camera 2 Reconstructed")

    # Sight rays for a subset of landmarks to show intersection:
    for idx in range(0, len(points_true), 5):
        ray1_seg = np.stack([c1_world, points_reconstructed[idx]], axis=0)
        ray2_seg = np.stack([c2_est, points_reconstructed[idx]], axis=0)
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


def draw_epipolar_figure(
    panel_cam1: tuple[np.ndarray, np.ndarray, np.ndarray | None, str, list[str]],
    panel_cam2: tuple[np.ndarray, np.ndarray, np.ndarray | None, str, list[str]],
    panel_3d: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, list[str]],
    plot_path: Path,
) -> plt.Figure:
    """Render and save the three-panel epipolar reconstruction figure."""
    fig = plt.figure(figsize=(19.0, 5.8), dpi=140, layout="constrained")

    ax1 = fig.add_subplot(1, 3, 1)
    draw_camera_view(ax1, *panel_cam1)

    ax2 = fig.add_subplot(1, 3, 2)
    draw_camera_view(ax2, *panel_cam2)

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    draw_3d_reconstruction(ax3, *panel_3d)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, bbox_inches="tight")
    print(f"Figure saved to {plot_path}")
    return fig
