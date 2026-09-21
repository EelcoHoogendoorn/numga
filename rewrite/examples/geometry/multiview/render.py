"""Drawing and visual helpers for multi-camera bundle adjustment and reconstruction.

Constructs viewport geometry and Gaussian splat uncertainty ellipsoids; the mathematics
in core never imports this module.
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
    label: str | None = None,
) -> None:
    """Draw a 3D camera wireframe pyramid representing center and sensor plane."""
    w, h = 0.45 * scale, 0.35 * scale
    corners_local = np.array([
        [-w, -h, scale],
        [ w, -h, scale],
        [ w,  h, scale],
        [-w,  h, scale],
    ])
    # X_world = center + R.T @ X_cam
    corners_world = center + (rotation.T @ corners_local.T).T

    # Sensor frame:
    loop = np.concatenate([corners_world, corners_world[:1]], axis=0)
    ax.plot(
        loop[:, 0], loop[:, 1], loop[:, 2],
        color=color, linestyle=linestyle, linewidth=linewidth, label=label,
    )

    # Rays to corners:
    for corner in corners_world:
        segment = np.stack([center, corner], axis=0)
        ax.plot(
            segment[:, 0], segment[:, 1], segment[:, 2],
            color=color, linestyle=linestyle, linewidth=linewidth * 0.75,
        )

    # Optical axis:
    axis_end = center + rotation.T @ np.array([0.0, 0.0, scale * 1.25])
    ax.plot(
        [center[0], axis_end[0]], [center[1], axis_end[1]], [center[2], axis_end[2]],
        color=color, linestyle=linestyle, linewidth=linewidth * 1.2,
    )

    # Camera center:
    ax.scatter([center[0]], [center[1]], [center[2]], color=color, s=50, depthshade=False)


def draw_ellipsoid_3d(
    ax: Axes,
    center: np.ndarray,
    covariance: np.ndarray,
    scale_factor: float = 0.15,
    color: str = "#f59e0b",
    alpha: float = 0.25,
) -> None:
    """Draw a 3D Gaussian splat uncertainty ellipsoid given center and covariance."""
    evals, evecs = np.linalg.eigh(covariance)
    radii = np.sqrt(np.maximum(evals, 1e-8)) * scale_factor

    # Parameterize unit sphere:
    u = np.linspace(0, 2 * np.pi, 18)
    v = np.linspace(0, np.pi, 12)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    unit_sphere = np.stack([x, y, z], axis=-1)  # [18, 12, 3]
    # Rotate and stretch:
    scaled = unit_sphere * radii[None, None, :]
    rotated = np.einsum("ij,...j->...i", evecs, scaled) + center

    ax.plot_wireframe(
        rotated[..., 0], rotated[..., 1], rotated[..., 2],
        color=color, alpha=alpha, linewidth=0.6,
    )


def draw_ellipse_2d(
    ax: Axes,
    center_xz: np.ndarray,
    cov_xz: np.ndarray,
    scale_factor: float = 0.08,
    color: str = "#0ea5e9",
    alpha: float = 0.18,
    edge_alpha: float = 0.7,
    linewidth: float = 1.0,
) -> None:
    """Draw a 2D Gaussian confidence ellipse in the X-Z depth plane."""
    evals, evecs = np.linalg.eigh(cov_xz)
    radii = np.sqrt(np.maximum(evals, 1e-8)) * scale_factor

    theta = np.linspace(0, 2 * np.pi, 60)
    circle = np.stack([np.cos(theta), np.sin(theta)], axis=0)
    ellipse = (evecs @ (radii[:, None] * circle)).T

    x_pts = center_xz[0] + ellipse[:, 0]
    z_pts = center_xz[1] + ellipse[:, 1]

    ax.fill(x_pts, z_pts, color=color, alpha=alpha, zorder=2)
    ax.plot(x_pts, z_pts, color=color, alpha=edge_alpha, linewidth=linewidth, zorder=3)


def draw_top_down_view(
    ax: Axes,
    cams_true: np.ndarray,
    cams_est: np.ndarray,
    points_true: np.ndarray,
    points_est: np.ndarray,
    covariances: np.ndarray,
    cam_colors: list[str],
    rots_est: list[np.ndarray] | None = None,
) -> None:
    """Render top-down floorplan (X vs Z depth) of camera constellation and landmarks."""
    # Cameras:
    for idx, (ct, ce, col) in enumerate(zip(cams_true, cams_est, cam_colors)):
        lbl_true = f"Cam {idx} (Ref)" if idx == 0 else f"Cam {idx}"
        ax.scatter([ct[0]], [ct[2]], color=col, s=80, marker="o", edgecolors="#1e293b", zorder=5, label=lbl_true)
        if idx > 0:
            ax.scatter([ce[0]], [ce[2]], color=col, s=70, marker="x", linewidths=2.0, zorder=5)

        # Draw optical axis pointer in X-Z plane:
        if rots_est is not None and idx < len(rots_est):
            axis_dir = rots_est[idx].T @ np.array([0.0, 0.0, 0.35])
            ax.plot([ce[0], ce[0] + axis_dir[0]], [ce[2], ce[2] + axis_dir[2]], color=col, linewidth=1.4, zorder=4)

    # 2D Gaussian uncertainty ellipses in the X-Z depth plane:
    for pt, cov in zip(points_est, covariances):
        cov_xz = np.array([
            [cov[0, 0], cov[0, 2]],
            [cov[2, 0], cov[2, 2]],
        ])
        draw_ellipse_2d(ax, pt[[0, 2]], cov_xz, scale_factor=0.08, color="#0ea5e9", alpha=0.18)
    ax.plot([], [], color="#0ea5e9", linewidth=1.2, label="Gaussian 1σ ellipse (X–Z)")

    # Landmarks:
    ax.scatter(
        points_true[:, 0], points_true[:, 2],
        color="#94a3b8", s=35, marker="o", alpha=0.6, zorder=4, label="Ground truth landmarks",
    )
    ax.scatter(
        points_est[:, 0], points_est[:, 2],
        color="#10b981", s=50, marker="*", zorder=4, label="Triangulated landmarks",
    )

    # Sight lines from Camera 0 and Camera 1 to first 4 landmarks:
    for p_idx in range(min(4, len(points_est))):
        pt = points_est[p_idx]
        ax.plot([cams_est[0, 0], pt[0]], [cams_est[0, 2], pt[2]], color=cam_colors[0], linestyle=":", alpha=0.35, linewidth=1.0)
        ax.plot([cams_est[1, 0], pt[0]], [cams_est[1, 2], pt[2]], color=cam_colors[1], linestyle=":", alpha=0.35, linewidth=1.0)

    ax.set_title("1. Top-Down Geometry (X–Z Depth Plane)\nCamera constellation & Gaussian uncertainty ellipses", fontsize=10, pad=8)
    ax.set_xlabel("X (meters)", fontsize=8, labelpad=2)
    ax.set_ylabel("Z (depth, meters)", fontsize=8, labelpad=2)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.85)


def draw_3d_world(
    ax: Axes,
    cams_true: np.ndarray,
    rots_true: list[np.ndarray],
    cams_est: np.ndarray,
    rots_est: list[np.ndarray],
    points_true: np.ndarray,
    points_est: np.ndarray,
    covariances: np.ndarray,
    cam_colors: list[str],
) -> None:
    """Render 3D world scene with camera frustums and Gaussian splat ellipsoids."""
    # Draw cameras:
    for idx, (ct, rt, ce, re, col) in enumerate(zip(cams_true, rots_true, cams_est, rots_est, cam_colors)):
        lbl = f"Cam {idx}" if idx <= 1 else None
        draw_camera_frustum(ax, ce, re, scale=0.3, color=col, linestyle="-", linewidth=1.4, label=lbl)

    # Landmarks:
    ax.scatter(
        points_true[:, 0], points_true[:, 1], points_true[:, 2],
        color="#94a3b8", s=30, alpha=0.6, label="Ground truth 3D",
    )
    ax.scatter(
        points_est[:, 0], points_est[:, 1], points_est[:, 2],
        color="#10b981", s=65, marker="*", depthshade=False, label="Reconstructed 3D",
    )

    # Draw Gaussian splat precision ellipsoids around landmarks:
    for pt, cov in zip(points_est, covariances):
        draw_ellipsoid_3d(ax, pt, cov, scale_factor=0.08, color="#0ea5e9", alpha=0.25)
    ax.plot([], [], color="#0ea5e9", linewidth=1.2, label="Gaussian Splat 1σ Ellipsoid")

    ax.set_title("2. 3D World Scene & Perspective Cone Quadrics\nFused quadrics form Gaussian splat uncertainty ellipsoids", fontsize=10, pad=8)
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(1.8, 4.4)
    ax.set_box_aspect([1.8, 1.6, 2.6])
    ax.set_xlabel("X (m)", fontsize=8, labelpad=-5)
    ax.set_ylabel("Y (m)", fontsize=8, labelpad=-5)
    ax.set_zlabel("Z (m)", fontsize=8, labelpad=-5)
    ax.view_init(elev=20, azim=-60)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.85)


def draw_convergence(
    ax: Axes,
    iteration_history: list[float],
) -> None:
    """Render bundle adjustment Gauss-Newton convergence plot."""
    iters = list(range(1, len(iteration_history) + 1))
    ax.semilogy(iters, iteration_history, color="#6366f1", marker="o", linewidth=1.8, markersize=5, label="Landmark RMSE (m)")
    ax.set_title("3. Gauss-Newton Convergence\nAlternating quadric triangulation & Lie algebra updates", fontsize=10, pad=8)
    ax.set_xlabel("Iteration", fontsize=8, labelpad=2)
    ax.set_ylabel("RMSE (meters)", fontsize=8, labelpad=2)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)


def draw_multiview_figure(
    top_down_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[np.ndarray] | None],
    world_3d_data: tuple[np.ndarray, list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[str]],
    convergence_history: list[float],
    plot_path: Path,
) -> plt.Figure:
    """Render and save the three-panel multi-camera reconstruction figure."""
    fig = plt.figure(figsize=(19.0, 5.8), dpi=140, layout="constrained")

    ax1 = fig.add_subplot(1, 3, 1)
    draw_top_down_view(ax1, *top_down_data)

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    draw_3d_world(ax2, *world_3d_data)

    ax3 = fig.add_subplot(1, 3, 3)
    draw_convergence(ax3, convergence_history)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, bbox_inches="tight")
    print(f"Figure saved to {plot_path}")
    return fig
