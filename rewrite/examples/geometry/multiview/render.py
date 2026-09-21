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
    # X_world = center + R @ X_cam
    corners_world = center + (rotation @ corners_local.T).T

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
    axis_end = center + rotation @ np.array([0.0, 0.0, scale * 1.25])
    ax.plot(
        [center[0], axis_end[0]], [center[1], axis_end[1]], [center[2], axis_end[2]],
        color=color, linestyle=linestyle, linewidth=linewidth * 1.2,
    )

    # Camera center:
    ax.scatter([center[0]], [center[1]], [center[2]], color=color, s=50, depthshade=False)


def draw_camera_wedge_2d(
    ax: Axes,
    center: np.ndarray,
    rotation: np.ndarray,
    scale: float = 0.35,
    half_fov_deg: float = 24.0,
    color: str = "#0284c7",
    label: str | None = None,
) -> None:
    """Draw a 2D camera FOV wedge, sensor plane, and optical axis in the X-Z plane."""
    half_fov = np.radians(half_fov_deg)
    dx = scale * np.tan(half_fov)

    # Sensor plane corners in local camera frame:
    corners_local = np.array([
        [-dx, 0.0, scale],
        [ dx, 0.0, scale],
    ])
    corners_world = center + (rotation @ corners_local.T).T
    c_xz = center[[0, 2]]
    p_left = corners_world[0, [0, 2]]
    p_right = corners_world[1, [0, 2]]
    tip = center + rotation @ np.array([0.0, 0.0, scale * 1.15])
    tip_xz = tip[[0, 2]]

    # Shaded FOV wedge:
    triangle = np.stack([c_xz, p_left, p_right], axis=0)
    ax.fill(triangle[:, 0], triangle[:, 1], color=color, alpha=0.18, zorder=3)

    # FOV boundary rays:
    ax.plot([c_xz[0], p_left[0]], [c_xz[1], p_left[1]], color=color, linewidth=1.1, linestyle="-", zorder=4)
    ax.plot([c_xz[0], p_right[0]], [c_xz[1], p_right[1]], color=color, linewidth=1.1, linestyle="-", zorder=4)

    # Sensor plane bar:
    ax.plot([p_left[0], p_right[0]], [p_left[1], p_right[1]], color=color, linewidth=2.4, linestyle="-", zorder=4)

    # Optical axis centerline:
    ax.plot([c_xz[0], tip_xz[0]], [c_xz[1], tip_xz[1]], color=color, linewidth=1.2, linestyle="--", zorder=4)

    # Camera center:
    ax.scatter([c_xz[0]], [c_xz[1]], color=color, s=70, edgecolors="#0f172a", linewidths=1.5, zorder=5, label=label)


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


def project_quadric_xz(q_mat: np.ndarray) -> np.ndarray:
    """Project a 4x4 homogeneous quadric to 3x3 in (x, z, w) via Schur complement on y."""
    idx = [0, 2, 3]
    q_sub = q_mat[np.ix_(idx, idx)]
    q_col = q_mat[idx, 1:2]
    q_yy = q_mat[1, 1]
    if abs(q_yy) < 1e-12:
        return q_sub
    return q_sub - (q_col @ q_col.T) / q_yy


def draw_implicit_conic_2d(
    ax: Axes,
    q_mat_xz: np.ndarray,
    x_grid: np.ndarray,
    z_grid: np.ndarray,
    level: float,
    color: str,
    alpha_fill: float = 0.10,
    alpha_edge: float = 0.40,
    linewidth: float = 1.0,
    z_min: float | None = None,
    zorder_fill: int = 2,
    zorder_edge: int = 3,
) -> None:
    """Render a 2D quadric level set f(x, z) <= level as shaded fill and contour edge."""
    pts = np.stack([x_grid, z_grid, np.ones_like(x_grid)], axis=-1)
    val = np.einsum("...i,ij,...j->...", pts, q_mat_xz, pts).copy()
    if z_min is not None:
        val[z_grid < z_min] = np.nan

    ax.contourf(
        x_grid, z_grid, val,
        levels=[0, level], colors=[color], alpha=alpha_fill, zorder=zorder_fill,
    )
    ax.contour(
        x_grid, z_grid, val,
        levels=[level], colors=[color], linewidths=linewidth, alpha=alpha_edge, zorder=zorder_edge,
    )


def draw_top_down_view(
    ax: Axes,
    cams_true: np.ndarray,
    cams_est: np.ndarray,
    points_true: np.ndarray,
    points_est: np.ndarray,
    covariances: np.ndarray,
    cam_colors: list[str],
    rots_est: list[np.ndarray] | None = None,
    world_cones: np.ndarray | None = None,
    fused_quadrics: np.ndarray | None = None,
) -> None:
    """Render top-down floorplan (X vs Z depth) of camera constellation and landmarks."""
    # Cameras: draw FOV wedge triangle with sensor plane and optical axis
    for idx, (ct, ce, col) in enumerate(zip(cams_true, cams_est, cam_colors)):
        lbl_true = f"Cam {idx} (Ref)" if idx == 0 else f"Cam {idx}"
        rot = rots_est[idx] if (rots_est is not None and idx < len(rots_est)) else np.eye(3)
        draw_camera_wedge_2d(ax, ce, rot, scale=0.35, color=col, label=lbl_true)
        if idx > 0:
            ax.scatter([ce[0]], [ce[2]], color=col, s=70, marker="x", linewidths=2.0, zorder=6)

    # If perspective cone quadrics are available, render ray conics and splats implicitly:
    if world_cones is not None:
        x_grid = np.linspace(-1.30, 1.30, 260)
        z_grid = np.linspace(-0.35, 4.85, 260)
        X, Z = np.meshgrid(x_grid, z_grid)
        level = 0.018

        n_points, n_cams = world_cones.shape[:2]
        for p_idx in range(n_points):
            for c_idx in range(n_cams):
                q_cone_xz = project_quadric_xz(world_cones[p_idx, c_idx])
                col = cam_colors[c_idx % len(cam_colors)]
                draw_implicit_conic_2d(
                    ax, q_cone_xz, X, Z, level=level, color=col,
                    alpha_fill=0.08, alpha_edge=0.35, linewidth=0.75,
                    z_min=cams_est[c_idx, 2] + 0.15,
                    zorder_fill=2, zorder_edge=2,
                )

            if fused_quadrics is not None:
                q_fused_xz = project_quadric_xz(fused_quadrics[p_idx])
                draw_implicit_conic_2d(
                    ax, q_fused_xz, X, Z, level=level, color="#10b981",
                    alpha_fill=0.35, alpha_edge=0.90, linewidth=1.5,
                    zorder_fill=4, zorder_edge=5,
                )

        for c_idx in range(n_cams):
            col = cam_colors[c_idx % len(cam_colors)]
            ax.plot([], [], color=col, linewidth=1.2, alpha=0.6, label=rf"Cam {c_idx} ray conic ($P^T Q_{c_idx} P \leq c$)")
        ax.plot([], [], color="#059669", linewidth=1.6, label=r"Fused splat ($P^T \sum Q_c P \leq c$)")
    else:
        # Fallback: 2D Gaussian uncertainty ellipses in the X-Z depth plane
        for pt, cov in zip(points_est, covariances):
            cov_xz = np.array([
                [cov[0, 0], cov[0, 2]],
                [cov[2, 0], cov[2, 2]],
            ])
            draw_ellipse_2d(ax, pt[[0, 2]], cov_xz, scale_factor=0.08, color="#0ea5e9", alpha=0.18)
        ax.plot([], [], color="#0ea5e9", linewidth=1.2, label="Gaussian 1σ ellipse (X–Z)")

        for c_idx in range(len(cams_est)):
            cam_col = cam_colors[c_idx % len(cam_colors)]
            for pt in points_est:
                ax.plot(
                    [cams_est[c_idx, 0], pt[0]],
                    [cams_est[c_idx, 2], pt[2]],
                    color=cam_col,
                    linestyle=":",
                    alpha=0.30,
                    linewidth=0.9,
                    zorder=1,
                )

    # Landmarks:
    ax.scatter(
        points_true[:, 0], points_true[:, 2],
        color="#94a3b8", s=35, marker="o", alpha=0.6, zorder=6, label="Ground truth landmarks",
    )
    ax.scatter(
        points_est[:, 0], points_est[:, 2],
        color="#059669", s=55, marker="*", zorder=7, label="Triangulated landmarks",
    )

    ax.set_aspect("equal")
    ax.set_xlim(-1.30, 1.30)
    ax.set_ylim(-0.35, 4.85)
    ax.set_title("Top-Down Geometry (X–Z Depth Plane)\nImplicit ray conics intersect to form Gaussian splats ($Q_{fused} = Q_0 + Q_1$)", fontsize=10.5, pad=10)
    ax.set_xlabel("X (meters)", fontsize=9, labelpad=4)
    ax.set_ylabel("Z (depth, meters)", fontsize=9, labelpad=4)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.92)


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

    ax.set_title("3D World Scene & Perspective Cone Quadrics\nFused quadrics form Gaussian splat uncertainty ellipsoids", fontsize=11, pad=10)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.0, 1.0)
    ax.set_zlim(1.0, 4.4)
    ax.set_box_aspect([1.8, 1.5, 2.6])
    ax.set_xlabel("X (m)", fontsize=8, labelpad=-5)
    ax.set_ylabel("Y (m)", fontsize=8, labelpad=-5)
    ax.set_zlabel("Z (m)", fontsize=8, labelpad=-5)
    ax.view_init(elev=20, azim=-60)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.85)


def draw_top_down_figure(
    top_down_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[np.ndarray] | None],
    plot_path: Path | None = None,
    auto_increment: bool = True,
) -> plt.Figure:
    """Render and save a dedicated standalone 2D top-down geometry figure."""
    fig, ax = plt.subplots(figsize=(7.5, 9.0), dpi=140, layout="constrained")
    draw_top_down_view(ax, *top_down_data)

    if plot_path is not None:
        target_path = plot_path
        if auto_increment:
            from examples import auto_increment_path
            target_path = auto_increment_path(plot_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_path, bbox_inches="tight")
        print(f"[render 2D] Figure saved to: {target_path}")
    return fig


def draw_3d_figure(
    world_3d_data: tuple[np.ndarray, list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[str]],
    plot_path: Path | None = None,
    auto_increment: bool = True,
) -> plt.Figure:
    """Render and save a dedicated standalone 3D world scene figure."""
    fig = plt.figure(figsize=(9.0, 8.0), dpi=140, layout="constrained")
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    draw_3d_world(ax, *world_3d_data)

    if plot_path is not None:
        target_path = plot_path
        if auto_increment:
            from examples import auto_increment_path
            target_path = auto_increment_path(plot_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_path, bbox_inches="tight")
        print(f"[render 3D] Figure saved to: {target_path}")
    return fig


def draw_multiview_figure(
    top_down_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[np.ndarray] | None],
    world_3d_data: tuple[np.ndarray, list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray, np.ndarray, np.ndarray, list[str]],
    convergence_history: list[float] | None = None,
    plot_path: Path | None = None,
    auto_increment: bool = True,
) -> plt.Figure:
    """Render and save the two-panel multi-camera reconstruction figure."""
    fig = plt.figure(figsize=(14.0, 6.0), dpi=140, layout="constrained")

    ax1 = fig.add_subplot(1, 2, 1)
    draw_top_down_view(ax1, *top_down_data)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    draw_3d_world(ax2, *world_3d_data)

    if plot_path is not None:
        target_path = plot_path
        if auto_increment:
            from examples import auto_increment_path
            target_path = auto_increment_path(plot_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_path, bbox_inches="tight")
        print(f"[render] Figure saved to: {target_path}")
    return fig
