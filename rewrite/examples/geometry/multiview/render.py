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
    ax.plot(c_xz[0], c_xz[1], marker="o", markersize=6.5, color=color, markeredgecolor="#0f172a", markeredgewidth=1.3, zorder=5, label=label)


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


def rasterize_implicit_conics(
    world_cones_mat: np.ndarray,
    fused_quadrics_mat: np.ndarray,
    cams_pos: np.ndarray,
    cams_rots: list[np.ndarray],
    landmarks_xyz: np.ndarray,
    shape: tuple[int, int] = (700, 800),
    x_range: tuple[float, float] = (-1.80, 1.80),
    z_range: tuple[float, float] = (-0.35, 3.15),
    theta_0: float = 0.016,
    splat_scale: float = 1.5,
) -> np.ndarray:
    """Rasterize 2D camera ray conics and fused Gaussian splats directly from implicit quadrics.

    Evaluates the homogeneous quadratic forms on the 2D pixel grid and computes anti-aliased
    logistic edge coverage at the zero locus, fundamentally mirroring the raytracer pattern.
    """
    height, width = shape
    xs = np.linspace(x_range[0], x_range[1], width)
    zs = np.linspace(z_range[0], z_range[1], height)
    X, Z = np.meshgrid(xs, zs)
    pixel_w = xs[1] - xs[0]

    opt_axes = [rot[:, 2] for rot in cams_rots]

    image = np.ones((height, width, 3), dtype=np.float32) * 0.99
    col_cam0 = np.array([0.05, 0.52, 0.88])  # blue
    col_cam1 = np.array([0.88, 0.18, 0.55])  # pink
    col_splat = np.array([0.04, 0.70, 0.42]) # emerald

    n_points, n_cams = world_cones_mat.shape[:2]

    # Precompute local depths along each camera optical axis:
    z_cams = []
    for c_pos, opt in zip(cams_pos, opt_axes):
        dx = X.ravel() - c_pos[0]
        dz = Z.ravel() - c_pos[2]
        z_loc = dx * opt[0] + dz * opt[2]
        z_cams.append(z_loc)

    cov0_total = np.zeros((height, width), dtype=np.float32)
    cov1_total = np.zeros((height, width), dtype=np.float32)
    covf_total = np.zeros((height, width), dtype=np.float32)
    covedge_total = np.zeros((height, width), dtype=np.float32)

    col_cam0 = np.array([0.05, 0.52, 0.88])   # blue
    col_cam1 = np.array([0.88, 0.18, 0.55])   # pink
    col_splat = np.array([0.98, 0.48, 0.04])  # high-viz radiant amber / orange
    col_edge = np.array([0.62, 0.18, 0.02])   # deep burnt amber rim

    for i in range(n_points):
        pt_xyz = landmarks_xyz[i]

        # Common epipolar plane through camera baseline and landmark i: y(Z) = (y_i / z_i) * Z
        # Ensures cones and fused splats intersect dead-center.
        slope_y = pt_xyz[1] / max(pt_xyz[2], 1e-4)
        Y_plane = slope_y * Z.ravel()
        pts = np.stack([X.ravel(), Y_plane, Z.ravel(), np.ones_like(X.ravel())], axis=-1)

        # Distance from each camera pinhole to all grid points:
        d0 = pts[:, :3] - cams_pos[0]
        R0 = np.linalg.norm(d0, axis=-1)
        z0 = z_cams[0]

        d1 = pts[:, :3] - cams_pos[1]
        R1 = np.linalg.norm(d1, axis=-1)
        z1 = z_cams[1]

        # Ray unit directions and cosine angles with optical axis:
        ray_dir0 = (pt_xyz - cams_pos[0]) / np.linalg.norm(pt_xyz - cams_pos[0])
        ray_dir1 = (pt_xyz - cams_pos[1]) / np.linalg.norm(pt_xyz - cams_pos[1])
        cos_phi0 = float(np.dot(ray_dir0, opt_axes[0]))
        cos_phi1 = float(np.dot(ray_dir1, opt_axes[1]))

        # Camera 0 perspective ray conic with identical angular opening angle theta_0 for all rays:
        Q0 = world_cones_mat[i, 0]
        v0 = -np.einsum("...i,ij,...j->...", pts, Q0, pts)
        d_perp0 = cos_phi0 * np.sqrt(np.maximum(-v0, 0.0))
        w0 = np.maximum(theta_0 * R0, 0.003)
        dist0 = w0 - d_perp0
        cov0 = 1.0 / (1.0 + np.exp(np.clip(-dist0 / (0.75 * pixel_w), -30.0, 30.0))).reshape(height, width)
        cov0 *= (z0 > 0.05).reshape(height, width)
        cov0_total = np.maximum(cov0_total, cov0)

        # Camera 1 perspective ray conic with identical angular opening angle theta_0 for all rays:
        Q1 = world_cones_mat[i, 1]
        v1 = -np.einsum("...i,ij,...j->...", pts, Q1, pts)
        d_perp1 = cos_phi1 * np.sqrt(np.maximum(-v1, 0.0))
        w1 = np.maximum(theta_0 * R1, 0.003)
        dist1 = w1 - d_perp1
        cov1 = 1.0 / (1.0 + np.exp(np.clip(-dist1 / (0.75 * pixel_w), -30.0, 30.0))).reshape(height, width)
        cov1 *= (z1 > 0.05).reshape(height, width)
        cov1_total = np.maximum(cov1_total, cov1)

        # Fused Gaussian splat from normalized precision quadrics:
        R_lm0 = float(np.linalg.norm(pt_xyz - cams_pos[0]))
        R_lm1 = float(np.linalg.norm(pt_xyz - cams_pos[1]))
        w_lm0 = theta_0 * R_lm0
        w_lm1 = theta_0 * R_lm1
        w_avg = 0.5 * (w_lm0 + w_lm1)

        M0 = (cos_phi0**2 / w_lm0**2) * Q0
        M1 = (cos_phi1**2 / w_lm1**2) * Q1
        Mf = M0 + M1

        vf = np.einsum("...i,ij,...j->...", pts, Mf, pts)
        res_f = np.sqrt(np.maximum(vf, 0.0))
        dist_f = (splat_scale - res_f) * w_avg
        cov_f = 1.0 / (1.0 + np.exp(np.clip(-dist_f / (0.85 * pixel_w), -30.0, 30.0))).reshape(height, width)
        cov_f *= ((z0 > 0.05) & (z1 > 0.05)).reshape(height, width)
        covf_total = np.maximum(covf_total, cov_f)

        # High-viz edge contour ring around the splat boundary:
        cov_edge = np.exp(-0.5 * (dist_f / (1.1 * pixel_w))**2).reshape(height, width)
        cov_edge *= ((z0 > 0.05) & (z1 > 0.05)).reshape(height, width)
        covedge_total = np.maximum(covedge_total, cov_edge)

    # 1. Blend ray conics into background:
    image -= cov0_total[..., None] * (1.0 - col_cam0) * 0.35
    image -= cov1_total[..., None] * (1.0 - col_cam1) * 0.35
    image = np.clip(image, 0.0, 1.0)

    # 2. Overlay Gaussian splats on top with vibrant fill:
    alpha_f = covf_total[..., None] * 0.92
    image = image * (1.0 - alpha_f) + col_splat * alpha_f

    # 3. High-viz dark edge stroke:
    alpha_edge = covedge_total[..., None] * 0.85
    image = image * (1.0 - alpha_edge) + col_edge * alpha_edge

    return np.clip(image, 0.0, 1.0)


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
    rots = rots_est if rots_est is not None else [np.eye(3)] * len(cams_est)
    x_range = (-1.25, 1.25)
    z_range = (-0.30, 2.95)

    if world_cones is not None and fused_quadrics is not None:
        img = rasterize_implicit_conics(
            world_cones, fused_quadrics, cams_est, rots, points_est,
            shape=(750, 750), x_range=x_range, z_range=z_range,
        )
        ax.imshow(img, extent=[x_range[0], x_range[1], z_range[0], z_range[1]], origin="lower")

    # Overlay camera frustums:
    for c_idx in range(len(cams_est)):
        col = cam_colors[c_idx % len(cam_colors)]
        draw_camera_wedge_2d(ax, cams_est[c_idx], rots[c_idx], scale=0.28, half_fov_deg=38.0, color=col)

    ax.set_aspect("equal")
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(z_range[0], z_range[1])
    ax.axis("off")


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
        draw_camera_frustum(ax, ce, re, scale=0.3, color=col, linestyle="-", linewidth=1.4)

    # Landmarks:
    ax.scatter(
        points_true[:, 0], points_true[:, 1], points_true[:, 2],
        color="#94a3b8", s=25, alpha=0.5,
    )
    ax.scatter(
        points_est[:, 0], points_est[:, 1], points_est[:, 2],
        color="#ea580c", s=55, marker="*", depthshade=False,
    )

    # Draw Gaussian splat precision ellipsoids around landmarks (1.5x scaled):
    for pt, cov in zip(points_est, covariances):
        draw_ellipsoid_3d(ax, pt, cov, scale_factor=0.12, color="#f97316", alpha=0.35)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.8, 0.8)
    ax.set_zlim(-0.3, 3.1)
    ax.set_box_aspect([2.4, 1.6, 3.4])
    ax.view_init(elev=28, azim=-65)
    ax.axis("off")


def draw_top_down_figure(
    top_down_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[np.ndarray] | None],
    plot_path: Path | None = None,
    auto_increment: bool = True,
) -> plt.Figure:
    """Render and save a dedicated standalone 2D top-down geometry figure."""
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=140, layout="constrained")
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
