"""2D visualization and convergence animation for multi-camera bundle adjustment in PGA2D.

Renders camera sight ray cones and fused landmark splats natively via NumGA
quadric evaluation on plane pixels: (Q(pixels) & pixels).
"""

from __future__ import annotations

from pathlib import Path
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from examples.geometry.multiview.types import (
    Motor,
    Point,
    Quadric,
    TwistMap,
    coordinates,
    mv,
    point,
    w,
)


def extract_covariances(quadrics: Quadric) -> np.ndarray:
    """Extract Cartesian covariance matrices from fused precision quadrics.

    Parameters
    ----------
    quadrics : [n_points] Quadric
        Fused perspective cone quadrics (Plane <- Point).

    Returns
    -------
    covariances : [n_points, 2, 2] np.ndarray
        Spatial covariance matrices in Cartesian coordinates.
    """
    return np.linalg.pinv(quadrics.kernel[..., :-1, :-1], rcond=1e-4)


def draw_camera_wedge_2d(
    ax: plt.Axes,
    center: np.ndarray,
    optical_axis: np.ndarray,
    scale: float = 0.28,
    half_fov_deg: float = 38.0,
    color: str = "#0284c7",
    label: str | None = None,
) -> None:
    """Draw a 2D camera FOV wedge, sensor line, and optical axis in the 2D plane."""
    half_fov = np.radians(half_fov_deg)
    perp = np.array([-optical_axis[1], optical_axis[0]])
    p_left = center + optical_axis * scale - perp * (scale * np.tan(half_fov))
    p_right = center + optical_axis * scale + perp * (scale * np.tan(half_fov))
    tip = center + optical_axis * (scale * 1.15)

    # Shaded FOV wedge:
    triangle = np.stack([center, p_left, p_right], axis=0)
    ax.fill(triangle[:, 0], triangle[:, 1], color=color, alpha=0.18, zorder=3)

    # Boundary rays:
    ax.plot([center[0], p_left[0]], [center[1], p_left[1]], color=color, linewidth=1.1, alpha=0.6, zorder=3)
    ax.plot([center[0], p_right[0]], [center[1], p_right[1]], color=color, linewidth=1.1, alpha=0.6, zorder=3)

    # Sensor line segment:
    ax.plot([p_left[0], p_right[0]], [p_left[1], p_right[1]], color=color, linewidth=2.0, zorder=4)

    # Optical axis dashed centerline:
    ax.plot([center[0], tip[0]], [center[1], tip[1]], color=color, linewidth=1.2, linestyle="--", zorder=4)

    # Camera center:
    ax.plot(
        center[0], center[1], marker="o", markersize=6.5, color=color,
        markeredgecolor="#0f172a", markeredgewidth=1.3, zorder=5, label=label,
    )


def draw_camera_pose_covariance_2d(
    ax: plt.Axes,
    center: np.ndarray,
    optical_axis: np.ndarray,
    cov_twist: np.ndarray,
    color: str = "#ec4899",
    scale_factor: float = 0.08,
    label: str | None = None,
) -> None:
    """Draw 2D camera pose uncertainty (position covariance ellipse and angular fan).

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes.
    center : np.ndarray
        Camera center position in world frame (x, y).
    optical_axis : np.ndarray
        Camera optical axis vector in world frame (ux, uy).
    cov_twist : [3, 3] np.ndarray
        Pose covariance matrix on se(2) twist generators [yw, xw, xy].
    color : str
        Color for ellipse and fan.
    scale_factor : float
        Visual scaling factor for 1-sigma uncertainty radii.
    label : str | None
        Legend label for the ellipse.
    """
    from matplotlib.patches import Ellipse, Arc

    # 1. Extract local Cartesian translation covariance:
    # Under twist [yw, xw, xy], local displacement is dx = -xw, dy = yw.
    cov_local = np.array([
        [ cov_twist[1, 1], -cov_twist[1, 0]],
        [-cov_twist[0, 1],  cov_twist[0, 0]],
    ])

    # 2. Rotate to world frame:
    norm_opt = np.linalg.norm(optical_axis)
    u_opt = optical_axis / (norm_opt if norm_opt > 0 else 1.0)
    u_perp = np.array([u_opt[1], -u_opt[0]])
    R = np.column_stack([u_perp, u_opt])
    cov_world = R @ cov_local @ R.T

    # 3. Position uncertainty ellipse:
    evals, evecs = np.linalg.eigh(cov_world)
    radii = np.sqrt(np.maximum(evals, 1e-8)) * scale_factor
    angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))

    ell = Ellipse(
        xy=center,
        width=2 * radii[0],
        height=2 * radii[1],
        angle=angle,
        edgecolor=color,
        facecolor=color,
        alpha=0.28,
        linewidth=1.6,
        linestyle="--",
        zorder=6,
        label=label,
    )
    ax.add_patch(ell)

    # 4. Angular uncertainty fan (orientation standard deviation on xy generator):
    rot_std_rad = np.sqrt(max(cov_twist[2, 2], 0.0))
    rot_std_deg = float(np.degrees(rot_std_rad) * 0.35)
    arc_r = 0.38
    theta_cam = float(np.degrees(np.arctan2(u_opt[1], u_opt[0])))

    arc = Arc(
        xy=center,
        width=2 * arc_r,
        height=2 * arc_r,
        angle=0,
        theta1=theta_cam - rot_std_deg,
        theta2=theta_cam + rot_std_deg,
        color=color,
        linewidth=1.3,
        linestyle=":",
        zorder=6,
    )
    ax.add_patch(arc)

    for sign in [-1, 1]:
        th = np.radians(theta_cam + sign * rot_std_deg)
        p_end = center + arc_r * np.array([np.cos(th), np.sin(th)])
        ax.plot([center[0], p_end[0]], [center[1], p_end[1]], color=color, linestyle=":", linewidth=1.1, zorder=6)


def blend_quadric_level_set(
    image: np.ndarray,
    quadric_val: np.ndarray,
    level_set: float | np.ndarray,
    color: np.ndarray,
    alpha: float,
    mask: np.ndarray,
    pixel_w: float,
) -> np.ndarray:
    """Blend an implicit quadric level set into image with anti-aliased hard edge."""
    dist = level_set - np.sqrt(np.maximum(quadric_val, 0.0))
    cov = (1.0 / (1.0 + np.exp(np.clip(-dist / (0.75 * pixel_w), -30.0, 30.0)))) * mask
    return image * (1.0 - (alpha * cov)[..., None]) + color * (alpha * cov)[..., None]


def rasterize_implicit_conics(
    world_cones: Quadric | None = None,
    fused_quadrics: Quadric | None = None,
    points: Point | None = None,
    cams_pos: list[np.ndarray] | None = None,
    cams_dirs: list[np.ndarray] | None = None,
    cam_colors: list[str] = ("#0284c7", "#ec4899"),
    shape: tuple[int, int] = (750, 750),
    x_range: tuple[float, float] = (-1.25, 1.25),
    y_range: tuple[float, float] = (-0.30, 2.95),
    theta_0: float = 0.035,
    motors: Motor | None = None,
    landmarks: Point | None = None,
) -> np.ndarray:
    """Rasterize camera perspective cones and fused splats via native quadric evaluation on 2D pixels.

    Parameters
    ----------
    world_cones : [n_points, n_cams] Quadric | None
        Perspective cone quadrics in world frame.
    fused_quadrics : [n_points] Quadric | None
        Fused precision quadrics (Gaussian splats).
    points : [n_points] Point | None
        Reconstructed scene points. If None, extracted from fused quadric centers.
    cams_pos : list[[2] np.ndarray] | None
        Camera center positions in world frame.
    cams_dirs : list[[2] np.ndarray] | None
        Camera optical axis unit vectors in world frame.
    cam_colors : list[str]
        Color hex strings for each camera.
    shape : tuple[int, int]
        Output image resolution (height, width).
    x_range : tuple[float, float]
        Grid bounding range along X axis.
    y_range : tuple[float, float]
        Grid bounding range along Y axis.
    theta_0 : float
        Angular pixel half-width.
    motors : [n_cams] Motor | None
        Camera poses in world frame.
    landmarks : [n_points] Point | None
        Legacy alias for points.

    Returns
    -------
    image : [height, width, 3] np.ndarray
        Rendered RGB image in [0, 1].
    """
    import matplotlib.colors as mcolors

    n_cams = len(cams_pos) if cams_pos is not None else 0
    if world_cones is not None:
        n_points = world_cones.shape[0]
    elif fused_quadrics is not None:
        n_points = fused_quadrics.shape[0]
    elif points is not None:
        n_points = len(points)
    else:
        n_points = 0

    xs = np.linspace(x_range[0], x_range[1], shape[1])
    ys = np.linspace(y_range[0], y_range[1], shape[0])
    xx, yy = np.meshgrid(xs, ys)
    pixel_w = (x_range[1] - x_range[0]) / shape[1]

    grid = point(np.stack([xx, yy], axis=-1))

    image = np.ones((*shape, 3), dtype=np.float32)

    # Camera focal planes in world frame:
    focal_planes = motors >> mv.y if motors is not None else None

    # Optical depth from each camera's focal plane to the 2D grid via native PGA inner product:
    depths = [
        (focal_planes[c] & grid).kernel[..., 0]
        for c in range(n_cams)
    ] if focal_planes is not None else []
    front = [d > 0.02 for d in depths] if depths else []
    front_all = np.zeros(shape, dtype=bool)
    for f in front:
        front_all = front_all | f

    if world_cones is not None:
        for c in range(n_cams):
            col_rgb = np.array(mcolors.to_rgb(cam_colors[c % len(cam_colors)]))
            cone_radius = theta_0 * np.maximum(depths[c], 0.05) if depths else 0.05
            for p in range(n_points):
                val = np.maximum((world_cones[p, c](grid) & grid).kernel[..., 0], 0.0)
                image = blend_quadric_level_set(
                    image=image,
                    quadric_val=val,
                    level_set=cone_radius,
                    color=col_rgb,
                    alpha=0.30,
                    mask=front[c] if front else np.ones(shape, dtype=bool),
                    pixel_w=pixel_w,
                )

    if fused_quadrics is not None:
        if points is None:
            points = landmarks
        if points is None:
            points = (fused_quadrics + w * (w & Point)).solve(w).normalized()

        col_splat = np.array([0.98, 0.48, 0.04])
        f_min = np.maximum((fused_quadrics(points) & points).kernel[..., 0], 0.0)

        # Point depth across cameras is natively the inner product with each focal plane:
        for p in range(n_points):
            d_p = float((focal_planes & points[p]).kernel.mean())
            base_r = theta_0 * np.maximum(d_p, 0.2)
            eff_r = np.sqrt(base_r**2 + f_min[p])
            val_splat = (fused_quadrics[p](grid) & grid).kernel[..., 0] - f_min[p]
            image = blend_quadric_level_set(
                image=image,
                quadric_val=val_splat,
                level_set=eff_r,
                color=col_splat,
                alpha=0.95,
                mask=front_all,
                pixel_w=pixel_w,
            )

    return np.clip(image, 0.0, 1.0)


def draw_top_down_view(
    ax: plt.Axes,
    world_cones: Quadric | None = None,
    fused_quadrics: Quadric | None = None,
    points: Point | None = None,
    motors: Motor | None = None,
    cams_pos: list[np.ndarray] | None = None,
    cams_dirs: list[np.ndarray] | None = None,
    cam_colors: list[str] = ("#0284c7", "#ec4899"),
    pose_covariances: TwistMap | np.ndarray | None = None,
    splats: Quadric | None = None,
    landmarks: Point | None = None,
    poses: Motor | None = None,
    cones: Quadric | None = None,
) -> None:
    """Render 2D floorplan of camera constellation, perspective cones, and fused splats.

    Parameters
    ----------
    ax : plt.Axes
        Target matplotlib axes.
    world_cones : [n_points, n_cams] Quadric | None
        Perspective cone quadrics in world frame.
    fused_quadrics : [n_points] Quadric | None
        Fused precision quadrics.
    points : [n_points] Point | None
        Reconstructed scene points.
    motors : [n_cams] Motor | None
        Camera poses in world frame (or poses).
    cams_pos : list[[2] np.ndarray] | None
        Camera center positions in world frame.
    cams_dirs : list[[2] np.ndarray] | None
        Camera optical axis unit vectors in world frame.
    cam_colors : list[str]
        Color hex strings for each camera.
    pose_covariances : [n_cams] TwistMap | None
        Camera pose covariance operators.
    splats : [n_points] Quadric | None
        Alias for fused_quadrics.
    landmarks : [n_points] Point | None
        Legacy alias for points.
    poses : [n_cams] Motor | None
        Alias for motors.
    cones : [n_points, n_cams] Quadric | None
        Alias for world_cones.
    """
    if splats is not None:
        fused_quadrics = splats
    if points is None:
        points = landmarks
    if motors is None:
        motors = poses
    if world_cones is None:
        world_cones = cones
    x_range = (-1.25, 1.25)
    y_range = (-0.30, 2.95)

    if cams_pos is None and motors is not None:
        c0 = point([0.0, 0.0])
        cams_pos = [coordinates(m >> c0) for m in motors]
    if cams_dirs is None and motors is not None:
        cams_dirs = [(m >> mv.y).kernel[:2] for m in motors]

    if world_cones is not None or fused_quadrics is not None:
        img = rasterize_implicit_conics(
            world_cones=world_cones,
            fused_quadrics=fused_quadrics,
            points=points,
            cams_pos=cams_pos,
            cams_dirs=cams_dirs,
            cam_colors=cam_colors,
            shape=(750, 750),
            x_range=x_range,
            y_range=y_range,
            motors=motors,
        )
        ax.imshow(img, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], origin="lower")

    # Overlay scene landmarks if provided:
    if points is not None:
        pts_xy = coordinates(points)
        ax.plot(
            pts_xy[:, 0], pts_xy[:, 1],
            marker="o", markersize=6.0, linestyle="none",
            color="#ea580c" if fused_quadrics is not None else "#0f172a",
            markeredgecolor="white", markeredgewidth=1.2, zorder=6,
            label="Reconstructed Points" if fused_quadrics is not None else "Scene Points",
        )

    # Overlay camera frustum wedges and pose covariance:
    if cams_pos is not None and cams_dirs is not None:
        for c_idx in range(len(cams_pos)):
            col = cam_colors[c_idx % len(cam_colors)]
            lbl = f"Cam {c_idx} (ref)" if c_idx == 0 else f"Cam {c_idx}"
            draw_camera_wedge_2d(ax, cams_pos[c_idx], cams_dirs[c_idx], scale=0.28, half_fov_deg=38.0, color=col, label=lbl)

            if pose_covariances is not None:
                cov_k = pose_covariances[c_idx].kernel if hasattr(pose_covariances[c_idx], "kernel") else pose_covariances[c_idx]
                if np.linalg.norm(cov_k) > 1e-6:
                    draw_camera_pose_covariance_2d(
                        ax=ax,
                        center=cams_pos[c_idx],
                        optical_axis=cams_dirs[c_idx],
                        cov_twist=cov_k,
                        color=col,
                        scale_factor=0.08,
                        label=f"Cam {c_idx} Pose Covariance (1σ)",
                    )

    ax.set_aspect("equal")
    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.axis("off")


def draw_top_down_figure(
    world_cones: Quadric | None = None,
    fused_quadrics: Quadric | None = None,
    points: Point | None = None,
    motors: Motor | None = None,
    cams_pos: list[np.ndarray] | None = None,
    cams_dirs: list[np.ndarray] | None = None,
    cam_colors: list[str] = ("#0284c7", "#ec4899"),
    pose_covariances: TwistMap | np.ndarray | None = None,
    plot_path: Path | None = None,
    auto_increment: bool = True,
    splats: Quadric | None = None,
    landmarks: Point | None = None,
    poses: Motor | None = None,
    cones: Quadric | None = None,
) -> plt.Figure:
    """Render and save 2D floorplan figure.

    Parameters
    ----------
    world_cones : [n_points, n_cams] Quadric | None
        Perspective cone quadrics in world frame (or cones).
    fused_quadrics : [n_points] Quadric | None
        Fused precision quadrics (or splats).
    points : [n_points] Point | None
        Reconstructed scene points.
    motors : [n_cams] Motor | None
        Camera poses in world frame (or poses).
    cams_pos : list[[2] np.ndarray] | None
        Camera center positions in world frame.
    cams_dirs : list[[2] np.ndarray] | None
        Camera optical axis unit vectors in world frame.
    cam_colors : list[str]
        Color hex strings for each camera.
    pose_covariances : [n_cams] TwistMap | None
        Camera pose covariance operators.
    plot_path : Path | None
        Output filepath.
    auto_increment : bool
        Whether to increment output filename.
    splats : [n_points] Quadric | None
        Alias for fused_quadrics.
    landmarks : [n_points] Point | None
        Legacy alias for points.
    poses : [n_cams] Motor | None
        Alias for motors.
    cones : [n_points, n_cams] Quadric | None
        Alias for world_cones.
    """
    if splats is not None:
        fused_quadrics = splats
    if points is None:
        points = landmarks
    if motors is None:
        motors = poses
    if world_cones is None:
        world_cones = cones
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=140, layout="constrained")
    draw_top_down_view(
        ax=ax,
        world_cones=world_cones,
        fused_quadrics=fused_quadrics,
        points=points,
        motors=motors,
        cams_pos=cams_pos,
        cams_dirs=cams_dirs,
        cam_colors=cam_colors,
        pose_covariances=pose_covariances,
    )

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92, facecolor="#ffffff", edgecolor="#cbd5e1")

    if plot_path is not None:
        target_path = plot_path
        if auto_increment:
            from examples import auto_increment_path
            target_path = auto_increment_path(plot_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_path, bbox_inches="tight")
        print(f"[render] Figure saved to: {target_path}")
    return fig


def animate_top_down_convergence(
    history: list[tuple],
    local_cones: Quadric,
    cam_colors: list[str] = ("#0284c7", "#ec4899"),
    gif_path: Path | str | None = None,
    fps: int = 3,
    auto_increment: bool = True,
) -> Path | None:
    """Render and save an animated GIF of bundle adjustment convergence in PGA2D.

    Parameters
    ----------
    history : list[tuple[Motor, Point, Quadric] | tuple[Motor, Point, Quadric, TwistMap]]
        Per-iteration optimization state tuples.
    local_cones : [n_points, n_cams] Quadric
        Perspective cone quadrics in camera local frames.
    cam_colors : list[str]
        Color hex strings for each camera.
    gif_path : Path | str | None
        Target path to save GIF.
    fps : int
        Frames per second in GIF.
    auto_increment : bool
        Whether to auto-increment file name if target path exists.

    Returns
    -------
    target_path : Path | None
        Path to written GIF file.
    """
    mv = history[0][0].context.multivector
    c_local = mv.antivector([0.0, 0.0, 1.0])

    frames = []
    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=120, layout="constrained")

    for it, item in enumerate(history):
        ax.clear()
        if len(item) == 4:
            m_curr, pts_curr, q_fused_curr, cov_curr = item
        else:
            m_curr, pts_curr, q_fused_curr = item[:3]
            cov_curr = None

        c_world = m_curr >> c_local
        cams_pos = [coordinates(c) for c in c_world]
        cams_dirs = [(m >> mv.y).kernel[:2] for m in m_curr]
        world_cones = m_curr >> local_cones(m_curr << Point)

        draw_top_down_view(
            ax=ax,
            cams_pos=cams_pos,
            cams_dirs=cams_dirs,
            world_cones=world_cones,
            fused_quadrics=q_fused_curr,
            points=pts_curr,
            cam_colors=cam_colors,
            pose_covariances=cov_curr,
            motors=m_curr,
        )

        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frames.append(rgba[..., :3].copy())

    plt.close(fig)

    target_path = Path(gif_path) if gif_path is not None else None
    if auto_increment and target_path is not None:
        from examples import auto_increment_path
        target_path = auto_increment_path(target_path)

    if target_path is not None:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        duration_ms = int(1000 / fps) if fps > 0 else 330
        imageio.mimsave(target_path, frames, duration=duration_ms, loop=0)
        print(f"[render animation] GIF saved to: {target_path}")

    return target_path
