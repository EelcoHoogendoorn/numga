"""Matplotlib rendering and visualization for the scenegraph package.

ALL array extraction (.kernel, coordinate readout, and conversions for matplotlib)
belongs strictly inside this module at the JIT visualization boundary.
Notebooks and mathematical modules (core.py) operate purely on algebraic Extensors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from numga.algebras import PGA3D
from numga import NumpyContext

from examples import PLOT_DIR
from examples.animation import capture, save_gif
from .core import (
    Point,
    Line,
    Plane,
    Motor,
    PointMap,
    canonical_unit_box,
    unit_box_topology,
    make_robot_arm,
    make_camera_pose,
    make_multi_lens_camera,
    make_viewport,
    collapse_scenegraph,
    project_vertices,
    trace_sample_rays,
)

# Colors for robot arm bodies (curated dark/tech modern palette):
BODY_COLORS = [
    "#4A5568",  # Body 0: Base pedestal (charcoal slate)
    "#2B6CB0",  # Body 1: Turret (steel blue)
    "#3182CE",  # Body 2: Upper arm (vibrant azure)
    "#4299E1",  # Body 3: Forearm (sky blue)
    "#ED8936",  # Body 4: Gripper wrist (accent amber)
]
LINK_COLORS = BODY_COLORS  # Backward compatibility alias

RAY_COLORS = [
    "#E53E3E",  # Red / crimson
    "#38A169",  # Green / emerald
    "#D69E2E",  # Golden amber
]


# --- JIT coordinate extraction helpers (boundary to matplotlib only) ------------------
def _extract_xyz(points: Point) -> np.ndarray:
    """JIT boundary: extract Euclidean (x, y, z) coordinates from Point multivectors."""
    # Perspective-normalize points:
    ctx = NumpyContext(PGA3D)
    mv = ctx.multivector
    norm_points = points / (mv.w & points)
    k = norm_points.kernel
    # In PGA3D antivector:
    # mv.yzw is x, mv.zxw is y, mv.xyw is z, mv.zyx is w.
    return k[..., :3]


def _extract_uv(pixels: Point) -> tuple[np.ndarray, np.ndarray]:
    """JIT boundary: extract 2D pixel coordinates (u, v) from normalized screen points."""
    ctx = NumpyContext(PGA3D)
    mv = ctx.multivector
    u = (mv.x & pixels).kernel
    v = (mv.y & pixels).kernel
    return u, v


# --- 3D Scene View --------------------------------------------------------------------
def draw_scenegraph_scene_3d(
    ax: Axes3D,
    world_vertices: Point | None = None,
    camera_pose: Motor | None = None,
    sample_rays: list[tuple[Point, Point, Point, Point]] | None = None,
    bodies_to_world: PointMap | None = None,
    unit_box: Point | None = None,
    links_to_world: PointMap | None = None,
) -> None:
    """Render the 3D world scene: articulated robot arm, camera body, lens planes, and rays."""
    ax.clear()

    if world_vertices is None:
        if bodies_to_world is None:
            bodies_to_world = links_to_world
        if bodies_to_world is not None:
            if unit_box is None:
                unit_box = canonical_unit_box()
            world_vertices = bodies_to_world[:, None](unit_box[None, :])
        else:
            raise ValueError("Either world_vertices or bodies_to_world must be provided")

    edges, faces = unit_box_topology()

    # Directional light source for shading 3D body faces:
    light_dir = np.array([0.4, -0.6, 0.7])
    light_dir = light_dir / np.linalg.norm(light_dir)

    # Floor grid at z = 0 (drawn behind everything at zorder=0):
    grid_coords = np.linspace(-1.2, 1.2, 7)
    for g in grid_coords:
        ax.plot3D([g, g], [-1.2, 1.2], [0, 0], color="#E2E8F0", linewidth=0.7, zorder=0)
        ax.plot3D([-1.2, 1.2], [g, g], [0, 0], color="#E2E8F0", linewidth=0.7, zorder=0)

    num_bodies = world_vertices.shape[0] if world_vertices.shape else 1
    all_polygons = []
    all_face_colors = []
    for body_idx in range(num_bodies):
        world_verts = world_vertices[body_idx] if world_vertices.shape else world_vertices
        v = _extract_xyz(world_verts)  # (8, 3)

        base_color = np.array(plt.matplotlib.colors.to_rgba(BODY_COLORS[body_idx % len(BODY_COLORS)]))

        for face in faces:
            poly_v = v[face]
            all_polygons.append(poly_v)

            # Compute outward face normal for diffuse shading:
            edge1 = poly_v[1] - poly_v[0]
            edge2 = poly_v[2] - poly_v[0]
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm > 1e-8:
                normal = normal / norm
            diffuse = max(0.0, float(np.dot(normal, light_dir)))
            shade = 0.50 + 0.50 * diffuse

            shaded_color = base_color.copy()
            shaded_color[:3] *= shade
            shaded_color[3] = 1.0  # Fully opaque to prevent z-order / see-through artifacts
            all_face_colors.append(shaded_color)

    # Combine all bodies into a single collection so matplotlib sorts faces globally:
    robot_poly = Poly3DCollection(
        all_polygons,
        facecolors=all_face_colors,
        edgecolors="#1A202C",
        linewidths=0.8,
        alpha=1.0,
        zorder=2,
    )
    ax.add_collection3d(robot_poly)

    # --- Draw Camera Body & Multi-Lens Optical Train in 3D ---
    ctx = NumpyContext(PGA3D)
    mv = ctx.multivector

    # Local optical elements transformed into world space:
    def to_world(local_xyz: np.ndarray) -> np.ndarray:
        pts = mv.antivector(np.concatenate([local_xyz, np.ones_like(local_xyz[..., :1])], axis=-1))
        return _extract_xyz(camera_pose >> pts)

    # 1. Front lens aperture ring (radius 0.3 at z = 0):
    theta = np.linspace(0, 2 * np.pi, 36)
    lens_ring_front = np.stack([0.3 * np.cos(theta), 0.3 * np.sin(theta), np.zeros_like(theta)], axis=-1)
    world_lens_front = to_world(lens_ring_front)
    ax.plot3D(
        world_lens_front[:, 0], world_lens_front[:, 1], world_lens_front[:, 2],
        color="#319795", linewidth=2.0, zorder=3,
    )

    # 2. Rear lens aperture ring (radius 0.25 at z = -0.3):
    lens_ring_rear = np.stack([0.25 * np.cos(theta), 0.25 * np.sin(theta), -0.3 * np.ones_like(theta)], axis=-1)
    world_lens_rear = to_world(lens_ring_rear)
    ax.plot3D(
        world_lens_rear[:, 0], world_lens_rear[:, 1], world_lens_rear[:, 2],
        color="#805AD5", linewidth=2.0, zorder=3,
    )

    # 3. Sensor plane rectangle matching physical 1.6 x 1.2 sensor chip at z = -1.25:
    half_w = 1.6 * 0.5
    half_h = 1.2 * 0.5
    sensor_corners = np.array([
        [-half_w, -half_h, -1.25],
        [ half_w, -half_h, -1.25],
        [ half_w,  half_h, -1.25],
        [-half_w,  half_h, -1.25],
    ])
    world_sensor = to_world(sensor_corners)
    sensor_poly = Poly3DCollection([world_sensor], facecolors="#DD6B20", edgecolors="#C05621", linewidths=1.5, alpha=0.55, zorder=3)
    ax.add_collection3d(sensor_poly)

    # --- Draw Traced Optical Ray Bundle ---
    if sample_rays is None and camera_pose is not None and world_vertices is not None:
        try:
            from numga import stack
            if world_vertices.shape and len(world_vertices.shape) >= 2:
                sample_pts = stack([world_vertices[-1, 7], world_vertices[-1, 6], world_vertices[-1, 2]])
                sample_rays = trace_sample_rays(sample_pts, camera_pose)
        except Exception:
            sample_rays = None

    if sample_rays is not None:
        for idx, (p_scene, p_front, p_rear, p_sensor) in enumerate(sample_rays):
            color = RAY_COLORS[idx % len(RAY_COLORS)]
            c_scene = _extract_xyz(p_scene).ravel()
            c_front = _extract_xyz(p_front).ravel()
            c_rear = _extract_xyz(p_rear).ravel()
            c_sensor = _extract_xyz(p_sensor).ravel()

            # Leg 1: Scene point to entrance pupil:
            ax.plot3D(
                [c_scene[0], c_front[0]], [c_scene[1], c_front[1]], [c_scene[2], c_front[2]],
                color=color, linewidth=1.4, linestyle="-", alpha=0.85, zorder=4,
            )
            # Leg 2: Front lens to rear lens:
            ax.plot3D(
                [c_front[0], c_rear[0]], [c_front[1], c_rear[1]], [c_front[2], c_rear[2]],
                color=color, linewidth=1.8, linestyle="-", alpha=0.95, zorder=4,
            )
            # Leg 3: Rear lens to sensor plane:
            ax.plot3D(
                [c_rear[0], c_sensor[0]], [c_rear[1], c_sensor[1]], [c_rear[2], c_sensor[2]],
                color=color, linewidth=1.8, linestyle="-", alpha=0.95, zorder=4,
            )

            # Hit dot on sensor:
            ax.scatter3D([c_sensor[0]], [c_sensor[1]], [c_sensor[2]], color=color, s=28, edgecolors="white", linewidths=0.8, zorder=5)

    ax.set_xlim(-0.9, 1.1)
    ax.set_ylim(-4.95, 0.65)
    ax.set_zlim(0.0, 3.1)
    ax.set_box_aspect([2.0, 5.6, 3.1])
    ax._dist = 6.0
    ax.dist = 6.0
    ax.set_xlabel("x", labelpad=1, fontsize=8)
    ax.set_ylabel("y", labelpad=1, fontsize=8)
    ax.set_zlabel("z", labelpad=1, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.view_init(elev=18, azim=-55)
    ax.set_title("3d scene", fontsize=10, pad=8)


# --- 2D Sensor Photographed Image -----------------------------------------------------
def draw_camera_image(
    ax: plt.Axes,
    projected_pixels: Point,
    width: int = 640,
    height: int = 480,
    title: str = "camera sensor",
) -> None:
    """Render the 2D photographed camera image captured through the multi-lens optical train."""
    ax.clear()

    _, faces = unit_box_topology()
    num_bodies = projected_pixels.shape[0] if projected_pixels.shape else 1

    u, v = _extract_uv(projected_pixels)  # (num_bodies, 8)

    # Painter's algorithm: draw bodies from base to end-effector:
    for body_idx in range(num_bodies):
        body_u = u[body_idx]
        body_v = v[body_idx]
        color = BODY_COLORS[body_idx % len(BODY_COLORS)]

        for face in faces:
            poly_coords = np.column_stack([body_u[face], body_v[face]])
            poly = Polygon(
                poly_coords,
                closed=True,
                facecolor=color,
                edgecolor="#1E293B",
                linewidth=1.0,
                alpha=0.88,
            )
            ax.add_patch(poly)

        # Joint vertices:
        ax.scatter(body_u, body_v, color="white", s=6, edgecolors=color, linewidths=0.6, zorder=4)

    # Sensor frame boundary:
    sensor_frame = Rectangle(
        (0, 0), width, height,
        fill=False, edgecolor="#4A5568", linewidth=1.2, linestyle="-",
    )
    ax.add_patch(sensor_frame)

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Invert y so image origin (0, 0) is at top-left
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, pad=4)


# --- High-Level Figure Renderer -------------------------------------------------------
def draw_scenegraph_figure(
    world_vertices: Point | None = None,
    projected_pixels: Point | None = None,
    camera_pose: Motor | None = None,
    sample_rays: list[tuple[Point, Point, Point, Point]] | None = None,
    plot_path: str | Path | None = None,
    bodies_to_world: PointMap | None = None,
    unit_box: Point | None = None,
    links_to_world: PointMap | None = None,
) -> plt.Figure:
    """Render the combined 2-panel figure: 3D scene with optics + 2D photographed sensor image."""
    fig = plt.figure(figsize=(11.5, 5.5), dpi=130)

    # Left panel: 3D World Scene (generously sized to eliminate whitespace)
    ax_3d = fig.add_axes([-0.05, -0.04, 0.63, 1.05], projection="3d")
    draw_scenegraph_scene_3d(
        ax_3d,
        world_vertices=world_vertices,
        camera_pose=camera_pose,
        sample_rays=sample_rays,
        bodies_to_world=bodies_to_world if bodies_to_world is not None else links_to_world,
        unit_box=unit_box,
    )

    # Right panel: 2D Sensor Photograph
    ax_2d = fig.add_axes([0.56, 0.08, 0.41, 0.84])
    draw_camera_image(ax_2d, projected_pixels)

    if plot_path:
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, bbox_inches="tight", dpi=140)

    return fig


# --- Kinematic Animation --------------------------------------------------------------
def animate_scenegraph(
    frames_generator: Iterable,
    camera_pose: Motor,
    output_gif_path: str | Path = str(PLOT_DIR / "scenegraph.gif"),
    duration_ms: int = 60,
    unit_box: Point | None = None,
) -> str:
    """Animate frames yielded by a generator and export to GIF."""
    fig = plt.figure(figsize=(11.5, 5.5), dpi=100)
    ax_3d = fig.add_axes([-0.05, -0.04, 0.63, 1.05], projection="3d")
    ax_2d = fig.add_axes([0.56, 0.08, 0.41, 0.84])

    frames = []
    for frame_data in frames_generator:
        if len(frame_data) == 3:
            first_item, frame_projected, frame_rays = frame_data
        else:
            first_item, frame_projected = frame_data
            frame_rays = None
        # If first_item is already world_vertices ([5, 8] Point), pass it directly:
        if hasattr(first_item, "gatype") and first_item.gatype == Point:
            draw_scenegraph_scene_3d(ax_3d, world_vertices=first_item, camera_pose=camera_pose, sample_rays=frame_rays)
        else:
            draw_scenegraph_scene_3d(ax_3d, bodies_to_world=first_item, camera_pose=camera_pose, sample_rays=frame_rays, unit_box=unit_box)
        draw_camera_image(ax_2d, frame_projected, title="camera sensor")
        frames.append(capture(fig))

    plt.close(fig)
    return save_gif(frames, str(output_gif_path), duration_ms=duration_ms)


def animate_robot_kinematics(
    output_gif_path: str | Path = str(PLOT_DIR / "scenegraph.gif"),
    num_frames: int = 48,
    duration_ms: int = 60,
) -> str:
    """Generate and export a smooth looping GIF of forward kinematics and camera rendering."""
    camera_pose = make_camera_pose()
    camera = make_multi_lens_camera(camera_pose)
    viewport = make_viewport()
    unit_box = canonical_unit_box()

    def frames_gen():
        t_vals = np.linspace(0, 2 * np.pi, num_frames, endpoint=False)
        for t in t_vals:
            joint_angles = (
                0.45 * np.sin(t),
                -0.40 + 0.25 * np.cos(t),
                0.85 + 0.35 * np.sin(t),
                -0.45 + 0.25 * np.cos(2 * t),
            )
            bodies_to_world, _ = make_robot_arm(joint_angles)
            local_to_pixel = collapse_scenegraph(bodies_to_world, camera, viewport)
            projected = project_vertices(local_to_pixel, unit_box)
            gripper_tip = bodies_to_world[-1](unit_box[7])
            sample_rays = trace_sample_rays(gripper_tip, camera_pose)
            yield bodies_to_world, projected, sample_rays

    return animate_scenegraph(
        frames_gen(),
        camera_pose=camera_pose,
        output_gif_path=output_gif_path,
        duration_ms=duration_ms,
        unit_box=unit_box,
    )
