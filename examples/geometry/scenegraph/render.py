"""Drawing and coordinate read-out for the scenegraph example.

Figures show the robot arm and the camera's optics in the world, and the photograph on
the camera's sensor. The mathematics never imports this module.
"""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from examples.animation import capture
from examples.geometry.scenegraph.core import Motor, Point, ga, point, unit_box_topology


# Colors for the robot arm bodies:
BODY_COLORS = [
    "#4A5568",  # Body 0: Base pedestal (charcoal slate)
    "#2B6CB0",  # Body 1: Turret (steel blue)
    "#3182CE",  # Body 2: Upper arm (vibrant azure)
    "#4299E1",  # Body 3: Forearm (sky blue)
    "#ED8936",  # Body 4: Gripper wrist (accent amber)
]

RAY_COLORS = [
    "#E53E3E",  # Red / crimson
    "#38A169",  # Green / emerald
    "#D69E2E",  # Golden amber
]


def euclidean(points: Point) -> np.ndarray:
    """Read xyz coordinates using an explicit coordinate basis, independent of storage order."""
    k = points.cast(ga.subspace("yzw zxw xyw zyx")).kernel
    return k[..., :3] / k[..., 3:]


# --- 3D scene view ----------------------------------------------------------------------
def plot_scene_3d(ax: Axes3D, world_vertices: Point, camera_pose: Motor, rays: tuple) -> None:
    """The 3D world scene: articulated robot arm, camera lenses and sensor, and traced rays."""
    ax.clear()
    _, faces = unit_box_topology()

    # Directional light source for shading 3D body faces:
    light_dir = np.array([0.4, -0.6, 0.7])
    light_dir = light_dir / np.linalg.norm(light_dir)

    # Floor grid in the plane z == 0 (drawn behind everything at zorder=0):
    for g in np.linspace(-1.2, 1.2, 7):
        ax.plot3D([g, g], [-1.2, 1.2], [0, 0], color="#E2E8F0", linewidth=0.7, zorder=0)
        ax.plot3D([-1.2, 1.2], [g, g], [0, 0], color="#E2E8F0", linewidth=0.7, zorder=0)

    # All faces of all bodies in one collection so matplotlib sorts them globally, each
    # shaded by the diffuse light on its outward normal:
    polygons = euclidean(world_vertices)[:, faces]                  # [bodies, faces, face_corners, 3]
    normals = np.cross(polygons[..., 1, :] - polygons[..., 0, :], polygons[..., 2, :] - polygons[..., 0, :])
    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    shade = 0.50 + 0.50 * np.maximum(normals @ light_dir, 0.0)       # [bodies, faces]
    base = np.array([mcolors.to_rgb(c) for c in BODY_COLORS])[:, None, :]
    face_colors = np.concatenate([base * shade[..., None], np.ones((*shade.shape, 1))], axis=-1)
    ax.add_collection3d(Poly3DCollection(
        polygons.reshape(-1, 4, 3),
        facecolors=face_colors.reshape(-1, 4),
        edgecolors="#1A202C",
        linewidths=0.8,
        alpha=1.0,
        zorder=2,
    ))

    # Optical elements in the camera frame, moved into the world by the camera pose:
    theta = np.linspace(0, 2 * np.pi, 36)
    ring = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=-1)
    # The front lens rim, radius 0.3 at z == 0, and the rear lens rim, radius 0.25 at z == -0.3:
    front_ring = euclidean(camera_pose >> point(ring * 0.3))                       # [theta, 3]
    rear_ring = euclidean(camera_pose >> point(ring * 0.25 + [0.0, 0.0, -0.3]))   # [theta, 3]
    ax.plot3D(front_ring[:, 0], front_ring[:, 1], front_ring[:, 2], color="#319795", linewidth=2.0, zorder=3)
    ax.plot3D(rear_ring[:, 0], rear_ring[:, 1], rear_ring[:, 2], color="#805AD5", linewidth=2.0, zorder=3)

    # Sensor plane rectangle matching the physical 1.6 x 1.2 sensor chip at z == -1.25:
    sensor = euclidean(camera_pose >> point(np.array([
        [-0.8, -0.6, -1.25],
        [ 0.8, -0.6, -1.25],
        [ 0.8,  0.6, -1.25],
        [-0.8,  0.6, -1.25],
    ])))
    ax.add_collection3d(Poly3DCollection([sensor], facecolors="#DD6B20", edgecolors="#C05621", linewidths=1.5, alpha=0.55, zorder=3))

    # Traced rays: scene point to pupil, pupil to rear lens, rear lens to sensor.
    stages = np.stack([euclidean(stage) for stage in rays], axis=-2)      # [rays, stages, 3]
    for path, color in zip(stages, RAY_COLORS):
        for leg, width in zip(range(3), (1.4, 1.8, 1.8)):
            ax.plot3D(*path[leg:leg + 2].T, color=color, linewidth=width, alpha=0.9, zorder=4)
        ax.scatter3D(*path[3:].T, color=color, s=28, edgecolors="white", linewidths=0.8, zorder=5)

    ax.set_xlim(-0.9, 1.1)
    ax.set_ylim(-4.95, 0.65)
    ax.set_zlim(0.0, 3.1)
    ax.set_box_aspect([2.0, 5.6, 3.1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=18, azim=-55)


# --- 2D sensor photograph ---------------------------------------------------------------
def plot_camera_image(ax: plt.Axes, projected_pixels: Point) -> None:
    """The photograph on the 640 x 480 pixel sensor, bodies painted from base to end-effector."""
    ax.clear()
    width, height = 640, 480
    _, faces = unit_box_topology()
    pixels = euclidean(projected_pixels)[..., :2]                  # [bodies, vertices, 2]

    for body, color in zip(pixels, BODY_COLORS):
        for face in faces:
            ax.add_patch(Polygon(body[face], closed=True, facecolor=color, edgecolor="#1E293B", linewidth=1.0, alpha=0.88))
        ax.scatter(body[:, 0], body[:, 1], color="white", s=6, edgecolors=color, linewidths=0.6, zorder=4)

    # Sensor frame boundary:
    ax.add_patch(Rectangle((0, 0), width, height, fill=False, edgecolor="#4A5568", linewidth=1.2, linestyle="-"))
    ax.set_xlim(0, width)
    # Invert y so image origin (0, 0) is at top-left:
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.axis("off")


# --- figures ----------------------------------------------------------------------------
def draw_scene_3d(world_vertices: Point, camera_pose: Motor, rays: tuple) -> plt.Figure:
    """The 3D scene on its own."""
    fig = plt.figure(figsize=(7.5, 6.0), dpi=140)
    plot_scene_3d(fig.add_subplot(1, 1, 1, projection="3d"), world_vertices, camera_pose, rays)
    fig.tight_layout()
    return fig


def draw_camera_image(projected_pixels: Point) -> plt.Figure:
    """The sensor photograph on its own."""
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=140)
    plot_camera_image(ax, projected_pixels)
    fig.tight_layout()
    return fig


def scenegraph_axes(dpi: int):
    """Side-by-side 3D scene and sensor photograph."""
    fig = plt.figure(figsize=(11.5, 5.5), dpi=dpi)
    ax_3d = fig.add_axes([-0.05, -0.04, 0.63, 1.05], projection="3d")
    ax_2d = fig.add_axes([0.56, 0.08, 0.41, 0.84])
    return fig, ax_3d, ax_2d


def draw_scenegraph(world_vertices: Point, projected_pixels: Point, camera_pose: Motor, rays: tuple) -> plt.Figure:
    """The 3D scene with its optics beside the photograph on the sensor."""
    fig, ax_3d, ax_2d = scenegraph_axes(140)
    plot_scene_3d(ax_3d, world_vertices, camera_pose, rays)
    plot_camera_image(ax_2d, projected_pixels)
    return fig


def animate_scenegraph(frames: Iterable) -> list[np.ndarray]:
    """RGB frames of the two-panel figure, one per state."""
    fig, ax_3d, ax_2d = scenegraph_axes(100)
    images = []
    for world_vertices, projected_pixels, camera_pose, rays in frames:
        plot_scene_3d(ax_3d, world_vertices, camera_pose, rays)
        plot_camera_image(ax_2d, projected_pixels)
        images.append(capture(fig))
    plt.close(fig)
    return images
