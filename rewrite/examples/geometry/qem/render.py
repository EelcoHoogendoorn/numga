"""Drawing and coordinate read-out for Garland–Heckbert QEM.

This module consumes the geometry that `core` produces and turns it into a figure.
It constructs viewport geometry only, never scene geometry, and the mathematics
never imports it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from examples.geometry.qem.core import Point, Quadric, direction, ga

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# Blue: shared flanking ridge triangles (faces 0, 1)
# Red/Rose: steep corner triangles at vertex a (faces 2, 3)
# Purple/Violet: gentle ramp triangles at vertex b (faces 4, 5)
FACE_COLORS = np.array(["#0ea5e9", "#0284c7", "#f43f5e", "#e11d48", "#8b5cf6", "#7c3aed"])


def euclidean(points: Point) -> np.ndarray:
    """Read Cartesian coordinates from homogeneous points."""
    values = points.cast(ga.subspace("yzw zxw xyw zyx")).kernel
    return values[..., :3] / values[..., 3:]


def draw_mesh_triangles(
    ax: Axes,
    triangles: np.ndarray,
    colors: np.ndarray,
    alpha: float,
    edgecolor: str,
    linewidth: float,
) -> None:
    """Draw shaded triangular mesh facets with visible boundary edges."""
    poly = Poly3DCollection(
        triangles,
        alpha=alpha,
        edgecolor=edgecolor,
        linewidths=linewidth,
    )
    poly.set_facecolor(colors)
    ax.add_collection3d(poly)


def draw_error_ellipsoid(
    ax: Axes,
    quadric: Quadric,
    center: np.ndarray,
    epsilon: float,
    max_radius: float,
    color: str,
    alpha: float,
) -> None:
    """Draw the error quadric isosurface Q(x) & x <= epsilon^2 around center."""
    theta = np.linspace(0.0, np.pi, 17)
    phi = np.linspace(0.0, 2.0 * np.pi, 25)
    dx = np.sin(theta)[:, None] * np.cos(phi)[None, :]
    dy = np.sin(theta)[:, None] * np.sin(phi)[None, :]
    dz = np.cos(theta)[:, None] * np.ones_like(phi)[None, :]
    dirs = np.stack([dx, dy, dz], axis=-1)

    # The error grows quadratically along each direction; where it stays flat the radius
    # is capped at max_radius.
    dir_pts = direction(dirs)
    quad_vals = (quadric(dir_pts) & dir_pts).to_array()
    r = epsilon / np.sqrt(np.maximum(quad_vals, (epsilon / max_radius) ** 2))

    surface = center + r[..., None] * dirs
    ax.plot_wireframe(
        surface[..., 0],
        surface[..., 1],
        surface[..., 2],
        color=color,
        alpha=alpha,
        rstride=1,
        cstride=1,
        linewidth=0.6,
    )


def style_axes(ax: Axes, title: str) -> None:
    """Shared view, limits and labels of the three panels."""
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_box_aspect([1.8, 1.4, 0.75])
    ax.set_xlim(-0.9, 0.9)
    ax.set_ylim(-0.7, 0.7)
    ax.set_zlim(-0.3, 0.45)
    ax.set_xlabel("x", labelpad=-5, fontsize=8)
    ax.set_ylabel("y", labelpad=-5, fontsize=8)
    ax.set_zlabel("z", labelpad=-5, fontsize=8)
    ax.view_init(elev=24, azim=48)
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.7)


def draw_vertex_panel(
    ax: Axes,
    triangles: np.ndarray,
    vertex: Point,
    quadric: Quadric,
    title: str,
    colors: np.ndarray,
    ellipsoid_color: str,
) -> None:
    """Draw a vertex with its incident mesh triangles and error quadric ellipsoid."""
    vertex_coord = euclidean(vertex)
    draw_mesh_triangles(ax, triangles, colors, 0.45, "#1e293b", 1.2)
    draw_error_ellipsoid(ax, quadric, vertex_coord, 0.12, 0.4, ellipsoid_color, 0.4)
    ax.scatter(
        [vertex_coord[0]],
        [vertex_coord[1]],
        [vertex_coord[2]],
        color="#0f172a",
        s=70,
        depthshade=False,
        label="Mesh vertex",
    )
    style_axes(ax, title)


def draw_collapse_panel(
    ax: Axes,
    original_triangles: np.ndarray,
    simplified_triangles: np.ndarray,
    edge_coords: np.ndarray,
    quadric: Quadric,
    optimal_vertex: Point,
    title: str,
    original_colors: np.ndarray,
    simplified_colors: np.ndarray,
) -> None:
    """Draw the edge collapse showing contracting edge, simplified mesh, and optimal vertex."""
    opt_xyz = euclidean(optimal_vertex)

    # Original mesh faintly in background:
    draw_mesh_triangles(ax, original_triangles, original_colors, 0.12, "#94a3b8", 0.8)

    # Simplified mesh with contracted vertex:
    draw_mesh_triangles(ax, simplified_triangles, simplified_colors, 0.55, "#0f172a", 1.4)

    # Contracting edge:
    ax.plot(
        edge_coords[:, 0],
        edge_coords[:, 1],
        edge_coords[:, 2],
        color="#ef4444",
        linewidth=2.8,
        linestyle="--",
        label="Contracted edge (a, b)",
    )

    ax.scatter(
        edge_coords[:, 0],
        edge_coords[:, 1],
        edge_coords[:, 2],
        color="#475569",
        s=50,
        depthshade=False,
        label="Original endpoints",
    )

    ax.scatter(
        [opt_xyz[0]],
        [opt_xyz[1]],
        [opt_xyz[2]],
        color="#10b981",
        marker="*",
        s=260,
        depthshade=False,
        label="Optimal vertex v_edge",
    )

    draw_error_ellipsoid(ax, quadric, opt_xyz, 0.12, 0.4, "#10b981", 0.35)
    style_axes(ax, title)


def draw_qem(
    vertices: Point,
    faces: np.ndarray,
    incident_a: np.ndarray,
    incident_b: np.ndarray,
    qa: Quadric,
    qb: Quadric,
    q_edge: Quadric,
    v_edge: Point,
    collapsed: Point,
    surviving: np.ndarray,
) -> plt.Figure:
    """Both endpoint neighbourhoods with their error ellipsoids, and the collapsed mesh."""
    coords = euclidean(vertices)
    triangles = coords[faces]
    fig = plt.figure(figsize=(19.0, 5.8), dpi=140, layout="constrained")

    draw_vertex_panel(
        fig.add_subplot(1, 3, 1, projection="3d"),
        triangles[incident_a], vertices[0], qa,
        "1. Vertex a Neighborhood (Sharp Corner)\nFlanking ridge faces (blue) + end faces (red)",
        FACE_COLORS[incident_a], "#f43f5e",
    )
    draw_vertex_panel(
        fig.add_subplot(1, 3, 2, projection="3d"),
        triangles[incident_b], vertices[1], qb,
        "2. Vertex b Neighborhood (Crease Transition)\nShared ridge faces (blue) + transition ramp (purple)",
        FACE_COLORS[incident_b], "#8b5cf6",
    )
    draw_collapse_panel(
        fig.add_subplot(1, 3, 3, projection="3d"),
        triangles, euclidean(collapsed)[faces[surviving]], coords[:2], q_edge, v_edge,
        "3. Simplified Mesh (Edge Contraction)\nFlanking faces vanish; optimal vertex minimizes Q_edge",
        FACE_COLORS, FACE_COLORS[surviving],
    )
    return fig
