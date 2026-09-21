"""Drawing and coordinate read-out for Garland–Heckbert QEM.

This module consumes the geometry that `core` produces and turns it into a figure.
It constructs viewport geometry only, never scene geometry, and the mathematics
never imports it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from examples.geometry.qem import core
from examples.geometry.qem.core import Point, Quadric

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def euclidean(points: Point) -> np.ndarray:
    """Read Cartesian coordinates from homogeneous points."""
    values = points.cast(core.ga.subspace("yzw zxw xyw zyx")).kernel
    with np.errstate(divide="ignore", invalid="ignore"):
        return values[..., :3] / values[..., 3:]


def direction(coords: np.ndarray) -> Point:
    """Construct ideal direction points in an explicitly named basis."""
    return core.mv("yzw zxw xyw", coords)


def draw_mesh_triangles(
    ax: Axes,
    triangles: np.ndarray,
    colors: list[str],
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

    dir_pts = direction(dirs)
    quad_vals = np.maximum((quadric(dir_pts) & dir_pts).to_array(), 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(quad_vals > 1e-8, epsilon / np.sqrt(np.maximum(quad_vals, 1e-12)), max_radius)
    r = np.clip(r, 0.0, max_radius)

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


def draw_vertex_panel(
    ax: Axes,
    triangles: np.ndarray,
    vertex_coord: np.ndarray,
    quadric: Quadric,
    title: str,
    colors: list[str],
    ellipsoid_color: str,
) -> None:
    """Draw a vertex with its incident mesh triangles and error quadric ellipsoid."""
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


def draw_collapse_panel(
    ax: Axes,
    original_triangles: np.ndarray,
    simplified_triangles: np.ndarray,
    edge_coords: np.ndarray,
    quadric: Quadric,
    optimal_vertex: Point,
    title: str,
    original_colors: list[str],
    simplified_colors: list[str],
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


def draw_qem_figure(
    panel_a: tuple[np.ndarray, np.ndarray, Quadric, str, list[str], str],
    panel_b: tuple[np.ndarray, np.ndarray, Quadric, str, list[str], str],
    panel_edge: tuple[np.ndarray, np.ndarray, np.ndarray, Quadric, Point, str, list[str], list[str]],
    plot_path: Path,
) -> plt.Figure:
    """Render and save the three-panel QEM figure."""
    fig = plt.figure(figsize=(19.0, 5.8), dpi=140, layout="constrained")

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    draw_vertex_panel(ax1, *panel_a)

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    draw_vertex_panel(ax2, *panel_b)

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    draw_collapse_panel(ax3, *panel_edge)

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, bbox_inches="tight")
    print(f"Figure saved to {plot_path}")
    return fig
