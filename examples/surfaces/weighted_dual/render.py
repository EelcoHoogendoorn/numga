"""The hull and its polar dual in space, and both projected onto the sphere."""

from __future__ import annotations

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np

from examples.animation import capture
from examples.surfaces.weighted_dual import core

PRIMAL_COLOUR = "#287fa0"
DUAL_COLOUR = "#d06c36"
SPHERE_MESH = 48
ARC_SAMPLES = 16
REACH = 1.3
ZOOM = 1.7
# The colours of a vertex's distance from the sphere, and the distance at either end of them.
DISTANCE_MAP = "coolwarm"
DISTANCE_RANGE = 0.1
# How far each ray is drawn either side of the sphere.
RAY_SPAN = 0.15
ELEVATION, AZIMUTH = 20.0, -60.0


# --- plumbing -------------------------------------------------------------------------
def scalars(values: core.Scalar) -> np.ndarray:
    return values.cast(values.algebra.subspace.scalar()).kernel[..., 0]


def coordinates(points: core.Vector) -> np.ndarray:
    """x, y and z of each point, on the last axis."""
    return points.cast(core.ga.subspace("x y z")).kernel


def arcs(ends: np.ndarray) -> np.ndarray:
    """Great arcs on the unit sphere between the directions of pairs of points, [pairs, 2, 3]."""
    share = np.linspace(0, 1, ARC_SAMPLES)[:, None]
    chords = ends[:, None, 0] * (1 - share) + ends[:, None, 1] * share         # [pairs, samples, 3]
    return chords / np.linalg.norm(chords, axis=-1, keepdims=True)


def sphere(ax: plt.Axes) -> None:
    longitude, latitude = np.meshgrid(np.linspace(0, 2 * np.pi, SPHERE_MESH), np.linspace(-np.pi / 2, np.pi / 2, SPHERE_MESH))
    ax.plot_surface(np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude),
                    color="0.85", alpha=0.15, linewidth=0, shade=False)


def frame_axes(ax: plt.Axes) -> None:
    """One framing for both panels, so the sphere is the same size in each."""
    ax.set(xlim=(-REACH, REACH), ylim=(-REACH, REACH), zlim=(-REACH, REACH))
    ax.set_box_aspect((1, 1, 1), zoom=ZOOM)
    ax.set_proj_type("ortho")
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)
    ax.set_axis_off()


def facing() -> np.ndarray:
    """The direction towards the viewer."""
    elevation, turn = np.radians(ELEVATION), np.radians(AZIMUTH)
    return np.array([np.cos(elevation) * np.cos(turn), np.cos(elevation) * np.sin(turn), np.sin(elevation)])


def front(paths: np.ndarray) -> np.ndarray:
    """Paths on the sphere with the parts behind it, seen from the viewer, removed."""
    return np.where((paths @ facing() > 0)[..., None], paths, np.nan)


def draw_complex(points: core.Vector, faces: np.ndarray, duals: core.Vector) -> plt.Figure:
    """In space, the polyhedron of the points and its polar dual, the parts facing away from the viewer
    culled; on the sphere, the half facing the viewer of both projected from the centre."""
    xyz, poles = coordinates(points), coordinates(duals)                       # [points, 3], [faces, 3]
    edges, beside = core.adjacency(faces)                                      # [edges, 2] each
    corners = xyz[faces]                                                       # [faces, 3, 3]
    visible = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]) @ facing() > 0   # [faces]
    # A dual face is the polar plane of a hull vertex, facing along it; a dual edge is in front where
    # either of its two dual faces is.
    dual_visible = (xyz[edges] @ facing() > 0).any(axis=-1)                   # [edges]
    figure = plt.figure(figsize=(10.0, 5.0))
    space = figure.add_axes((0.0, 0.0, 0.5, 1.0), projection="3d")
    sphere(space)
    space.add_collection3d(Poly3DCollection(corners[visible], facecolor=PRIMAL_COLOUR, alpha=0.15, edgecolor=PRIMAL_COLOUR, linewidth=0.8))
    space.add_collection3d(Line3DCollection(poles[beside[dual_visible]], colors=DUAL_COLOUR, linewidths=0.9))
    # Each vertex coloured by its distance from the sphere, blue inside, grey on it, red outside, on a
    # stretch of the ray from the centre it moves along.
    shown = xyz[xyz @ facing() > 0]
    rays = shown / np.linalg.norm(shown, axis=-1, keepdims=True)
    space.add_collection3d(Line3DCollection(np.stack([rays * (1 - RAY_SPAN), rays * (1 + RAY_SPAN)], axis=1), colors="0.55", linewidths=0.6))
    space.scatter(*shown.T, c=np.linalg.norm(shown, axis=-1) - 1, cmap=DISTANCE_MAP, vmin=-DISTANCE_RANGE, vmax=DISTANCE_RANGE,
                  s=22, depthshade=False, edgecolors="none")
    frame_axes(space)

    image = figure.add_axes((0.5, 0.0, 0.5, 1.0), projection="3d")
    sphere(image)
    on_sphere = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
    duals_on_sphere = poles / np.linalg.norm(poles, axis=-1, keepdims=True)
    image.add_collection3d(Line3DCollection(front(arcs(on_sphere[edges])), colors=PRIMAL_COLOUR, linewidths=0.9))
    image.add_collection3d(Line3DCollection(front(arcs(duals_on_sphere[beside])), colors=DUAL_COLOUR, linewidths=0.9))
    frame_axes(image)
    return figure


def animate(frames: list[tuple[core.Vector, np.ndarray, core.Vector]]) -> list[np.ndarray]:
    """The complex at every frame, from one view."""

    def picture(geometry: tuple) -> np.ndarray:
        figure = draw_complex(*geometry)
        pixels = capture(figure)
        plt.close(figure)
        return pixels
    return [picture(geometry) for geometry in frames]
