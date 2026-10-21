"""A Koebe polyhedron and its dual about the sphere they both touch, and their circles on the sphere."""

from __future__ import annotations

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

from numga import stack
from examples.animation import capture
from examples.surfaces.koebe import core

PRIMAL_COLOUR = "#287fa0"
DUAL_COLOUR = "#d06c36"
SPHERE_MESH = 48
CIRCLE_SAMPLES = 64
REACH = 1.35
ZOOM = 1.6
ELEVATION, AZIMUTH = 20.0, -60.0


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """x, y and z of each point on the last axis: its pairings with the planes x, y and z, over its weight.
    A point here is the dual of a plane, and w squares to minus one, so its weight is its pairing with -w."""
    mv = core.mv
    return ((stack((mv.x, mv.y, mv.z)) & points[..., None]) / (-mv.w & points)[..., None]).to_array()


def edge_distances(apexes: core.Point, edges: np.ndarray) -> np.ndarray:
    """The distance from the centre to the line of each edge between apexes."""
    xyz = coordinates(apexes)
    ends = xyz[..., edges[:, 0], :], xyz[..., edges[:, 1], :]
    return np.linalg.norm(np.cross(*ends), axis=-1) / np.linalg.norm(ends[1] - ends[0], axis=-1)


def beside(faces: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """The two faces either side of each edge, [edges, 2]."""
    sides = np.sort(np.stack([faces, np.roll(faces, -1, axis=1)], axis=-1), axis=-1).reshape(-1, 2)
    order = np.lexsort(sides.T[::-1])
    return (order // 3).reshape(-1, 2)[np.argsort(np.lexsort(edges.T[::-1]))]


def circles_on_sphere(circles: core.Vector) -> np.ndarray:
    """Points around each circle on the unit sphere, [..., samples, 3]: the plane `n | x == s` of a circle
    `n + s * w` cuts the sphere at the distance `s / |n|` along `n`."""
    mv = core.mv
    # A circle's normal part, its inner products with x, y and z, which square to one, and its part along w,
    # which squares to minus one.
    along = (circles[..., None] | stack((mv.x, mv.y, mv.z))).to_array()      # [..., 3]
    offset = -(circles | mv.w).to_array()                                      # [...]
    normal = along / np.linalg.norm(along, axis=-1, keepdims=True)
    height = offset / np.linalg.norm(along, axis=-1)
    side = np.cross(normal, np.where(np.abs(normal[..., :1]) < 0.9, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]))
    side /= np.linalg.norm(side, axis=-1, keepdims=True)
    other = np.cross(normal, side)
    angle = np.linspace(0, 2 * np.pi, CIRCLE_SAMPLES)[:, None]
    spread = np.sqrt(1 - height**2)[..., None, None]
    return (height[..., None, None] * normal[..., None, :]
            + spread * (np.cos(angle) * side[..., None, :] + np.sin(angle) * other[..., None, :]))


def facing() -> np.ndarray:
    elevation, turn = np.radians(ELEVATION), np.radians(AZIMUTH)
    return np.array([np.cos(elevation) * np.cos(turn), np.cos(elevation) * np.sin(turn), np.sin(elevation)])


def sphere(ax: plt.Axes) -> None:
    longitude, latitude = np.meshgrid(np.linspace(0, 2 * np.pi, SPHERE_MESH), np.linspace(-np.pi / 2, np.pi / 2, SPHERE_MESH))
    ax.plot_surface(np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude),
                    color="0.85", alpha=0.18, linewidth=0, shade=False)


def frame_axes(ax: plt.Axes) -> None:
    ax.set(xlim=(-REACH, REACH), ylim=(-REACH, REACH), zlim=(-REACH, REACH))
    ax.set_box_aspect((1, 1, 1), zoom=ZOOM)
    ax.set_proj_type("ortho")
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)
    ax.set_axis_off()


def draw(circles: core.Vector, faces: np.ndarray, edges: np.ndarray) -> plt.Figure:
    """Left, the polyhedron of the circles' apexes and its dual, the polyhedron of the face circles' apexes,
    every edge of both touching the sphere where the two circles of its ends touch; right, the vertex circles
    and the face circles on the sphere. Only the near side of each is drawn."""
    touch = coordinates(core.touching(circles, edges).dual())                  # [edges, 3]
    near = touch @ facing() > 0                                                # [edges]
    apexes = coordinates(core.apexes(circles))                                 # [vertices, 3]
    duals = core.face_circles(circles, faces)                                  # [faces] Vector
    dual_apexes = coordinates(core.apexes(duals))                              # [faces, 3]
    figure = plt.figure(figsize=(10.0, 5.0))
    space = figure.add_axes((0.0, 0.0, 0.5, 1.0), projection="3d")
    sphere(space)
    space.add_collection3d(Line3DCollection(apexes[edges[near]], colors=PRIMAL_COLOUR, linewidths=1.2))
    space.add_collection3d(Line3DCollection(dual_apexes[beside(faces, edges)[near]], colors=DUAL_COLOUR, linewidths=1.0))
    space.scatter(*touch[near].T, color="0.25", s=6, depthshade=False)
    frame_axes(space)
    pattern = figure.add_axes((0.5, 0.0, 0.5, 1.0), projection="3d")
    sphere(pattern)
    for loops, colour in ((circles_on_sphere(circles), PRIMAL_COLOUR), (circles_on_sphere(duals), DUAL_COLOUR)):
        pieces = np.stack([loops[..., :-1, :], loops[..., 1:, :]], axis=-2).reshape(-1, 2, 3)   # [circles * samples, ends, 3]
        pattern.add_collection3d(Line3DCollection(pieces[(pieces @ facing() > 0).all(axis=-1)], colors=colour, linewidths=1.0))
    frame_axes(pattern)
    return figure


def animate(circles: core.Vector, faces: np.ndarray, edges: np.ndarray) -> list[np.ndarray]:
    """The figure at every frame of a batch of circles."""

    def frame(index: int) -> np.ndarray:
        figure = draw(circles[index], faces, edges)
        pixels = capture(figure)
        plt.close(figure)
        return pixels
    return [frame(index) for index in range(circles.shape[0])]
