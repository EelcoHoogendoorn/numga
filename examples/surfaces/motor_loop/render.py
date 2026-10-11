"""Refined surfaces over their control nets, with the frames of the motors."""

from __future__ import annotations

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np

from examples.animation import capture
from examples.surfaces.motor_loop import core
from examples.surfaces.projective_loop.render import SURFACE_COLOUR, edges

AXIS_COLOURS = np.array([[0.85, 0.15, 0.15], [0.2, 0.65, 0.2], [0.15, 0.3, 0.85]])
CONTROL_COLOUR = "0.15"
ELEVATION, AZIMUTH = 12.0, -60.0
# The direction towards the eye at that elevation and azimuth.
EYE = np.array([np.cos(np.radians(ELEVATION)) * np.cos(np.radians(AZIMUTH)),
                np.cos(np.radians(ELEVATION)) * np.sin(np.radians(AZIMUTH)), np.sin(np.radians(ELEVATION))])


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """x, y and z of each point at the sites of a field, on the last axis."""
    xyzw = points.batch().cast(core.ga.subspace("yzw zxw xyw zyx")).kernel
    return xyzw[..., :3] / xyzw[..., 3:]


def reach(points: core.Point) -> np.ndarray:
    """The mean distance from the z axis of the points at the sites of a field, per case."""
    xyz = coordinates(points)
    return np.hypot(xyz[..., 0], xyz[..., 1]).mean(axis=-1)


def separation(first: core.Point, second: core.Point) -> float:
    """The largest distance between where two fields put the same site."""
    return float(np.linalg.norm(coordinates(first) - coordinates(second), axis=-1).max())


def frame_axes(ax: plt.Axes, xyz: np.ndarray) -> None:
    centre, half = (xyz.max(axis=0) + xyz.min(axis=0)) / 2, np.ptp(xyz, axis=0).max() / 2
    ax.set(xlim=(centre[0] - half, centre[0] + half), ylim=(centre[1] - half, centre[1] + half), zlim=(centre[2] - half, centre[2] + half))
    ax.set_box_aspect((1, 1, 1), zoom=1.3)
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)
    ax.set_axis_off()


def draw_shaded(ax: plt.Axes, xyz: np.ndarray, faces: np.ndarray) -> None:
    """The faces lit from the eye, either side alike, each face's edges in its own colour so the faces join
    without seams."""
    triangles = xyz[faces]                                                     # [faces, 3, 3]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    facing = np.abs(normals @ EYE) / np.linalg.norm(normals, axis=-1)
    shaded = np.clip(SURFACE_COLOUR * (0.35 + 0.75 * facing[:, None]), 0, 1)
    ax.add_collection3d(Poly3DCollection(triangles, facecolors=shaded, edgecolors=shaded, linewidths=0.4, zorder=1))


def draw_net(ax: plt.Axes, xyz: np.ndarray, faces: np.ndarray) -> None:
    """The control net, dashed, over the surface."""
    ax.add_collection3d(Line3DCollection(xyz[edges(faces)], colors=CONTROL_COLOUR, linestyles="--", linewidths=1.1, zorder=2))


def draw_frames(ax: plt.Axes, xyz: np.ndarray, width: float) -> None:
    """Frames `[axes + 1, sites, 3]`, their origins first: each axis from the origin, x red, y green, z blue."""
    segments = np.stack([np.broadcast_to(xyz[0], xyz[1:].shape), xyz[1:]], axis=2).reshape(-1, 2, 3)   # [axes * sites, ends, 3]
    colours = np.repeat(AXIS_COLOURS, xyz.shape[1], axis=0)
    ax.add_collection3d(Line3DCollection(segments, colors=colours, linewidths=width, zorder=3))


def animate(control: core.Point, surfaces: core.Point, control_faces: np.ndarray, faces: np.ndarray) -> list[np.ndarray]:
    """Per frame, the surfaces of each scheme side by side, each over the control net, dashed, and the
    control frames."""
    frames, surfaces = coordinates(control), coordinates(surfaces)           # [frames, axes + 1, vertices, 3], [frames, schemes, refined, 3]
    extent = frames[:, 0].reshape(-1, 3)

    def frame(index: int) -> np.ndarray:
        figure = plt.figure(figsize=(8.0, 5.0), layout="constrained")
        # Drawn in order, surface, net, frames, rather than sorted by depth, so the net and frames stay on top.
        axes = [figure.add_subplot(1, 2, column + 1, projection="3d", computed_zorder=False) for column in range(surfaces.shape[1])]
        for ax, surface in zip(axes, surfaces[index]):
            draw_shaded(ax, surface, faces)
            draw_net(ax, frames[index, 0], control_faces)
            draw_frames(ax, frames[index], 2.0)
            frame_axes(ax, extent)
        pixels = capture(figure)
        plt.close(figure)
        return pixels
    return [frame(index) for index in range(surfaces.shape[0])]
