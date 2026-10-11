"""Shaded surfaces over their control meshes, and the gap between two refinements."""

from __future__ import annotations

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np

from examples.animation import capture
from examples.surfaces.projective_loop import core

SURFACE_COLOUR = np.array([0.16, 0.5, 0.63])
GAP_COLOUR = np.array([0.82, 0.42, 0.21])
CONTROL_COLOUR = "0.45"
LIGHT = np.array([0.4, -0.5, 0.8])
ELEVATION, AZIMUTH = 18.0, -60.0


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """x, y and z of each point at the sites of a field, on the last axis."""
    xyzw = points.batch().cast(core.ga.subspace("yzw zxw xyw zyx")).kernel
    return xyzw[..., :3] / xyzw[..., 3:]


def separation(orders: core.Point) -> float:
    """The largest distance between where two refinements, on the first axis, put the same vertex."""
    xyz = coordinates(orders)
    return float(np.linalg.norm(xyz[0] - xyz[1], axis=-1).max())


def edges(faces: np.ndarray) -> np.ndarray:
    return np.unique(np.sort(np.stack([faces, np.roll(faces, -1, axis=1)], axis=-1), axis=-1).reshape(-1, 2), axis=0)


def draw_surface(ax: plt.Axes, xyz: np.ndarray, faces: np.ndarray, colours: np.ndarray) -> None:
    """A surface in the given face colours, shaded by how its faces turn to the light."""
    triangles = xyz[faces]                                                     # [faces, 3, 3]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lit = np.clip(normals @ (LIGHT / np.linalg.norm(LIGHT)) / np.linalg.norm(normals, axis=-1), 0, 1)
    shaded = np.clip(colours * (0.45 + 0.75 * lit[:, None]), 0, 1)
    ax.add_collection3d(Poly3DCollection(triangles, facecolors=shaded, edgecolors="none"))


def draw_mesh(ax: plt.Axes, xyz: np.ndarray, faces: np.ndarray, colour: str, style: str, width: float) -> None:
    ax.add_collection3d(Line3DCollection(xyz[edges(faces)], colors=colour, linestyles=style, linewidths=width))


def frame_axes(ax: plt.Axes, xyz: np.ndarray) -> None:
    centre, half = (xyz.max(axis=0) + xyz.min(axis=0)) / 2, np.ptp(xyz, axis=0).max() / 2
    ax.set(xlim=(centre[0] - half, centre[0] + half), ylim=(centre[1] - half, centre[1] + half), zlim=(centre[2] - half, centre[2] + half))
    ax.set_box_aspect((1, 1, 1), zoom=1.2)
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)
    ax.set_axis_off()


def draw_weights(controls: core.Point, control_faces: np.ndarray, surfaces: core.Point, faces: np.ndarray) -> plt.Figure:
    """Each case's control mesh, dotted, and the surface it refines to."""
    figure = plt.figure(figsize=(10.5, 3.8), layout="constrained")
    for index, (control, surface) in enumerate(zip(coordinates(controls), coordinates(surfaces))):
        ax = figure.add_subplot(1, controls.shape[0], index + 1, projection="3d")
        draw_surface(ax, surface, faces, np.broadcast_to(SURFACE_COLOUR, (len(faces), 3)))
        draw_mesh(ax, control, control_faces, CONTROL_COLOUR, ":", 0.8)
        frame_axes(ax, control)
    return figure


def draw_orders(control: core.Point, control_faces: np.ndarray, weighted_orders: core.Point, positional_orders: core.Point,
                faces: np.ndarray) -> plt.Figure:
    """For points with their weights and for positions: the surface refined after the perspective, each
    face coloured by how far the surface refined before it lies, on one scale for both, from blue where
    they agree to orange at the largest gap."""
    figure = plt.figure(figsize=(8.0, 4.0), layout="constrained")
    xyz = coordinates(control)
    gaps = [np.linalg.norm(np.subtract(*coordinates(orders)), axis=-1)[faces].mean(axis=-1)
            for orders in (weighted_orders, positional_orders)]                # [faces] each
    largest = max(gap.max() for gap in gaps)
    for index, (orders, gap) in enumerate(zip((weighted_orders, positional_orders), gaps)):
        ax = figure.add_subplot(1, 2, index + 1, projection="3d")
        share = (gap / largest)[:, None]
        draw_surface(ax, coordinates(orders)[0], faces, SURFACE_COLOUR * (1 - share) + GAP_COLOUR * share)
        draw_mesh(ax, xyz, control_faces, CONTROL_COLOUR, ":", 0.8)
        frame_axes(ax, xyz)
    return figure


def animate_weights(controls: core.Point, surfaces: core.Point, control_faces: np.ndarray, faces: np.ndarray) -> list[np.ndarray]:
    """One frame per weighting: the control mesh, dotted, and the surface it refines to, at one framing."""
    xyz = coordinates(controls)                                                # [frames, vertices, 3]
    shown = coordinates(surfaces)                                              # [frames, refined, 3]

    def frame(index: int) -> np.ndarray:
        figure = plt.figure(figsize=(4.0, 4.0), layout="constrained")
        ax = figure.add_subplot(projection="3d")
        draw_surface(ax, shown[index], faces, np.broadcast_to(SURFACE_COLOUR, (len(faces), 3)))
        draw_mesh(ax, xyz[index], control_faces, CONTROL_COLOUR, ":", 0.8)
        frame_axes(ax, xyz[0])
        pixels = capture(figure)
        plt.close(figure)
        return pixels
    return [frame(index) for index in range(controls.shape[0])]
