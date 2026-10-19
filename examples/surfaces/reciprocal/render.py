"""The lifted terrain with its creases, and the planar truss with its reciprocal diagram."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np

from examples.animation import capture
from examples.surfaces.reciprocal import core

TERRAIN_COLOUR = "#9fb7c4"
RECIPROCAL_COLOUR = "0.2"
# Tension blue, compression red, white at zero, on one scale for every frame: compression is far weaker than
# tension, so each side saturates at its own end. The widths of the bars.
STRESS_MAP = "coolwarm_r"
COMPRESSION, TENSION = -0.3, 1.5
BAR_WIDTH = 2.2
ELEVATION, AZIMUTH = 28.0, -60.0
REACH = 1.25


# --- plumbing -------------------------------------------------------------------------
def scalars(values: core.Scalar) -> np.ndarray:
    """The values of a batch of scalars, for colouring and checks."""
    return values.cast(core.ga.subspace.scalar()).kernel[..., 0]


def coordinates(points: core.Vector) -> np.ndarray:
    """x, y and z of each point, on the last axis."""
    return points.cast(core.ga.subspace("x y z")).kernel


def unbalanced(mesh: core.Mesh, stresses: core.Scalar) -> float:
    """The largest net force on an interior vertex from the bars meeting there, each pulling along itself
    by its stress times its length."""
    inner, _, _ = core.beside(mesh)
    pull = coordinates(mesh.edge_vectors.batch()[inner])[..., :2] * scalars(stresses)[..., None]   # [..., inner, 2]
    ends = np.zeros((len(mesh.vertices.batch()), len(inner)))                 # [vertices, inner]
    ends[mesh.edges[inner, 0], np.arange(len(inner))] = 1
    ends[mesh.edges[inner, 1], np.arange(len(inner))] = -1
    net = np.einsum("vi,...ic->...vc", ends, pull)                             # [..., vertices, 2]
    interior = np.setdiff1d(np.arange(len(ends)), mesh.edges[mesh.boundary_edges])
    return float(np.abs(net[..., interior, :]).max())


def colours(stresses: np.ndarray) -> np.ndarray:
    return plt.get_cmap(STRESS_MAP)(TwoSlopeNorm(0.0, COMPRESSION, TENSION)(np.clip(stresses, COMPRESSION, TENSION)))


def draw(mesh: core.Mesh, heights: np.ndarray, stresses: core.Scalar, duals: core.Vector) -> plt.Figure:
    """Left, the terrain the heights lift the mesh to, its creases coloured by stress; right, the planar
    truss, its bars coloured by stress, and the reciprocal diagram over it, an edge across every bar."""
    inner, left, right = core.beside(mesh)
    xy = coordinates(mesh.vertices.batch())[:, :2]                             # [vertices, 2]
    lifted = np.concatenate([xy, heights[:, None]], axis=-1)                   # [vertices, 3]
    shades = colours(scalars(stresses))                                        # [inner, 4]
    figure = plt.figure(figsize=(10.0, 5.0))
    terrain = figure.add_axes((0.0, 0.0, 0.5, 1.0), projection="3d", computed_zorder=False)
    terrain.add_collection3d(Poly3DCollection(lifted[mesh.faces], facecolor=TERRAIN_COLOUR, alpha=0.35, edgecolor="none", zorder=1))
    terrain.add_collection3d(Line3DCollection(lifted[mesh.edges[inner]], colors=shades, linewidths=BAR_WIDTH, zorder=2))
    terrain.add_collection3d(Line3DCollection(lifted[mesh.edges[mesh.boundary_edges]], colors="0.5", linewidths=1.0, zorder=2))
    terrain.set(xlim=(-REACH, REACH), ylim=(-REACH, REACH), zlim=(0, REACH))
    terrain.set_box_aspect((1, 1, 0.5), zoom=1.3)
    terrain.view_init(elev=ELEVATION, azim=AZIMUTH)
    terrain.set_axis_off()
    plane = figure.add_axes((0.5, 0.05, 0.5, 0.9))
    plane.add_collection(LineCollection(xy[mesh.edges[inner]], colors=shades, linewidths=BAR_WIDTH))
    plane.add_collection(LineCollection(xy[mesh.edges[mesh.boundary_edges]], colors="0.5", linewidths=1.0))
    centres = coordinates(duals)[:, :2]                                        # [faces, 2]
    plane.add_collection(LineCollection(centres[np.stack([left, right], axis=-1)], colors=RECIPROCAL_COLOUR, linewidths=1.0))
    plane.scatter(*centres.T, color=RECIPROCAL_COLOUR, s=8, zorder=3)
    plane.set(xlim=(-REACH, REACH), ylim=(-REACH, REACH), aspect="equal")
    plane.set_axis_off()
    return figure


def animate(mesh: core.Mesh, frames: Iterable[tuple[np.ndarray, core.Scalar, core.Vector]]) -> list[np.ndarray]:
    """The figure at every frame: its heights `[vertices]`, stresses `Scalar[inner]` and reciprocal
    vertices `[faces] Vector`."""
    images = []
    for heights, stresses, duals in frames:
        figure = draw(mesh, heights, stresses, duals)
        images.append(capture(figure))
        plt.close(figure)
    return images
