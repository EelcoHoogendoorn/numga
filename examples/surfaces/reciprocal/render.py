"""The lifted terrain with its creases, and the planar truss with its reciprocal diagram."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm, to_rgb
import numpy as np

from examples.util.animation import capture
from examples.surfaces.reciprocal import core
from examples.util.wgsl import meshes, show, solids

TERRAIN_COLOUR = np.array(to_rgb("#9fb7c4"))
BOUNDARY_COLOUR = np.full(3, 0.65)
RECIPROCAL_COLOUR = "0.2"
# Tension blue, compression red, white at zero, on one scale for every frame: compression is far weaker than
# tension, so each side saturates at its own end. The widths of the bars.
STRESS_MAP = "coolwarm_r"
COMPRESSION, TENSION = -0.3, 1.5
BAR_WIDTH = 2.2
REACH = 1.25
# The terrain's view, elevation and azimuth in radians, azimuth zero looking along +y, the margin around it as a
# share of its larger side, and the pixels across its picture; how much each surface of the terrain hides, so the
# creases behind it show through, and the widths of the creases and of the boundary as rods, in pixels.
ELEVATION, AZIMUTH = np.radians([28.0, 30.0])
MARGIN = 0.08
N_PIXELS = 1000
TERRAIN_OPACITY = 0.16
CREASE_WIDTH, BOUNDARY_WIDTH = 7.0, 3.5
BACKGROUND = np.ones(3)


# --- plumbing -------------------------------------------------------------------------
def scalars(values: core.Scalar) -> np.ndarray:
    """The values of a batch of scalars, for colouring and checks."""
    return values.cast(core.ga.subspace.scalar()).kernel[..., 0]


def coordinates(vectors: core.Vector) -> np.ndarray:
    """x, y and z of each vector, on the last axis."""
    return vectors.cast(core.ga.subspace("x y z")).kernel


def positions(points: core.Point) -> np.ndarray:
    """x, y and z of each point, an antivector, on the last axis."""
    return points.cast(core.ga.subspace("yz zx xy")).kernel


def unbalanced(mesh: core.Mesh, stresses: core.Scalar) -> float:
    """The largest net force on an interior vertex from the bars meeting there, each pulling along itself
    by its stress times its length."""
    inner, _, _ = core.beside(mesh)
    pull = positions(mesh.edge_displacements.batch()[inner])[..., :2] * scalars(stresses)[..., None]   # [..., inner, 2]
    ends = np.zeros((len(mesh.vertices.batch()), len(inner)))                 # [vertices, inner]
    ends[mesh.edges[inner, 0], np.arange(len(inner))] = 1
    ends[mesh.edges[inner, 1], np.arange(len(inner))] = -1
    net = np.einsum("vi,...ic->...vc", ends, pull)                             # [..., vertices, 2]
    interior = np.setdiff1d(np.arange(len(ends)), mesh.edges[mesh.boundary_edges])
    return float(np.abs(net[..., interior, :]).max())


def colours(stresses: np.ndarray) -> np.ndarray:
    return plt.get_cmap(STRESS_MAP)(TwoSlopeNorm(0.0, COMPRESSION, TENSION)(np.clip(stresses, COMPRESSION, TENSION)))


def view(mesh: core.Mesh, heights: np.ndarray) -> solids.View:
    """A view framing the terrain at all the heights `[..., vertices]`."""
    xy = positions(mesh.vertices.batch())[:, :2]                               # [vertices, 2]
    lifted = np.concatenate([np.broadcast_to(xy, heights.shape + (2,)), heights[..., None]], axis=-1)   # [..., vertices, 3]
    return solids.view([solids.mv.point(lifted.reshape(-1, 3))], 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)


def terrain(mesh: core.Mesh, heights: np.ndarray, shades: np.ndarray, framing: solids.View) -> np.ndarray:
    """The terrain the heights lift the mesh to, translucent and flat over its faces, its creases rods coloured
    by their shades `[inner, 3]`, and its boundary grey."""
    inner, _, _ = core.beside(mesh)
    xy = positions(mesh.vertices.batch())[:, :2]                               # [vertices, 2]
    lifted = np.concatenate([xy, heights[:, None]], axis=-1)                   # [vertices, 3]
    places = solids.mv.point(lifted)
    pixel = framing.pixel
    # The faces as a mesh, which the tracer composites sample by sample, where thin plates would each be searched
    # through every translucent layer; a plate is crossed twice, a face once, so a face hides what both would.
    surface = meshes.samples(places, mesh.faces, TERRAIN_COLOUR, meshes.flat(places, mesh.faces), framing)
    creases = solids.rods(places[mesh.edges[inner, 0]], places[mesh.edges[inner, 1]], CREASE_WIDTH / 2 * pixel)
    boundary = solids.rods(places[mesh.edges[mesh.boundary_edges, 0]], places[mesh.edges[mesh.boundary_edges, 1]],
                           BOUNDARY_WIDTH / 2 * pixel)
    return solids.draw([(creases, shades), (boundary, BOUNDARY_COLOUR)], framing, BACKGROUND,
                       meshes=[(surface, 1 - (1 - TERRAIN_OPACITY) ** 2)])


def panels(mesh: core.Mesh, heights: np.ndarray, stresses: core.Scalar, duals: core.Vector,
           framing: solids.View) -> plt.Figure:
    """Left, the terrain the heights lift the mesh to, its creases coloured by stress, seen by the framing camera;
    right, the planar truss, its bars coloured by stress, and the reciprocal diagram over it, an edge across every
    bar."""
    inner, left, right = core.beside(mesh)
    xy = positions(mesh.vertices.batch())[:, :2]                               # [vertices, 2]
    shades = colours(scalars(stresses))                                        # [inner, 4]
    figure = plt.figure(figsize=(10.0, 5.0))
    show(figure.add_axes((0.0, 0.0, 0.5, 1.0)), terrain(mesh, heights, shades[:, :3], framing), framing)
    plane = figure.add_axes((0.5, 0.05, 0.5, 0.9))
    plane.add_collection(LineCollection(xy[mesh.edges[inner]], colors=shades, linewidths=BAR_WIDTH))
    plane.add_collection(LineCollection(xy[mesh.edges[mesh.boundary_edges]], colors="0.5", linewidths=1.0))
    centres = coordinates(duals)[:, :2]                                        # [faces, 2]
    plane.add_collection(LineCollection(centres[np.stack([left, right], axis=-1)], colors=RECIPROCAL_COLOUR, linewidths=1.0))
    plane.scatter(*centres.T, color=RECIPROCAL_COLOUR, s=8, zorder=3)
    plane.set(xlim=(-REACH, REACH), ylim=(-REACH, REACH), aspect="equal")
    plane.set_axis_off()
    return figure


def draw(mesh: core.Mesh, heights: np.ndarray, stresses: core.Scalar, duals: core.Vector) -> plt.Figure:
    """The terrain beside the planar truss and its reciprocal diagram, the terrain framed on itself."""
    return panels(mesh, heights, stresses, duals, view(mesh, heights))


def animate(mesh: core.Mesh, frames: Iterable[tuple[np.ndarray, core.Scalar, core.Vector]]) -> list[np.ndarray]:
    """The figure at every frame: its heights `[vertices]`, stresses `Scalar[inner]` and reciprocal
    vertices `[faces] Vector`, the terrain seen by one camera framing every frame."""
    frames = list(frames)
    framing = view(mesh, np.stack([heights for heights, _, _ in frames]))
    images = []
    for heights, stresses, duals in frames:
        figure = panels(mesh, heights, stresses, duals, framing)
        images.append(capture(figure))
        plt.close(figure)
    return images
