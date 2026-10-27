"""The lifted terrain with its creases, and the planar truss with its reciprocal diagram."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm, to_rgb
import numpy as np

from examples.util.animation import capture
from examples.surfaces.reciprocal import core
from examples.util.wgsl import fitted, meshes, show, solids

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
# share of its larger side, and the pixels across its picture, until the view is fitted to its axes; how much each
# surface of the terrain hides, so the creases behind it show through, and the widths of the creases and of the
# boundary as rods, in pixels.
ELEVATION, AZIMUTH = np.radians([28.0, 30.0])
MARGIN = 0.08
N_PIXELS = 1000
TERRAIN_OPACITY = 0.16
CREASE_WIDTH, BOUNDARY_WIDTH = 3.5, 1.75
# The room an animation's view leaves above and below the terrain of its first frame, for the heights to move in.
SWING = 0.02
BACKGROUND = np.ones(3)


# --- plumbing -------------------------------------------------------------------------
def scalars(values: core.Scalar) -> np.ndarray:
    """The values of a batch of scalars, for colouring."""
    return values.cast(core.ga.subspace.scalar()).kernel[..., 0]


def coordinates(vectors: core.Plane) -> np.ndarray:
    """x, y and z of each vector, on the last axis."""
    return vectors.cast(core.ga.subspace("x y z")).kernel


def positions(points: core.Point) -> np.ndarray:
    """x, y and z of each point, an antivector, on the last axis."""
    return points.cast(core.ga.subspace("yz zx xy")).kernel


def colours(stresses: np.ndarray) -> np.ndarray:
    return plt.get_cmap(STRESS_MAP)(TwoSlopeNorm(0.0, COMPRESSION, TENSION)(np.clip(stresses, COMPRESSION, TENSION)))


def view(points: core.Point, heights: np.ndarray) -> solids.View:
    """A view framing the terrain over the points at all the heights `[..., vertices]`."""
    xy = positions(points.batch())[:, :2]                               # [vertices, 2]
    lifted = np.concatenate([np.broadcast_to(xy, heights.shape + (2,)), heights[..., None]], axis=-1)   # [..., vertices, 3]
    return solids.view([solids.mv.point(lifted.reshape(-1, 3))], 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)


def terrain(mesh: core.Mesh, points: core.Point, heights: np.ndarray, shades: np.ndarray,
            framing: solids.View) -> np.ndarray:
    """The terrain the heights lift the mesh at its points to, translucent and flat over its faces, its creases
    rods coloured by their shades `[inner, 3]`, and its boundary grey."""
    inner = mesh.interior_edges
    xy = positions(points.batch())[:, :2]                               # [vertices, 2]
    lifted = np.concatenate([xy, heights[:, None]], axis=-1)                   # [vertices, 3]
    places = solids.mv.point(lifted)
    pixel = framing.pixel
    # The faces as a mesh, which the tracer composites sample by sample, where thin plates would each be searched
    # through every translucent layer; a plate is crossed twice, a face once, so a face hides what both would.
    surface = meshes.samples(places, mesh.faces, TERRAIN_COLOUR, meshes.flat(places, mesh.faces), framing,
                             face_values=np.zeros(4), colour=meshes.plain, gloss=solids.gloss)
    creases = solids.rods(places[mesh.edges[inner, 0]], places[mesh.edges[inner, 1]], CREASE_WIDTH / 2 * pixel)
    boundary = solids.rods(places[mesh.edges[mesh.boundary_edges, 0]], places[mesh.edges[mesh.boundary_edges, 1]],
                           BOUNDARY_WIDTH / 2 * pixel)
    return solids.draw([(creases, shades), (boundary, BOUNDARY_COLOUR)], framing, BACKGROUND,
                       inks=[], translucent=[], meshes=[(surface, 1 - (1 - TERRAIN_OPACITY) ** 2)])


def panels(mesh: core.Mesh, points: core.Point, framing: solids.View
           ) -> tuple[plt.Figure, Callable[[np.ndarray, core.Scalar, core.Plane], None]]:
    """The figure, laid out once, and what draws heights `[vertices]`, stresses `Scalar[inner]` and reciprocal
    vertices `[faces] Plane+` on it: left, the terrain the heights lift the mesh to, its creases coloured by stress,
    seen by the framing camera, fitted to its axes; right, the planar truss, its bars coloured by stress, and the reciprocal diagram over
    it, an edge across every bar."""
    inner, (left, right) = mesh.interior_edges, mesh.edge_faces.T
    xy = positions(points.batch())[:, :2]                               # [vertices, 2]
    figure = plt.figure(figsize=(10.0, 5.0))
    terrain_axes = figure.add_axes((0.0, 0.0, 0.5, 1.0))
    framing = fitted(framing, terrain_axes)
    picture = show(terrain_axes, np.zeros(framing.shape + (3,), dtype=np.uint8), framing)
    plane = figure.add_axes((0.5, 0.05, 0.5, 0.9))
    bars = plane.add_collection(LineCollection(xy[mesh.edges[inner]], linewidths=BAR_WIDTH))
    plane.add_collection(LineCollection(xy[mesh.edges[mesh.boundary_edges]], colors="0.5", linewidths=1.0))
    diagram = plane.add_collection(LineCollection([], colors=RECIPROCAL_COLOUR, linewidths=1.0))
    vertices = plane.scatter(np.zeros(0), np.zeros(0), color=RECIPROCAL_COLOUR, s=8, zorder=3)
    plane.set(xlim=(-REACH, REACH), ylim=(-REACH, REACH), aspect="equal")
    plane.set_axis_off()

    def drawn(heights: np.ndarray, stresses: core.Scalar, duals: core.Plane) -> None:
        shades = colours(scalars(stresses))                                    # [inner, 4]
        picture.set_data(terrain(mesh, points, heights, shades[:, :3], framing))
        bars.set_color(shades)
        centres = coordinates(duals)[:, :2]                                    # [faces, 2]
        diagram.set_segments(centres[np.stack([left, right], axis=-1)])
        vertices.set_offsets(centres)

    return figure, drawn


def draw(mesh: core.Mesh, points: core.Point, heights: np.ndarray, stresses: core.Scalar,
         duals: core.Plane) -> plt.Figure:
    """The terrain beside the planar truss and its reciprocal diagram, the terrain framed on itself."""
    figure, drawn = panels(mesh, points, view(points, heights))
    drawn(heights, stresses, duals)
    return figure


def animate(mesh: core.Mesh, points: core.Point, frames: Iterable[tuple[np.ndarray, core.Scalar, core.Plane]]) -> Iterator[np.ndarray]:
    """The figure at every frame: its heights `[vertices]`, stresses `Scalar[inner]` and reciprocal
    vertices `[faces] Plane+`, the terrain seen by one camera framing the first frame's with room for the heights to
    move."""
    frames = iter(frames)
    first = next(frames)
    figure, drawn = panels(mesh, points, view(points, first[0] + np.array([-SWING, SWING])[:, None]))
    for heights, stresses, duals in chain([first], frames):
        drawn(heights, stresses, duals)
        yield capture(figure)
    plt.close(figure)
