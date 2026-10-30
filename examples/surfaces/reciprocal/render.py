"""The lifted terrain with its creases, and the planar truss with its reciprocal diagram."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm, to_rgb
import numpy as np

from numga import stack
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
BAR_WIDTH = 1.6
REACH = 1.25
# The terrain's view, elevation and azimuth in radians, azimuth zero looking along +y, the margin around it as a
# share of its larger side, and the pixels across its picture, until the view is fitted to its axes; how much each
# surface of the terrain hides, so the creases behind it show through, and the widths of the creases and of the
# boundary as rods, in pixels.
ELEVATION, AZIMUTH = np.radians([28.0, 30.0])
MARGIN = 0.08
N_PIXELS = 1000
TERRAIN_OPACITY = 0.16
CREASE_WIDTH, BOUNDARY_WIDTH = 2.2, 1.5
# The room an animation's view leaves above and below the terrain of its first frame, for the heights to move in.
SWING = 0.02
BACKGROUND = np.ones(3)


# --- plumbing -------------------------------------------------------------------------
def plane_places(points: core.Point) -> np.ndarray:
    """x and y of points `[...]` on the last axis, for the planar truss's artists: their pairings with the planes x
    and y."""
    mv = core.mv
    return (stack((mv.x, mv.y)) & points.batch()[..., None]).to_array()


def colours(stresses: np.ndarray) -> np.ndarray:
    return plt.get_cmap(STRESS_MAP)(TwoSlopeNorm(0.0, COMPRESSION, TENSION)(np.clip(stresses, COMPRESSION, TENSION)))


def view(points: core.Point, heights: core.Scalar) -> solids.View:
    """A view framing the terrain over the points at all the heights `[...] Scalar+[vertices]`."""
    lifted = solids.origin + solids.context.include_from_algebra((points + core.up * heights).batch())   # [...] Point-[vertices]
    return solids.view([lifted], 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)


def terrain(mesh: core.Mesh, points: core.Point, heights: core.Scalar, shades: np.ndarray,
            framing: solids.View) -> np.ndarray:
    """The terrain the heights lift the mesh at its points to, translucent and flat over its faces, its creases
    rods coloured by their shades `[inner, 3]`, and its boundary grey."""
    # Each point of the plane lifted by its height along the direction up.
    lifted = points + core.up * heights                                         # Point-[vertices]
    places = solids.origin + solids.context.include_from_algebra(lifted.batch())          # [vertices] Point-
    pixel = framing.pixel
    # The faces as a mesh, which the tracer composites sample by sample, where thin plates would each be searched
    # through every translucent layer; a plate is crossed twice, a face once, so a face hides what both would.
    surface = meshes.samples(places, mesh.faces, TERRAIN_COLOUR, meshes.flat(places, mesh.faces), framing,
                             face_values=np.zeros(4), colour=meshes.plain, gloss=solids.gloss)
    ends = mesh.incidence[1, 0] * lifted                                        # [ends] Point-[edges]
    crease_tails, crease_heads = mesh.reading(mesh.interior_edges, 1) * ends    # Point-[inner], Point-[inner]
    rim_tails, rim_heads = mesh.reading(mesh.boundary_edges, 1) * ends          # Point-[rim], Point-[rim]
    creases = solids.rods(solids.origin + solids.context.include_from_algebra(crease_tails.batch()),
                          solids.origin + solids.context.include_from_algebra(crease_heads.batch()), CREASE_WIDTH / 2 * pixel)
    boundary = solids.rods(solids.origin + solids.context.include_from_algebra(rim_tails.batch()),
                           solids.origin + solids.context.include_from_algebra(rim_heads.batch()), BOUNDARY_WIDTH / 2 * pixel)
    return solids.draw([(creases, shades), (boundary, BOUNDARY_COLOUR)], framing, BACKGROUND,
                       inks=[], translucent=[], meshes=[(surface, 1 - (1 - TERRAIN_OPACITY) ** 2)])


def panels(mesh: core.Mesh, points: core.Point, framing: solids.View
           ) -> tuple[plt.Figure, Callable[[core.Scalar, core.Scalar, core.Plane], None]]:
    """The figure, laid out once, and what draws heights `Scalar+[vertices]`, stresses `Scalar[inner]` and
    reciprocal vertices `Plane+[faces]` on it: left, the terrain the heights lift the mesh to, its creases coloured by stress,
    seen by the framing camera, fitted to its axes; right, the planar truss, its bars coloured by stress, and the reciprocal diagram over
    it, an edge across every bar."""
    # The truss's bars and its boundary, each edge from its tail to its head, `[edges, ends, 2]` for the artists.
    ends = mesh.incidence[1, 0] * points                                        # [ends] Point-[edges]
    bar_ends = plane_places(mesh.reading(mesh.interior_edges, 1) * ends).transpose(1, 0, 2)   # [inner, ends, 2]
    rim_ends = plane_places(mesh.reading(mesh.boundary_edges, 1) * ends).transpose(1, 0, 2)   # [rim, ends, 2]
    # The faces on either side of each bar, whose reciprocal vertices its reciprocal edge joins.
    sides = mesh.reading(mesh.edge_faces.T, 2)                                  # [sides] Scalar+[inner <- faces]
    # The directions along x and y: a reciprocal vertex, a face's slope, pairs with them to how fast the face rises
    # along each, its place in the diagram.
    directions = stack((core.mv.x, core.mv.y)).dual()                           # [2] Point-
    figure = plt.figure(figsize=(6.0, 3.0))
    terrain_axes = figure.add_axes((0.0, 0.0, 0.5, 1.0))
    framing = fitted(framing, terrain_axes)
    picture = show(terrain_axes, np.zeros(framing.shape + (3,), dtype=np.uint8), framing)
    plane = figure.add_axes((0.5, 0.05, 0.5, 0.9))
    bars = plane.add_collection(LineCollection(bar_ends, linewidths=BAR_WIDTH))
    plane.add_collection(LineCollection(rim_ends, colors="0.5", linewidths=1.0))
    diagram = plane.add_collection(LineCollection([], colors=RECIPROCAL_COLOUR, linewidths=1.0))
    vertices = plane.scatter(np.zeros(0), np.zeros(0), color=RECIPROCAL_COLOUR, s=5, zorder=3)
    plane.set(xlim=(-REACH, REACH), ylim=(-REACH, REACH), aspect="equal")
    plane.set_axis_off()

    def drawn(heights: core.Scalar, stresses: core.Scalar, duals: core.Plane) -> None:
        shades = colours(stresses.batch().to_array())                          # [inner, 4]
        picture.set_data(terrain(mesh, points, heights, shades[:, :3], framing))
        bars.set_color(shades)
        reciprocal_edges = (sides * duals).batch()[..., None] & directions    # [sides, inner, 2] Scalar+
        diagram.set_segments(reciprocal_edges.to_array().transpose(1, 0, 2))
        vertices.set_offsets((duals.batch()[..., None] & directions).to_array())

    return figure, drawn


def draw(mesh: core.Mesh, points: core.Point, heights: core.Scalar, stresses: core.Scalar,
         duals: core.Plane) -> plt.Figure:
    """The terrain beside the planar truss and its reciprocal diagram, the terrain framed on itself."""
    figure, drawn = panels(mesh, points, view(points, heights))
    drawn(heights, stresses, duals)
    return figure


def animate(mesh: core.Mesh, points: core.Point, frames: Iterable[tuple[core.Scalar, core.Scalar, core.Plane]]) -> Iterator[np.ndarray]:
    """The figure at every frame: its heights `Scalar+[vertices]`, stresses `Scalar[inner]` and reciprocal
    vertices `Plane+[faces]`, the terrain seen by one camera framing the first frame's with room for the heights to
    move."""
    frames = iter(frames)
    first = next(frames)
    figure, drawn = panels(mesh, points, view(points, stack([first[0] - SWING, first[0] + SWING])))
    for heights, stresses, duals in chain([first], frames):
        drawn(heights, stresses, duals)
        yield capture(figure)
    plt.close(figure)
