"""The lifted terrain with its creases, and the planar truss with its reciprocal diagram."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, to_rgb
import numpy as np

from numga import stack
from examples.util.animation import capture
from examples.surfaces.reciprocal import core
from examples.util.wgsl import fitted, meshes, show, simplices, solids
from examples.util.wgsl.camera import flat_panel, panel

TERRAIN_COLOUR = np.array(to_rgb("#9fb7c4"))
BOUNDARY_COLOUR = np.full(3, 0.65)
RECIPROCAL_COLOUR, RIM_COLOUR = np.full(3, 0.2), np.full(3, 0.5)
# Tension blue, compression red, white at zero, on one scale for every frame: compression is far weaker than
# tension, so each side saturates at its own end. In pixels, the widths of the bars and of the other lines, and the
# radius of the reciprocal diagram's vertices.
STRESS_MAP = "coolwarm_r"
COMPRESSION, TENSION = -0.3, 1.5
BAR_WIDTH, LINE_WIDTH, CORNER_RADIUS = 2.2, 1.4, 1.5
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
def colours(stresses: np.ndarray) -> np.ndarray:
    return plt.get_cmap(STRESS_MAP)(TwoSlopeNorm(0.0, COMPRESSION, TENSION)(np.clip(stresses, COMPRESSION, TENSION)))


def view(points: core.Point, heights: core.Scalar) -> solids.View:
    """A view framing the terrain over the points at all the heights `[...] Scalar+[vertices]`."""
    lifted = solids.origin + solids.context.include_from_algebra((points + core.up * heights).batch())   # [...] Point-[vertices]
    return solids.view([lifted], 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)


def terrain(topology: core.Topology, points: core.Point, heights: core.Scalar, shades: np.ndarray,
            framing: solids.View) -> np.ndarray:
    """The terrain the heights lift the mesh at its points to, translucent and flat over its faces, its creases
    rods coloured by their shades `[inner, 3]`, and its boundary grey."""
    # Each point of the plane lifted by its height along the direction up.
    lifted = points + core.up * heights                                         # Point-[vertices]
    places = solids.origin + solids.context.include_from_algebra(lifted.batch())          # [vertices] Point-
    pixel = framing.pixel
    # The faces as a mesh, which the tracer composites sample by sample, where thin plates would each be searched
    # through every translucent layer; a plate is crossed twice, a face once, so a face hides what both would.
    triangles = topology.elements[2]
    surface = meshes.samples(places, triangles, TERRAIN_COLOUR, meshes.flat(places, triangles), framing,
                             face_values=np.zeros(4), colour=meshes.plain, gloss=solids.gloss)
    ends = topology.incidence[1, 0] * lifted                                    # [ends] Point-[edges]
    crease_tails, crease_heads = topology.reading(topology.interior[1], 1) * ends  # Point-[inner], Point-[inner]
    rim_tails, rim_heads = topology.reading(topology.exterior[1], 1) * ends     # Point-[rim], Point-[rim]
    creases = solids.rods(solids.origin + solids.context.include_from_algebra(crease_tails.batch()),
                          solids.origin + solids.context.include_from_algebra(crease_heads.batch()), CREASE_WIDTH / 2 * pixel)
    boundary = solids.rods(solids.origin + solids.context.include_from_algebra(rim_tails.batch()),
                           solids.origin + solids.context.include_from_algebra(rim_heads.batch()), BOUNDARY_WIDTH / 2 * pixel)
    return solids.draw([(creases, shades), (boundary, BOUNDARY_COLOUR)], framing, BACKGROUND,
                       inks=[], translucent=[], meshes=[(surface, 1 - (1 - TERRAIN_OPACITY) ** 2)])


def panels(topology: core.Topology, points: core.Point, framing: solids.View
           ) -> tuple[plt.Figure, Callable[[core.Scalar, core.Scalar, core.Plane], None]]:
    """The figure, laid out once, and what draws heights `Scalar+[vertices]`, stresses `Scalar[inner]` and
    reciprocal vertices `Plane+[faces]` on it: left, the terrain the heights lift the mesh to, its creases coloured by stress,
    seen by the framing camera, fitted to its axes; right, the planar truss, its bars coloured by stress, and the reciprocal diagram over
    it, an edge across every bar."""
    places = points.batch()                                                               # [vertices] Point-
    # The truss's bars and its boundary, the mesh's inner and boundary edges.
    bars = topology.subset(topology.interior[1], 1)
    rim = topology.border
    # The reciprocal diagram: a reciprocal vertex for each face, joined across every bar to the one on its other side
    # along the bar's dual edge, each join drawn as a chain of its two ends.
    across_bars = topology.dual.reading(topology.interior[1], 1) * topology.dual.incidence[1, 0]   # [ends] Scalar+[inner <- faces]
    joins = simplices.Topology.chains([2] * len(topology.interior[1]))
    figure = plt.figure(figsize=(6.0, 3.0))
    terrain_axes = figure.add_axes((0.0, 0.0, 0.5, 1.0))
    framing = fitted(framing, terrain_axes)
    picture = show(terrain_axes, np.zeros(framing.shape + (3,), dtype=np.uint8), framing)
    plane = figure.add_axes((0.5, 0.05, 0.5, 0.9))
    plane.set(xlim=(-REACH, REACH), ylim=(-REACH, REACH), aspect="equal")
    plane.set_axis_off()
    camera = flat_panel(plane, places.algebra)
    # The reciprocal vertices, the faces' slopes, read as they are by the directions along x and y: how fast each face
    # rises along each, its place in the diagram.
    slope_camera = panel(plane, core.mv.x.dual(), core.mv.y.dual())
    boundary = simplices.strokes(places, rim, RIM_COLOUR, LINE_WIDTH, camera)
    truss = simplices.overlay(plane, [boundary])

    def drawn(heights: core.Scalar, stresses: core.Scalar, duals: core.Plane) -> None:
        shades = colours(stresses.batch().to_array())                          # [inner, 4]
        picture.set_data(terrain(topology, points, heights, shades[:, :3], framing))
        corners = duals.batch()                                                # [faces] Plane+
        tails, heads = (across_bars * duals).batch()                           # [inner] Plane+, [inner] Plane+
        truss.set_data(simplices.composited([
            simplices.strokes(places, bars, shades[:, :3], BAR_WIDTH, camera), boundary,
            simplices.strokes(stack((tails, heads), axis=-1).reshape(-1), joins, RECIPROCAL_COLOUR, LINE_WIDTH, slope_camera),
            simplices.markers(corners, RECIPROCAL_COLOUR, CORNER_RADIUS, slope_camera)]))

    return figure, drawn


def draw(topology: core.Topology, points: core.Point, heights: core.Scalar, stresses: core.Scalar,
         duals: core.Plane) -> plt.Figure:
    """The terrain beside the planar truss and its reciprocal diagram, the terrain framed on itself."""
    figure, drawn = panels(topology, points, view(points, heights))
    drawn(heights, stresses, duals)
    return figure


def animate(topology: core.Topology, points: core.Point, frames: Iterable[tuple[core.Scalar, core.Scalar, core.Plane]]) -> Iterator[np.ndarray]:
    """The figure at every frame: its heights `Scalar+[vertices]`, stresses `Scalar[inner]` and reciprocal
    vertices `Plane+[faces]`, the terrain seen by one camera framing the first frame's with room for the heights to
    move."""
    frames = iter(frames)
    first = next(frames)
    figure, drawn = panels(topology, points, view(points, stack([first[0] - SWING, first[0] + SWING])))
    for heights, stresses, duals in chain([first], frames):
        drawn(heights, stresses, duals)
        yield capture(figure)
    plt.close(figure)
