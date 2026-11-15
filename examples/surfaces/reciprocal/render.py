"""Rendering for the lifted polyhedral terrain, its stressed creases, and the planar truss with its reciprocal diagram."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, to_rgb
import numpy as np

from numga import stack
from numga.render import drawing, meshes, simplices, solids
from numga.topology import Topology
from examples.surfaces.reciprocal import core
from examples.util.animation import capture
from examples.util.panels import fitted, offset_panel, overlay, show

terrain_colour = np.array(to_rgb("#9fb7c4"))
boundary_colour = np.full(3, 0.65)
reciprocal_colour, rim_colour = np.full(3, 0.2), np.full(3, 0.5)
# Tension renders blue, compression red, and zero stress white on one shared scale across frames.
stress_map = "coolwarm_r"
compression_limit, tension_limit = -0.3, 1.5
bar_width, line_width, corner_radius = 2.2, 1.4, 1.5
reach = 1.25
elevation, azimuth = np.radians([28.0, 30.0])
margin = 0.08
n_pixels = 1000
terrain_opacity = 0.16
crease_width, boundary_width = 2.2, 1.5
# Extra vertical headroom around the initial terrain so breathing heights stay inside the camera frustum.
height_swing = 0.02
background = np.ones(3)


# --- plumbing -------------------------------------------------------------------------
def stress_colours(stresses: np.ndarray) -> np.ndarray:
    """Maps clipped axial stresses to RGBA shades (`[..., 4]`) on a diverging two-slope scale centred at zero."""
    scale = TwoSlopeNorm(0.0, compression_limit, tension_limit)
    return plt.get_cmap(stress_map)(scale(np.clip(stresses, compression_limit, tension_limit)))


def terrain_view(points: core.Point, heights: core.Scalar) -> solids.View:
    """Frames a 3D view enclosing `points` (`Point-[V]`) lifted across `heights` (`[...] Scalar+[V]`)."""
    lifted = solids.origin + solids.context.include_from_algebra((points + core.up * heights).batch())  # [..., V] Point-
    return solids.view([lifted], 0.0, elevation, azimuth, margin, n_pixels)


def terrain_image(
    topology: Topology,
    points: core.Point,
    heights: core.Scalar,
    shades: np.ndarray,
    framing: solids.View,
) -> np.ndarray:
    """Renders the translucent polyhedral terrain over `points` (`Point-[V]`) with interior crease rods shaded by `shades` (`[E_interior, 3]`) and grey boundary rods."""
    lifted = points + core.up * heights                                         # Point-[V]
    places = solids.origin + solids.context.include_from_algebra(lifted.batch())  # [V] Point-
    pixel = framing.pixel
    # Ray-trace triangles as single-crossing translucent faces so occluded creases remain visible behind them.
    triangles = topology.elements[2]                                            # [F, 3]
    surface = meshes.samples(
        places, triangles, terrain_colour, meshes.flat(places, triangles), framing,
        face_values=np.zeros(4), colour=meshes.plain, gloss=solids.gloss,
    )
    ends = topology.incidence[1, 0] * lifted                                    # [2] Point-[E]
    crease_tails, crease_heads = topology.reading(topology.interior[1], 1) * ends  # Point-[E_interior], Point-[E_interior]
    rim_tails, rim_heads = topology.reading(topology.exterior[1], 1) * ends     # Point-[E_exterior], Point-[E_exterior]
    creases = solids.rods(
        solids.origin + solids.context.include_from_algebra(crease_tails.batch()),
        solids.origin + solids.context.include_from_algebra(crease_heads.batch()),
        crease_width / 2 * pixel,
    )
    boundary = solids.rods(
        solids.origin + solids.context.include_from_algebra(rim_tails.batch()),
        solids.origin + solids.context.include_from_algebra(rim_heads.batch()),
        boundary_width / 2 * pixel,
    )
    return solids.draw(
        [(creases, shades), (boundary, boundary_colour)], framing, background,
        inks=[], translucent=[], meshes=[(surface, 1 - (1 - terrain_opacity) ** 2)],
    )


def dual_panels(
    topology: Topology,
    points: core.Point,
    framing: solids.View,
) -> tuple[plt.Figure, Callable[[core.Scalar, core.Scalar, core.Plane], None]]:
    """Builds the side-by-side terrain and planar diagram figure alongside an updater for `(heights, stresses, dual_vertices)`."""
    places = points.batch()                                                     # [V] Point-
    bars = topology.subset(topology.interior[1], 1)
    rim = topology.border
    n_interior = len(topology.interior[1])
    # Each interior bar's dual edge connects the reciprocal vertices of the two triangles sharing that bar.
    dual_ends = topology.dual.reading(topology.interior[1], 1) * topology.dual.incidence[1, 0]  # [2] Scalar+[E_interior <- F]
    joins = Topology.chains([2] * n_interior)
    figure = plt.figure(figsize=(6.0, 3.0))
    terrain_axes = figure.add_axes((0.0, 0.0, 0.5, 1.0))
    framing = fitted(framing, terrain_axes)
    picture = show(terrain_axes, np.zeros(framing.shape + (3,), dtype=np.uint8), framing)
    plane_axes = figure.add_axes((0.5, 0.05, 0.5, 0.9))
    plane_axes.set(xlim=(-reach, reach), ylim=(-reach, reach), aspect="equal")
    plane_axes.set_axis_off()
    camera = offset_panel(plane_axes, core.mv.x, core.mv.y)
    # Pairing reciprocal plane vectors directly with the x and y directions reads out each face's horizontal slope components.
    slope_camera = offset_panel(plane_axes, core.mv.x.dual(), core.mv.y.dual())
    boundary = simplices.strokes(places, rim, rim_colour, line_width, camera)
    truss = overlay(plane_axes, [boundary])

    def update(heights: core.Scalar, stresses: core.Scalar, dual_vertices: core.Plane) -> None:
        shades = stress_colours(stresses.batch().to_array())                    # [E_interior, 4]
        picture.set_data(terrain_image(topology, points, heights, shades[:, :3], framing))
        vertices = dual_vertices.batch()                                        # [F] Plane+
        tails, heads = (dual_ends * dual_vertices).batch()                      # [E_interior] Plane+, [E_interior] Plane+
        truss.set_data(drawing.composited([
            simplices.strokes(places, bars, shades[:, :3], bar_width, camera),
            boundary,
            simplices.strokes(stack((tails, heads), axis=-1).reshape(-1), joins, reciprocal_colour, line_width, slope_camera),
            simplices.markers(vertices, reciprocal_colour, corner_radius, slope_camera),
        ]))

    return figure, update


def reciprocal_figure(
    topology: Topology,
    points: core.Point,
    heights: core.Scalar,
    stresses: core.Scalar,
    dual_vertices: core.Plane,
) -> plt.Figure:
    """Draws the lifted polyhedral terrain beside the stressed planar truss and its reciprocal diagram."""
    figure, update = dual_panels(topology, points, terrain_view(points, heights))
    update(heights, stresses, dual_vertices)
    return figure


def breath_animation(
    topology: Topology,
    points: core.Point,
    states: Iterable[tuple[core.Scalar, core.Scalar, core.Plane]],
) -> Iterator[np.ndarray]:
    """Yields RGB frames (`[n_rows, n_columns, 3]`) across `states` (`(Scalar+[V], Scalar+[E_interior], Plane+[F])`) from a fixed camera with vertical headroom."""
    iterator = iter(states)
    first = next(iterator)
    figure, update = dual_panels(topology, points, terrain_view(points, stack([first[0] - height_swing, first[0] + height_swing])))
    for heights, stresses, dual_vertices in chain([first], iterator):
        update(heights, stresses, dual_vertices)
        yield capture(figure)
    plt.close(figure)
