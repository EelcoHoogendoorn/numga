"""The hull and its polar dual in space, and both projected onto the sphere, ray traced on the GPU: the hull the
translucent solid inside the planes of its faces, edges tubes, and the arcs on the sphere inked on its surface."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from examples.surfaces.weighted_dual import core
from examples.util.wgsl import image_figure, meshes, solids

sm = solids.mv
PRIMAL_COLOUR = np.array(to_rgb("#287fa0"))
DUAL_COLOUR = np.array(to_rgb("#d06c36"))
HULL_COLOUR = np.array(to_rgb("#b4d4e0"))
RAY_COLOUR = np.full(3, 0.55)
SPHERE_COLOUR = np.full(3, 0.97)
background = np.ones(3)
# How much of what lies behind them each surface of the sphere and of the hull hides in space: the sphere is a faint
# shell and the hull, pale, all but hides the far side.
SPHERE_OPACITY, HULL_OPACITY = 0.08, 0.8
# The colours of a vertex's distance from the sphere, and the distance at either end of them.
DISTANCE_MAP = "coolwarm"
DISTANCE_RANGE = 0.1
# How far each ray is drawn either side of the sphere.
RAY_SPAN = 0.15
# The view's elevation and azimuth in radians, azimuth zero looking along +y; half the width of the square view
# about the centre, one framing for both panels so the sphere is the same size in each, and the pixels across each.
ELEVATION, AZIMUTH = np.radians(20.0), np.radians(30.0)
HALF, N_PIXELS = 1.3, 500
# The widths of the edges, the rays and the arcs, and the vertices' radius, in pixels.
EDGE_WIDTH, RAY_WIDTH, ARC_WIDTH, VERTEX_RADIUS = 2.2, 1.6, 2.4, 5.0


# --- plumbing -------------------------------------------------------------------------
def scalars(values: core.Scalar) -> np.ndarray:
    return values.cast(values.algebra.subspace.scalar()).kernel[..., 0]


def coordinates(points: core.Point) -> np.ndarray:
    """x, y and z of each point, on the last axis."""
    return points.cast(core.ga.subspace("yz zx xy")).kernel


def arcs(ends: np.ndarray) -> tuple[solids.Quadric, solids.Quadric]:
    """The great arcs on the unit sphere between pairs of points on it `[pairs, 2, 3]`, as lines to ink: the plane
    through the centre and both ends, and, to bound it, the side of the plane across the chord away from the centre,
    `[pairs]` each."""
    start, end = ends[:, 0], ends[:, 1]
    middle = start + end
    across = solids.plane(-middle, -np.sum(middle * start, axis=-1))
    return solids.half_space(solids.plane(np.cross(start, end), np.zeros(len(ends)))), solids.half_space(across)


def draw_complex(points: core.Point, faces: np.ndarray, duals: core.Point) -> plt.Figure:
    """In space, the polyhedron of the points and its polar dual; on the sphere, both projected from the
    centre."""
    return image_figure(picture(points, faces, duals))


def picture(points: core.Point, faces: np.ndarray, duals: core.Point) -> np.ndarray:
    """The two panels side by side: in space, the hull, translucent, with its edges, the dual's edges between
    the poles, and each vertex on a stretch of its ray from the centre; on the sphere, the arcs of both."""
    view = solids.View(solids.aim(solids.origin, ELEVATION, AZIMUTH), np.full(2, HALF), N_PIXELS)
    pixel = view.pixel
    xyz, poles = coordinates(points), coordinates(duals)                       # [points, 3], [faces, 3]
    edges, beside = core.adjacency(faces)                                      # [edges, 2] each
    vertices = sm.point(xyz)                                                   # [points] Point
    sphere = solids.ball(solids.origin, 1.0)[None, None]                       # [1, 1] Quadric
    # The hull: inside the plane of every face, the plane that pairs to one with the face's points, whose dual is the
    # face's pole.
    hull = solids.half_space(solids.plane(poles, np.ones(len(poles))))[None]  # [1, faces] Quadric
    # Each vertex coloured by its distance from the sphere, blue inside, grey on it, red outside, on a stretch of
    # the ray from the centre it moves along.
    distance = np.linalg.norm(xyz, axis=-1)
    rays = xyz / distance[:, None]
    colours = plt.get_cmap(DISTANCE_MAP)((distance - 1 + DISTANCE_RANGE) / (2 * DISTANCE_RANGE))[:, :3]
    # The edges and the rays are strokes, rasterized as tubes; the shells and the vertices are traced.
    strokes = [meshes.segment_samples(xyz[edges[:, 0]], xyz[edges[:, 1]], PRIMAL_COLOUR, EDGE_WIDTH, view),
               meshes.segment_samples(poles[beside[:, 0]], poles[beside[:, 1]], DUAL_COLOUR, EDGE_WIDTH, view),
               meshes.segment_samples(rays * (1 - RAY_SPAN), rays * (1 + RAY_SPAN), RAY_COLOUR, RAY_WIDTH, view)]
    space = solids.draw([(solids.ball(vertices, VERTEX_RADIUS * pixel)[:, None], colours)], view, background,
                        translucent=[(sphere, SPHERE_COLOUR, SPHERE_OPACITY), (hull, HULL_COLOUR, HULL_OPACITY)],
                        meshes=[(stroke, 1.0) for stroke in strokes])
    # On the sphere, the arcs of both, each coloured by its kind.
    on_sphere, duals_on_sphere = rays, poles / np.linalg.norm(poles, axis=-1, keepdims=True)
    primal_arcs, dual_arcs = arcs(on_sphere[edges]), arcs(duals_on_sphere[beside])
    inks = [solids.Ink(0, *primal_arcs, PRIMAL_COLOUR, ARC_WIDTH, 1.0),
            solids.Ink(0, *dual_arcs, DUAL_COLOUR, ARC_WIDTH, 1.0)]
    image = solids.draw([(sphere, SPHERE_COLOUR)], view, background, inks)
    return np.concatenate([space, image], axis=1)


def animate(frames: list[tuple[core.Point, np.ndarray, core.Point]]) -> list[np.ndarray]:
    """The complex at every frame, from one view."""
    return [picture(*geometry) for geometry in frames]
