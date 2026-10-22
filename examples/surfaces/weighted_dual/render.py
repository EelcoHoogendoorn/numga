"""The hull and its polar dual in space, and both projected onto the sphere, drawn on the GPU: the sphere traced,
the hull a translucent mesh, edges rods, the arcs on the sphere tubes lying on its surface."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from examples.surfaces.weighted_dual import core
from examples.wgsl import behind, compose, image_figure, meshes, solids

sm = solids.mv
PRIMAL_COLOUR = np.array(to_rgb("#287fa0"))
DUAL_COLOUR = np.array(to_rgb("#d06c36"))
HULL_COLOUR = np.array(to_rgb("#b4d4e0"))
RAY_COLOUR = np.full(3, 0.55)
SPHERE_COLOUR = np.full(3, 0.97)
background = np.ones(3)
# How much of what lies behind them the sphere and the hull hide: in space the sphere is a faint shell and the hull,
# pale, all but hides the far side; on the sphere the sphere hides its far side.
SPACE_SPHERE_OPACITY, HULL_OPACITY, IMAGE_SPHERE_OPACITY = 0.15, 0.8, 1.0
ARC_SAMPLES = 16
# The colours of a vertex's distance from the sphere, and the distance at either end of them.
DISTANCE_MAP = "coolwarm"
DISTANCE_RANGE = 0.1
# How far each ray is drawn either side of the sphere.
RAY_SPAN = 0.15
# The view's elevation and azimuth in radians, azimuth zero looking along +y; half the width of the square view
# about the centre, one framing for both panels so the sphere is the same size in each, and the pixels across each.
ELEVATION, AZIMUTH = np.radians(20.0), np.radians(30.0)
HALF, N_PIXELS = 1.3, 500
# The points around a tube; the widths of the edges, the rays and the arcs, and the vertices' radius, in pixels.
N_AROUND = 8
EDGE_WIDTH, RAY_WIDTH, ARC_WIDTH, VERTEX_RADIUS = 2.2, 1.6, 2.4, 5.0


# --- plumbing -------------------------------------------------------------------------
def scalars(values: core.Scalar) -> np.ndarray:
    return values.cast(values.algebra.subspace.scalar()).kernel[..., 0]


def coordinates(points: core.Point) -> np.ndarray:
    """x, y and z of each point, on the last axis."""
    return points.cast(core.ga.subspace("yz zx xy")).kernel


def arcs(ends: np.ndarray) -> np.ndarray:
    """Great arcs on the unit sphere between the directions of pairs of points, [pairs, 2, 3]."""
    share = np.linspace(0, 1, ARC_SAMPLES)[:, None]
    chords = ends[:, None, 0] * (1 - share) + ends[:, None, 1] * share         # [pairs, samples, 3]
    return chords / np.linalg.norm(chords, axis=-1, keepdims=True)


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
    vertices, centres = sm.point(xyz), sm.point(poles)                         # [points] Point, [faces] Point
    sphere = solids.layer([(solids.ball(solids.origin, 1.0)[None, None], SPHERE_COLOUR)], view)
    hull = meshes.layer(vertices, faces, HULL_COLOUR, meshes.flat(vertices, faces), view)
    # Each vertex coloured by its distance from the sphere, blue inside, grey on it, red outside, on a stretch of
    # the ray from the centre it moves along.
    distance = np.linalg.norm(xyz, axis=-1)
    rays = xyz / distance[:, None]
    colours = plt.get_cmap(DISTANCE_MAP)((distance - 1 + DISTANCE_RANGE) / (2 * DISTANCE_RANGE))[:, :3]
    marks = solids.layer([(solids.rods(vertices[edges[:, 0]], vertices[edges[:, 1]], EDGE_WIDTH / 2 * pixel), PRIMAL_COLOUR),
                          (solids.rods(centres[beside[:, 0]], centres[beside[:, 1]], EDGE_WIDTH / 2 * pixel), DUAL_COLOUR),
                          (solids.rods(sm.point(rays * (1 - RAY_SPAN)), sm.point(rays * (1 + RAY_SPAN)),
                                       RAY_WIDTH / 2 * pixel), RAY_COLOUR),
                          (solids.ball(vertices, VERTEX_RADIUS * pixel)[:, None], colours)], view)
    # The hull's edges lie on its faces: the faces pushed back by an edge's width, so the edges stay in front.
    space = compose([(*sphere, SPACE_SPHERE_OPACITY), (*behind(hull, EDGE_WIDTH * pixel), HULL_OPACITY), (*marks, 1.0)],
                    background)
    # The arcs drawn a tube's radius out, so that the tubes lie on the sphere rather than in it, each coloured by its
    # kind.
    lifted = 1 + ARC_WIDTH / 2 * pixel
    on_sphere, duals_on_sphere = rays, poles / np.linalg.norm(poles, axis=-1, keepdims=True)
    paths = lifted * np.concatenate([arcs(on_sphere[edges]), arcs(duals_on_sphere[beside])])   # [arcs, samples, 3]
    kinds = np.repeat(np.stack([PRIMAL_COLOUR, DUAL_COLOUR]), len(edges), axis=0)   # [arcs, 3]
    tubes = meshes.tubes(list(zip(paths, kinds)), ARC_WIDTH, view, N_AROUND)
    # The arcs lie on the sphere: the sphere pushed back by an arc's width, so they stay in front where they touch it.
    image = compose([(*behind(sphere, ARC_WIDTH * pixel), IMAGE_SPHERE_OPACITY), (*tubes, 1.0)], background)
    return np.concatenate([space, image], axis=1)


def animate(frames: list[tuple[core.Point, np.ndarray, core.Point]]) -> list[np.ndarray]:
    """The complex at every frame, from one view."""
    return [picture(*geometry) for geometry in frames]
