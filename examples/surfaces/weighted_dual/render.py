"""The hull and its polar dual in space, and both projected onto the sphere, ray traced on the GPU: the hull an
opaque solid, a cone from the centre over each face, its edges and the dual's edges rods, and the arcs on the sphere
inked on its surface."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from examples.surfaces.weighted_dual import core
from examples.util.wgsl import image_figure, solids, strokes

sm = solids.mv
PRIMAL_COLOUR = np.array(to_rgb("#287fa0"))
DUAL_COLOUR = np.array(to_rgb("#d06c36"))
HULL_COLOUR = np.array(to_rgb("#b4d4e0"))
RAY_COLOUR = np.full(3, 0.55)
SPHERE_COLOUR = np.full(3, 0.97)
background = np.ones(3)
# Each cone from the centre over a face: the centre, then the face's three corners, and its four faces.
CONE_FACES = np.array([[1, 2, 3], [0, 1, 2], [0, 2, 3], [0, 3, 1]])
# The colours of a vertex's distance from the sphere, and the distance at either end of them.
DISTANCE_MAP = "coolwarm"
DISTANCE_RANGE = 0.1
# How far each ray is drawn either side of the sphere.
RAY_SPAN = 0.15
# The view's elevation and azimuth in radians, azimuth zero looking along +y; half the width of the square view
# about the centre, one framing for both panels so the sphere is the same size in each, and the pixels across each.
ELEVATION, AZIMUTH = np.radians(20.0), np.radians(30.0)
HALF, N_PIXELS = 1.3, 300
# The widths of the edges, the rays and the arcs, and the vertices' radius, in pixels.
EDGE_WIDTH, RAY_WIDTH, ARC_WIDTH, VERTEX_RADIUS = 1.3, 1.0, 1.4, 3.0


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """x, y and z of each point, on the last axis."""
    return points.batch().cast(core.ga.subspace("yz zx xy")).kernel


def arcs(ends: solids.Point) -> tuple[solids.Quadric, solids.Quadric]:
    """The great arcs on the unit sphere between pairs of points on it of unit weight `[pairs, 2]`, as lines to ink:
    the plane through the centre and both ends, and, to bound it, the side of the plane across the chord away from
    the centre, `[pairs]` each."""
    start, end = ends[:, 0], ends[:, 1]
    # The direction from the centre to the chord's middle; the plane square to it through the start, facing the centre.
    middle = (start - solids.origin) + (end - solids.origin)                  # [pairs] Point-
    across = middle.dual() - sm.w * (middle.dual() & start)                   # [pairs] Plane+
    return solids.half_space((solids.origin & start & end) * solids.flip), solids.half_space(across)


def draw_complex(mesh: core.Mesh, points: core.Point, duals: core.Point) -> plt.Figure:
    """In space, the polyhedron of the points and its polar dual; on the sphere, both projected from the
    centre."""
    return image_figure(picture(mesh, points, duals))


def picture(mesh: core.Mesh, points: core.Point, duals: core.Point) -> np.ndarray:
    """The two panels side by side: in space, the hull, opaque, with its edges, the dual's edges between the poles
    where a face of the dual bounding them faces the viewer, and each vertex on a stretch of its ray from the centre;
    on the sphere, the arcs of both."""
    view = solids.View(solids.aim(solids.origin, ELEVATION, AZIMUTH), np.full(2, HALF), N_PIXELS)
    pixel = view.pixel
    xyz, poles = coordinates(points), coordinates(duals)                       # [points, 3], [faces, 3]
    edges, beside = mesh.edges, mesh.edge_faces                                # [edges, 2], [edges, 2]
    vertices, dual_vertices = sm.point(xyz), sm.point(poles)                   # [points], [faces] Point
    sphere = solids.ball(solids.origin, 1.0)[None, None]                       # [1, 1] Plane+ <- Point-
    # The hull, folded or not, as one cone from the centre over each face, so that it is solid wherever its faces
    # are.
    corners = np.concatenate([np.zeros((len(mesh.faces), 1, 3)), xyz[mesh.faces]], axis=1)   # [faces, 4, 3]
    hull = solids.polyhedra(sm.point(corners), CONE_FACES)                     # [faces, 4] Plane+
    # Each vertex coloured by its distance from the sphere, blue inside, grey on it, red outside, on a stretch of
    # the ray from the centre it moves along.
    distance = np.linalg.norm(xyz, axis=-1)
    rays = xyz / distance[:, None]
    colours = plt.get_cmap(DISTANCE_MAP)((distance - 1 + DISTANCE_RANGE) / (2 * DISTANCE_RANGE))[:, :3]
    # The face of the dual polar to a vertex faces along the vertex's ray; a dual edge, across a primal edge, is
    # drawn where either of the two faces it bounds, those polar to the edge's ends, faces the viewer.
    toward = (solids.axes & (view.camera >> sm.z.dual())).to_array()           # [3]
    front = (rays[edges] @ toward > 0).any(axis=-1)                            # [edges]
    primal_rods, primal_joints = solids.edges(vertices[edges[:, 0]], vertices[edges[:, 1]], EDGE_WIDTH / 2 * pixel)
    dual_rods, dual_joints = solids.edges(dual_vertices[beside[front, 0]], dual_vertices[beside[front, 1]],
                                          EDGE_WIDTH / 2 * pixel)
    ray_strokes = strokes.segments(sm.point(rays * (1 - RAY_SPAN)), sm.point(rays * (1 + RAY_SPAN)), RAY_COLOUR,
                                   RAY_WIDTH, view)
    parts = [(hull, HULL_COLOUR), (primal_rods, PRIMAL_COLOUR), (primal_joints, PRIMAL_COLOUR),
             (dual_rods, DUAL_COLOUR), (dual_joints, DUAL_COLOUR),
             (solids.ball(vertices, VERTEX_RADIUS * pixel)[:, None], colours)]
    space = solids.draw(parts, view, background, inks=[], translucent=[], meshes=[(ray_strokes, 1.0)])
    # On the sphere, the arcs of both, each coloured by its kind.
    on_sphere, duals_on_sphere = rays, poles / np.linalg.norm(poles, axis=-1, keepdims=True)
    primal_arcs, dual_arcs = arcs(sm.point(on_sphere[edges])), arcs(sm.point(duals_on_sphere[beside]))
    inks = [solids.Ink(0, *primal_arcs, PRIMAL_COLOUR, ARC_WIDTH, 1.0),
            solids.Ink(0, *dual_arcs, DUAL_COLOUR, ARC_WIDTH, 1.0)]
    image = solids.draw([(sphere, SPHERE_COLOUR)], view, background, inks, translucent=[], meshes=[])
    return np.concatenate([space, image], axis=1)


def animate(frames: Iterable[tuple[core.Mesh, core.Point, core.Point]]) -> Iterator[np.ndarray]:
    """The complex at every frame, from one view."""
    for geometry in frames:
        yield picture(*geometry)
