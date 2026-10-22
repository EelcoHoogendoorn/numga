"""A Koebe polyhedron and its dual about the sphere they both touch, and their circles on the sphere, drawn on the
GPU: the sphere traced, the edges rods, the circles tubes lying on its surface. The sphere all but hides what lies
behind it."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.surfaces.koebe import core
from examples.wgsl import compose, meshes, show, solids

sm = solids.mv
PRIMAL_COLOUR = np.array(to_rgb("#287fa0"))
DUAL_COLOUR = np.array(to_rgb("#d06c36"))
SPHERE_COLOUR, SPHERE_OPACITY, TOUCH_COLOUR = np.full(3, 0.97), 0.9, np.full(3, 0.25)
background = np.ones(3)
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the margin around the polyhedra as a
# share of the larger side, and the pixels across each panel.
ELEVATION, AZIMUTH = np.radians(20.0), np.radians(30.0)
MARGIN, N_PIXELS = 0.25, 500
# The points around each circle and around a tube; the widths of the edges and the circles and the touching points'
# radius, in pixels.
CIRCLE_SAMPLES, N_AROUND = 64, 8
EDGE_WIDTH, CIRCLE_WIDTH, TOUCH_RADIUS = 2.6, 2.4, 3.5


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """x, y and z of each point on the last axis: its pairings with the planes x, y and z, over its weight.
    A point here is the dual of a plane, and w squares to minus one, so its weight is its pairing with -w."""
    mv = core.mv
    return ((stack((mv.x, mv.y, mv.z)) & points[..., None]) / (-mv.w & points)[..., None]).to_array()


def edge_distances(apexes: core.Point, edges: np.ndarray) -> np.ndarray:
    """The distance from the centre to the line of each edge between apexes."""
    xyz = coordinates(apexes)
    ends = xyz[..., edges[:, 0], :], xyz[..., edges[:, 1], :]
    return np.linalg.norm(np.cross(*ends), axis=-1) / np.linalg.norm(ends[1] - ends[0], axis=-1)


def beside(faces: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """The two faces either side of each edge, [edges, 2]."""
    sides = np.sort(np.stack([faces, np.roll(faces, -1, axis=1)], axis=-1), axis=-1).reshape(-1, 2)
    order = np.lexsort(sides.T[::-1])
    return (order // 3).reshape(-1, 2)[np.argsort(np.lexsort(edges.T[::-1]))]


def circles_on_sphere(circles: core.Vector, radius: float) -> np.ndarray:
    """Points around each circle on the sphere of a radius about the centre, [..., samples + 1, 3], closed: the
    plane `n | x == s` of a circle `n + s * w` cuts the unit sphere at the distance `s / |n|` along `n`."""
    mv = core.mv
    # A circle's normal part, its inner products with x, y and z, which square to one, and its part along w,
    # which squares to minus one.
    along = (circles[..., None] | stack((mv.x, mv.y, mv.z))).to_array()      # [..., 3]
    offset = -(circles | mv.w).to_array()                                      # [...]
    normal = along / np.linalg.norm(along, axis=-1, keepdims=True)
    height = offset / np.linalg.norm(along, axis=-1)
    return meshes.circles(radius * height[..., None] * normal, radius * np.sqrt(1 - height**2), normal, CIRCLE_SAMPLES)


def view(apexes: core.Point) -> solids.View:
    """A view framing the polyhedra of apexes, of any batch, and the sphere inside them."""
    return solids.view([sm.point(coordinates(apexes))], 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)


def panels(circles: core.Vector, faces: np.ndarray, edges: np.ndarray, frame: solids.View) -> tuple[np.ndarray, np.ndarray]:
    """Two images: the polyhedron of the circles' apexes and its dual, the polyhedron of the face circles' apexes,
    every edge of both touching the sphere where the two circles of its ends touch; and the vertex circles and the
    face circles on the sphere."""
    pixel = frame.pixel
    duals = core.face_circles(circles, faces)                                  # [faces] Vector
    apexes, dual_apexes = sm.point(coordinates(core.apexes(circles))), sm.point(coordinates(core.apexes(duals)))
    sides = beside(faces, edges)                                               # [edges, 2]
    sphere = (*solids.layer([(solids.ball(solids.origin, 1.0)[None, None], SPHERE_COLOUR)], frame), SPHERE_OPACITY)
    rods = solids.layer([(solids.rods(apexes[edges[:, 0]], apexes[edges[:, 1]], EDGE_WIDTH / 2 * pixel), PRIMAL_COLOUR),
                         (solids.rods(dual_apexes[sides[:, 0]], dual_apexes[sides[:, 1]], EDGE_WIDTH / 2 * pixel),
                          DUAL_COLOUR),
                         (solids.ball(sm.point(coordinates(core.touching(circles, edges).dual())),
                                      TOUCH_RADIUS * pixel)[:, None], TOUCH_COLOUR)], frame)
    # The circles drawn a tube's radius out, so that the tubes lie on the sphere rather than in it, each coloured by
    # its kind.
    lifted = 1 + CIRCLE_WIDTH / 2 * pixel
    loops = np.concatenate([circles_on_sphere(circles, lifted), circles_on_sphere(duals, lifted)])   # [circles, samples + 1, 3]
    kinds = np.repeat(np.stack([PRIMAL_COLOUR, DUAL_COLOUR]), [circles.shape[-1], duals.shape[-1]], axis=0)   # [circles, 3]
    tubes = meshes.tubes(list(zip(loops, kinds)), CIRCLE_WIDTH, frame, N_AROUND)
    return compose([sphere, (*rods, 1.0)], background), compose([sphere, (*tubes, 1.0)], background)


def draw(circles: core.Vector, faces: np.ndarray, edges: np.ndarray) -> plt.Figure:
    """Left, the polyhedron and its dual about the sphere; right, the vertex circles and the face circles on it."""
    frame = view(core.apexes(circles))
    figure = plt.figure(figsize=(2 * N_PIXELS / 100, frame.shape[0] / 100), dpi=100)
    for place, image in zip(((0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 0.5, 1.0)), panels(circles, faces, edges, frame)):
        show(figure.add_axes(place), image, frame)
    return figure


def animate(circles: core.Vector, faces: np.ndarray, edges: np.ndarray) -> list[np.ndarray]:
    """The two panels side by side at every frame of a batch of circles, seen by one camera framing them all."""
    frame = view(core.apexes(circles))
    return [np.concatenate(panels(circles[index], faces, edges, frame), axis=1) for index in range(circles.shape[0])]
