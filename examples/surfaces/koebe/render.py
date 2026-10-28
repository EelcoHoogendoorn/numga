"""A Koebe polyhedron and its dual about the sphere they both touch, and their circles on the sphere, drawn on the
GPU: the sphere traced, the edges rods, the circles inked on its surface where their planes cut it. The sphere all but
hides what lies behind it."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import concatenate, stack
from examples.surfaces.koebe import core
from examples.util.wgsl import show, solids

sm = solids.mv
PRIMAL_COLOUR = np.array(to_rgb("#287fa0"))
DUAL_COLOUR = np.array(to_rgb("#d06c36"))
# The sphere's colour and how much each of its surfaces hides; the touching points'
# colour.
SPHERE_COLOUR, SPHERE_OPACITY, TOUCH_COLOUR = np.full(3, 0.97), 0.75, np.full(3, 0.25)
background = np.ones(3)
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the margin around the polyhedra as a
# share of the larger side, and the pixels across each panel.
ELEVATION, AZIMUTH = np.radians(20.0), np.radians(30.0)
MARGIN, N_PIXELS = 0.25, 500
# The widths of the edges and the circles and the touching points' radius, in pixels.
EDGE_WIDTH, CIRCLE_WIDTH, TOUCH_RADIUS = 2.6, 2.4, 3.5


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """x, y and z of each point on the last axis: its pairings with the planes x, y and z, over its weight.
    A point here is the dual of a plane, and w squares to minus one, so its weight is its pairing with -w."""
    mv = core.mv
    return ((stack((mv.x, mv.y, mv.z)) & points[..., None]) / (-mv.w & points)[..., None]).to_array()


def circle_planes(circles: core.Vector) -> solids.Plane:
    """The plane of each circle on the unit sphere: a circle `n + s * w` lies where the plane `n | x == s` cuts it."""
    mv = core.mv
    # A circle's normal part, its inner products with x, y and z, which square to one, and its part along w,
    # which squares to minus one.
    along = (circles[..., None] | stack((mv.x, mv.y, mv.z))).to_array()      # [..., 3]
    offset = -(circles | mv.w).to_array()                                      # [...]
    # The plane through the centre square to the normal, moved out by the offset.
    return -sm.direction(along).dual() - sm.w * offset


def view(apexes: core.Point) -> solids.View:
    """A view framing the polyhedra of apexes, of any batch, and the sphere inside them."""
    return solids.view([sm.point(coordinates(apexes))], 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)


def panels(circles: core.Vector, faces: np.ndarray, edges: np.ndarray, frame: solids.View) -> tuple[np.ndarray, np.ndarray]:
    """Two images: the polyhedron of the circles' apexes and its dual, the polyhedron of the face circles' apexes,
    every edge of both touching the sphere where the two circles of its ends touch; and the vertex circles and the
    face circles on the sphere."""
    pixel = frame.pixel
    duals = core.face_circles(circles, faces)                                  # [faces] Vector-
    apexes, dual_apexes = sm.point(coordinates(core.apexes(circles))), sm.point(coordinates(core.apexes(duals)))
    sides = core.Mesh(circles.shape[-1], faces).edge_faces                     # [edges, 2]
    sphere = [(solids.ball(solids.origin, 1.0)[None, None], SPHERE_COLOUR, SPHERE_OPACITY)]
    marks = [(solids.rods(apexes[edges[:, 0]], apexes[edges[:, 1]], EDGE_WIDTH / 2 * pixel), PRIMAL_COLOUR),
             (solids.rods(dual_apexes[sides[:, 0]], dual_apexes[sides[:, 1]], EDGE_WIDTH / 2 * pixel), DUAL_COLOUR),
             (solids.ball(sm.point(coordinates(core.touching(circles, edges))),
                          TOUCH_RADIUS * pixel)[:, None], TOUCH_COLOUR)]
    # The circles inked on the sphere, the only part of the second image, each coloured by its kind.
    planes = concatenate([circle_planes(circles), circle_planes(duals)])      # [circles] Plane+
    kinds = np.repeat(np.stack([PRIMAL_COLOUR, DUAL_COLOUR]), [circles.shape[-1], duals.shape[-1]], axis=0)   # [circles, 3]
    loops = solids.Ink(0, solids.half_space(planes), solids.everywhere[None].broadcast_to(planes.shape), kinds,
                       CIRCLE_WIDTH, 1.0)
    return (solids.draw(marks, frame, background, inks=[], translucent=sphere, meshes=[]),
            solids.draw([], frame, background, [loops], translucent=sphere, meshes=[]))


def draw(circles: core.Vector, faces: np.ndarray, edges: np.ndarray) -> plt.Figure:
    """Left, the polyhedron and its dual about the sphere; right, the vertex circles and the face circles on it."""
    frame = view(core.apexes(circles))
    figure = plt.figure(figsize=(2 * N_PIXELS / 100, frame.shape[0] / 100), dpi=100)
    for place, image in zip(((0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 0.5, 1.0)), panels(circles, faces, edges, frame)):
        show(figure.add_axes(place), image, frame)
    return figure


def animate(circles: core.Vector, faces: np.ndarray, edges: np.ndarray) -> Iterator[np.ndarray]:
    """The two panels side by side at every frame of a batch of circles, seen by one camera framing them all."""
    frame = view(core.apexes(circles))
    for index in range(circles.shape[0]):
        yield np.concatenate(panels(circles[index], faces, edges, frame), axis=1)
