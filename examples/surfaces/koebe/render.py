"""A Koebe polyhedron and its dual about the sphere they both touch, and their circles on the sphere, drawn on the
GPU: the sphere traced, the edges rods, the circles inked on its surface where their planes cut it. The sphere is
opaque and hides what lies behind it."""

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
# The sphere's colour, and the touching points' colour.
SPHERE_COLOUR, TOUCH_COLOUR = np.full(3, 0.97), np.full(3, 0.25)
background = np.ones(3)
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the margin around the polyhedra as a
# share of the larger side, and the pixels across each panel of the figure and of the animation.
ELEVATION, AZIMUTH = np.radians(20.0), np.radians(30.0)
MARGIN, N_PIXELS, N_ANIMATION_PIXELS = 0.25, 500, 250
# The widths of the edges and the circles and the touching points' radius, in pixels of a panel N_PIXELS across.
EDGE_WIDTH, CIRCLE_WIDTH, TOUCH_RADIUS = 2.6, 2.4, 3.5


# --- plumbing -------------------------------------------------------------------------
def gpu_points(points: core.Point) -> solids.Point:
    """View the rays in the projective slice read by -w; the spatial slices become the scene's frame."""
    mv = core.mv
    points = points.batch()
    readings = (stack((mv.x, mv.y, mv.z)) & points[..., None]) / (-mv.w & points)[..., None]
    return solids.origin + (solids.axes.dual() * solids.context.include_from_algebra(readings)).sum(axis=-1)



def gpu_planes(circles: core.Vector) -> solids.Plane:
    """Carry each circle's linear form into the scene, including its offset from the sphere's centre."""
    mv = core.mv
    frame_points = stack((mv.x, mv.y, mv.z, mv.w)).dual()
    readings = circles.batch()[..., None] & frame_points
    frame = stack((sm.x, sm.y, sm.z, sm.w))
    return (frame * solids.context.include_from_algebra(readings)).sum(axis=-1)



def view(apexes: core.Point, n_pixels: int) -> solids.View:
    """A view framing the polyhedra of apexes, of any batch, and the sphere inside them, `n_pixels` across."""
    return solids.view([gpu_points(apexes)], 0.0, ELEVATION, AZIMUTH, MARGIN, n_pixels)


def panels(mesh: core.Mesh, circles: core.Vector, frame: solids.View) -> tuple[np.ndarray, np.ndarray]:
    """Two images: the polyhedron of the circles' apexes and its dual, the polyhedron of the face circles' apexes,
    every edge of both touching the sphere where the two circles of its ends touch; and the vertex circles and the
    face circles on the sphere; the widths scale with the panel."""
    scale = frame.pixels / N_PIXELS
    pixel = frame.pixel * scale
    duals = core.face_circles(mesh, circles)                                   # Vector-[faces]
    # Each edge between the apexes of the circles at its ends, and each dual edge between the apexes of the face
    # circles on either side of it.
    tails, heads = mesh.incidence[1, 0] * core.apexes(circles)                 # Point-[edges], Point-[edges]
    left, right = mesh.reading(mesh.edge_faces.T, 2) * core.apexes(duals)      # Point-[edges], Point-[edges]
    sphere = (solids.ball(solids.origin, 1.0)[None, None], SPHERE_COLOUR)
    marks = [sphere, (solids.rods(gpu_points(tails), gpu_points(heads), EDGE_WIDTH / 2 * pixel), PRIMAL_COLOUR),
             (solids.rods(gpu_points(left), gpu_points(right), EDGE_WIDTH / 2 * pixel), DUAL_COLOUR),
             (solids.ball(gpu_points(core.touching(mesh, circles)), TOUCH_RADIUS * pixel)[:, None], TOUCH_COLOUR)]
    # The circles inked on the sphere, the only part of the second image, each coloured by its kind.
    # The face windings become forms for the ink; multiplication by the intrinsic unit changes their orientation.
    dual_forms = duals * core.mv.antipseudoscalar([1])                        # Vector+[faces]
    planes = concatenate([gpu_planes(circles), gpu_planes(dual_forms)])       # [circles] Plane+
    kinds = np.repeat(np.stack([PRIMAL_COLOUR, DUAL_COLOUR]), [mesh.n_vertices, len(mesh.faces)], axis=0)   # [circles, 3]
    loops = solids.Ink(0, solids.half_space(planes), solids.everywhere[None].broadcast_to(planes.shape), kinds,
                       CIRCLE_WIDTH * scale, 1.0)
    return (solids.draw(marks, frame, background, inks=[], translucent=[], meshes=[]),
            solids.draw([sphere], frame, background, [loops], translucent=[], meshes=[]))


def draw(mesh: core.Mesh, circles: core.Vector) -> plt.Figure:
    """Left, the polyhedron and its dual about the sphere; right, the vertex circles and the face circles on it."""
    frame = view(core.apexes(circles), N_PIXELS)
    figure = plt.figure(figsize=(2 * N_PIXELS / 100, frame.shape[0] / 100), dpi=100)
    for place, image in zip(((0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 0.5, 1.0)), panels(mesh, circles, frame)):
        show(figure.add_axes(place), image, frame)
    return figure


def animate(mesh: core.Mesh, circles: core.Vector) -> Iterator[np.ndarray]:
    """The two panels side by side at every frame of a batch of circles, seen by one camera framing them all."""
    frame = view(core.apexes(circles), N_ANIMATION_PIXELS)
    for index in range(circles.shape[0]):
        yield np.concatenate(panels(mesh, circles[index], frame), axis=1)
