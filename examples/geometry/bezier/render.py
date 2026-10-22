"""The rays of a quadratic curve and its complement in the three dimensions of x, y and weight, the cone they
sweep, and where both meet the unit sphere: drawn on the GPU, the sphere traced and translucent, the curves and loops
tubes, the swept rays a translucent mesh."""

from __future__ import annotations

import contourpy
import numpy as np
from matplotlib.colors import to_rgb

from examples.geometry.bezier import core
from examples.wgsl import compose, meshes, solids

sm = solids.mv
CURVE_COLOUR = np.array(to_rgb("#287fa0"))
COMPLEMENT_COLOUR = np.array(to_rgb("#d06c36"))
CONTROL_COLOUR, ORIGIN_COLOUR = np.full(3, 0.55), np.full(3, 0.3)
EQUATOR_COLOUR, LOOP_COLOUR, RAY_COLOUR = np.full(3, 0.55), np.full(3, 0.68), np.full(3, 0.75)
SPHERE_COLOUR, SPHERE_OPACITY, FAN_OPACITY = np.full(3, 0.85), 0.2, 0.22
# The samples of a circle, and of the grid the cone is found on.
CIRCLE = 200
SPHERE_GRID = 200
# The direction homogeneous space is viewed from, in degrees; the margin as a share of the larger side, and the
# pixels across a frame.
ELEVATION, AZIMUTH = 18.0, -62.0
MARGIN, N_PIXELS = 0.06, 600
# The widths of the equator, the cone's loops and the control rays, of the curves and of the curves on the sphere;
# the spacing and radius of the dots along a control polygon, and the radius of a control point; all in pixels.
LINE_WIDTH, CURVE_WIDTH, SPHERE_CURVE_WIDTH = 3.0, 3.5, 6.0
DOT_SPACING, DOT_RADIUS, POINT_RADIUS = 7.0, 1.4, 3.5


# --- plumbing -------------------------------------------------------------------------
def scalars(values: core.Scalar) -> np.ndarray:
    """The values of a batch of scalars, for printing and checks."""
    return values.cast(values.algebra.subspace.scalar()).kernel[..., 0]


def homogeneous(points: core.Point) -> np.ndarray:
    """x, y and weight of each point, on the last axis."""
    return points.cast(core.ga.subspace("yz zx xy")).kernel


def cones(shapes: core.Conic) -> list[list[np.ndarray]]:
    """Where each conic's form vanishes on the unit sphere, for a batch of conics: the loops its cone cuts,
    each an array of unit vectors in x, y and weight."""
    longitude, latitude = np.meshgrid(np.linspace(-np.pi, np.pi, SPHERE_GRID), np.linspace(-np.pi / 2, np.pi / 2, SPHERE_GRID))
    xyw = np.stack([np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude)], axis=-1)
    grid = core.mv.vector(xyw).dual()[..., None]                               # [grid, grid, 1] Point
    values = np.moveaxis(scalars(shapes(grid) & grid), -1, 0)                  # [conics, grid, grid]
    return [[np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)
             for lon, lat in (loop.T for loop in contourpy.contour_generator(longitude, latitude, value).lines(0.0))]
            for value in values]


def dotted(polygon: np.ndarray, spacing: float) -> np.ndarray:
    """Dots along a polyline `[corners, 3]`, about `spacing` apart, `[dots, 3]`."""
    # Along each side, as many dots as fit at the spacing, from its start up to the next side's; the last corner closes it.
    counts = np.maximum(1, np.round(np.linalg.norm(np.diff(polygon, axis=0), axis=-1) / spacing).astype(int))
    return np.concatenate([start + np.linspace(0, 1, count, endpoint=False)[:, None] * (end - start)
                           for start, end, count in zip(polygon[:-1], polygon[1:], counts)] + [polygon[-1:]])


def fan(xyw: np.ndarray, offset: int) -> np.ndarray:
    """The triangles from the origin, vertex `offset`, to each step along a curve, its vertices following it."""
    steps = offset + 1 + np.arange(len(xyw) - 1)
    return np.stack([np.full_like(steps, offset), steps, steps + 1], axis=-1)


def draw_rays(controls: core.Point, curve: core.Point, complement: core.Point, loops: list[np.ndarray],
              view: solids.View) -> np.ndarray:
    """A curve and its complement as their sums are, in the three dimensions of x, y and weight, with their
    weighted control points, and with the rays from the origin to the curves filled in: the part of the
    conic's cone each sweeps. Where the rays pierce
    the unit sphere, the curves normalised, and where the cone does, its two antipodal loops. The equator
    is the line at infinity."""
    sphere = solids.layer([(solids.ball(solids.origin, 1.0)[None, None], SPHERE_COLOUR)], view)
    polygons = homogeneous(controls)                                           # [2, controls, 3]
    arcs = [homogeneous(curve), homogeneous(complement)]                       # [samples, 3] each
    colours = [CURVE_COLOUR, COMPLEMENT_COLOUR]
    # Each weighted control point on its ray from the origin, the ray reaching out to the sphere where the point lies inside it.
    corners = polygons.reshape(-1, 3)
    reaches = corners * np.maximum(1.0, 1 / np.linalg.norm(corners, axis=-1, keepdims=True))
    equator = meshes.circles(np.zeros(3), np.array(1.0), np.array([0.0, 0.0, 1.0]), CIRCLE)
    lines = meshes.tubes([(equator, EQUATOR_COLOUR)] + [(loop, LOOP_COLOUR) for loop in loops]
                         + [(np.stack([np.zeros(3), reach]), RAY_COLOUR) for reach in reaches]
                         + [(xyw, colour) for xyw, colour in zip(arcs, colours)], LINE_WIDTH, view)
    on_sphere = meshes.tubes([(xyw / np.linalg.norm(xyw, axis=-1, keepdims=True), colour) for xyw, colour in zip(arcs, colours)],
                             SPHERE_CURVE_WIDTH, view)
    # The control polygons dotted, their corners and the origin as points.
    dots = sm.point(np.concatenate([dotted(polygon, DOT_SPACING * view.pixel) for polygon in polygons]))
    points = solids.layer([(solids.ball(dots, DOT_RADIUS * view.pixel)[:, None], CONTROL_COLOUR),
                           (solids.ball(sm.point(corners), POINT_RADIUS * view.pixel)[:, None], CONTROL_COLOUR),
                           (solids.ball(solids.origin, POINT_RADIUS * view.pixel)[None, None], ORIGIN_COLOUR)], view)
    # The rays from the origin to each curve: a fan of triangles, the origin and two neighbouring samples each.
    vertices = sm.point(np.concatenate([np.concatenate([np.zeros((1, 3)), xyw]) for xyw in arcs]))
    faces = np.concatenate([fan(arcs[0], 0), fan(arcs[1], len(arcs[0]) + 1)])
    face_colours = np.concatenate([np.broadcast_to(colour, (len(xyw) - 1, 3)) for xyw, colour in zip(arcs, colours)])
    fans = meshes.layer(vertices, faces, face_colours[:, None], meshes.flat(vertices, faces), view)
    return compose([(*sphere, SPHERE_OPACITY), (*fans, FAN_OPACITY), (*lines, 1.0), (*on_sphere, 1.0), (*points, 1.0)],
                   np.ones(3))


def views(points: list[core.Point], azimuths: np.ndarray) -> list[solids.View]:
    """Views at the origin from each azimuth, in degrees, one size for all: framing every point of the batches, as
    its sum is, and the unit sphere, from every azimuth."""
    elevation, turns = np.radians(ELEVATION), np.radians(azimuths + 90)
    cameras = [solids.aim(solids.origin, elevation, turn) for turn in turns]
    seen = sm.point(np.concatenate([homogeneous(batch).reshape(-1, 3) for batch in points]))
    places = np.concatenate([solids.View(camera, np.ones(2), N_PIXELS).across(seen) for camera in cameras])
    half = np.maximum(np.abs(places).max(axis=0), 1.0)
    return [solids.View(camera, half + MARGIN * half.max(), N_PIXELS) for camera in cameras]


def animate_rays(controls: core.Point, curves: core.Point, complements: core.Point, shapes: core.Conic) -> list[np.ndarray]:
    """The ray figure for each conic of a batch, viewed from turning once around, at one framing."""
    loops = cones(shapes)
    seen = views([curves, complements, controls], AZIMUTH + np.linspace(0, 360, curves.shape[0], endpoint=False))
    return [draw_rays(controls[index], curves[index], complements[index], loops[index], seen[index])
            for index in range(curves.shape[0])]
