"""The rays of a quadratic curve and its complement in the three dimensions of x, y and weight, the cone they
sweep, and where both meet the unit sphere: drawn on the GPU, the sphere traced and translucent with the equator and
the cone's loops inked on it, the control rays rods, the curves strokes, the swept rays a translucent mesh."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from matplotlib.colors import to_rgb

from examples.geometry.bezier import core
from examples.util.wgsl import meshes, solids, strokes

sm = solids.mv
CURVE_COLOUR = np.array(to_rgb("#287fa0"))
COMPLEMENT_COLOUR = np.array(to_rgb("#d06c36"))
CONTROL_COLOUR, ORIGIN_COLOUR = np.full(3, 0.55), np.full(3, 0.3)
EQUATOR_COLOUR, LOOP_COLOUR, RAY_COLOUR = np.full(3, 0.55), np.full(3, 0.68), np.full(3, 0.75)
# The sphere's colour and how much of what lies behind it each of its surfaces hides, and the swept rays' opacity.
SPHERE_COLOUR, SPHERE_OPACITY, FAN_OPACITY = np.full(3, 0.85), 0.11, 0.22
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


def cone(shape: core.Conic) -> solids.Quadric:
    """A conic's cone in the drawing's space, x, y and weight along x, y and z: its form read on each pair of the
    axes' rays, `shape(a) & b`, as the quadric pairing those coordinates of a point alike."""
    rays = core.mv.vector(np.eye(3)).dual()                                    # [3] Point
    form = scalars(shape(rays[:, None]) & rays[None, :])                       # [3, 3]
    return (solids.axes[:, None] * (solids.axes & solids.Point)[None, :] * form).sum(axis=(-1, -2))


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


def draw_rays(controls: core.Point, curve: core.Point, complement: core.Point, shape: core.Conic,
              view: solids.View) -> np.ndarray:
    """A curve and its complement as their sums are, in the three dimensions of x, y and weight, with their
    weighted control points, and with the rays from the origin to the curves filled in: the part of the
    conic's cone each sweeps. Where the rays pierce
    the unit sphere, the curves normalised, and where the cone does, its two antipodal loops. The equator
    is the line at infinity."""
    polygons = homogeneous(controls)                                           # [2, controls, 3]
    arcs = [homogeneous(curve), homogeneous(complement)]                       # [samples, 3] each
    colours = [CURVE_COLOUR, COMPLEMENT_COLOUR]
    # Each weighted control point on its ray from the origin, the ray reaching out to the sphere where the point lies inside it.
    corners = polygons.reshape(-1, 3)
    reaches = corners * np.maximum(1.0, 1 / np.linalg.norm(corners, axis=-1, keepdims=True))
    rays = solids.rods(sm.point(np.zeros_like(reaches)), sm.point(reaches), LINE_WIDTH / 2 * view.pixel)   # [corners, 3] Quadric
    arc_strokes = strokes.polylines([(xyw, colour) for xyw, colour in zip(arcs, colours)], LINE_WIDTH, view)
    on_sphere = strokes.polylines([(xyw / np.linalg.norm(xyw, axis=-1, keepdims=True), colour)
                                   for xyw, colour in zip(arcs, colours)], SPHERE_CURVE_WIDTH, view)
    # The control polygons dotted, their corners and the origin as points.
    dots = sm.point(np.concatenate([dotted(polygon, DOT_SPACING * view.pixel) for polygon in polygons]))
    parts = [(rays, RAY_COLOUR),
             (solids.ball(dots, DOT_RADIUS * view.pixel)[:, None], CONTROL_COLOUR),
             (solids.ball(sm.point(corners), POINT_RADIUS * view.pixel)[:, None], CONTROL_COLOUR),
             (solids.ball(solids.origin, POINT_RADIUS * view.pixel)[None, None], ORIGIN_COLOUR)]
    sphere = [(solids.ball(solids.origin, 1.0)[None, None], SPHERE_COLOUR, SPHERE_OPACITY)]
    # Inked on the sphere, the first translucent part after the opaque ones: the equator, where the plane of zero
    # weight cuts it, and the cone's two loops.
    inks = [solids.Ink(len(parts), solids.half_space(sm.z)[None], solids.everywhere[None], EQUATOR_COLOUR, LINE_WIDTH, 1.0),
            solids.Ink(len(parts), cone(shape)[None], solids.everywhere[None], LOOP_COLOUR, LINE_WIDTH, 1.0)]
    # The rays from the origin to each curve: a fan of triangles, the origin and two neighbouring samples each.
    vertices = sm.point(np.concatenate([np.concatenate([np.zeros((1, 3)), xyw]) for xyw in arcs]))
    faces = np.concatenate([fan(arcs[0], 0), fan(arcs[1], len(arcs[0]) + 1)])
    face_colours = np.concatenate([np.broadcast_to(colour, (len(xyw) - 1, 3)) for xyw, colour in zip(arcs, colours)])
    fans = meshes.samples(vertices, faces, face_colours[:, None], meshes.flat(vertices, faces), view)
    return solids.draw(parts, view, np.ones(3), inks, translucent=sphere,
                       meshes=[(fans, FAN_OPACITY), (arc_strokes, 1.0), (on_sphere, 1.0)])


def views(points: list[core.Point], azimuths: np.ndarray) -> list[solids.View]:
    """Views at the origin from each azimuth, in degrees, one size for all: framing every point of the batches, as
    its sum is, and the unit sphere, from every azimuth."""
    elevation, turns = np.radians(ELEVATION), np.radians(azimuths + 90)
    cameras = [solids.aim(solids.origin, elevation, turn) for turn in turns]
    seen = sm.point(np.concatenate([homogeneous(batch).reshape(-1, 3) for batch in points]))
    places = np.concatenate([solids.View(camera, np.ones(2), N_PIXELS).across(seen) for camera in cameras])
    half = np.maximum(np.abs(places).max(axis=0), 1.0)
    return [solids.View(camera, half + MARGIN * half.max(), N_PIXELS) for camera in cameras]


def animate_rays(controls: core.Point, curves: core.Point, complements: core.Point, shapes: core.Conic) -> Iterator[np.ndarray]:
    """The ray figure for each conic of a batch, viewed from turning once around, at one framing: the curves and the
    sphere's, which the weighted control points of the most hyperbolic conics run out of."""
    seen = views([curves, complements], AZIMUTH + np.linspace(0, 360, curves.shape[0], endpoint=False))
    for index in range(curves.shape[0]):
        yield draw_rays(controls[index], curves[index], complements[index], shapes[index], seen[index])
