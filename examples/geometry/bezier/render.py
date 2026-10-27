"""GPU rendering for the rational Bézier curve example draws the rays of a quadratic curve and its complement in homogeneous space, the cone they sweep, and their projection onto the unit sphere."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
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
def homogeneous(points: core.Point) -> np.ndarray:
    """The x, y, and weight coordinates of each point are its pairings with the three coordinate planes."""
    return (stack((core.mv.x, core.mv.y, core.mv.z)) & points[..., None]).to_array()


def cone(shape: core.Conic) -> solids.Quadric:
    """A conic's form in the drawing's space is the quadric cone with the same pairings on the three coordinate rays."""
    rays = core.mv.vector(np.eye(3)).dual()                                    # [3] Point-
    form = (shape(rays[:, None]) & rays[None, :]).to_array()                   # [3, 3]
    return (solids.axes[:, None] * (solids.axes & solids.Point)[None, :] * form).sum(axis=(-1, -2))


def dotted(polygon: np.ndarray, spacing: float) -> np.ndarray:
    """The dots along a closed polyline are spaced evenly along each edge."""
    # Along each side, as many dots as fit at the spacing, from its start up to the next side's; the last corner closes it.
    counts = np.maximum(1, np.round(np.linalg.norm(np.diff(polygon, axis=0), axis=-1) / spacing).astype(int))
    return np.concatenate([start + np.linspace(0, 1, count, endpoint=False)[:, None] * (end - start)
                           for start, end, count in zip(polygon[:-1], polygon[1:], counts)] + [polygon[-1:]])


def fan(xyw: np.ndarray, offset: int) -> np.ndarray:
    """The triangle fan connects the origin at vertex `offset` to each step along a curve."""
    steps = offset + 1 + np.arange(len(xyw) - 1)
    return np.stack([np.full_like(steps, offset), steps, steps + 1], axis=-1)


def draw_rays(controls: core.Point, curve: core.Point, complement: core.Point, shape: core.Conic,
              view: solids.View) -> np.ndarray:
    """Render a curve and its complement in homogeneous space with their weighted control points, the ray fans from the origin, and their projection onto the unit sphere."""
    polygons = homogeneous(controls)                                           # [2, controls, 3]
    arcs = [homogeneous(curve), homogeneous(complement)]                       # [samples, 3] each
    colours = [CURVE_COLOUR, COMPLEMENT_COLOUR]
    # Each weighted control point on its ray from the origin, the ray reaching out to the sphere where the point lies inside it.
    corners = polygons.reshape(-1, 3)
    reaches = corners * np.maximum(1.0, 1 / np.linalg.norm(corners, axis=-1, keepdims=True))
    rays = solids.rods(sm.point(np.zeros_like(reaches)), sm.point(reaches), LINE_WIDTH / 2 * view.pixel)   # [corners, 3] Quadric
    arc_strokes = strokes.polylines([(sm.point(xyw), colour) for xyw, colour in zip(arcs, colours)], LINE_WIDTH, view)
    on_sphere = strokes.polylines([(sm.point(xyw / np.linalg.norm(xyw, axis=-1, keepdims=True)), colour)
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
    fans = meshes.samples(vertices, faces, face_colours[:, None], meshes.flat(vertices, faces), view,
                          face_values=np.zeros(4), colour=meshes.plain, gloss=solids.gloss)
    return solids.draw(parts, view, np.ones(3), inks, translucent=sphere,
                       meshes=[(fans, FAN_OPACITY), (arc_strokes, 1.0), (on_sphere, 1.0)])


def views(points: list[core.Point], azimuths: np.ndarray) -> list[solids.View]:
    """The camera views orbit the origin across the given azimuths at a shared framing."""
    elevation, turns = np.radians(ELEVATION), np.radians(azimuths + 90)
    cameras = [solids.aim(solids.origin, elevation, turn) for turn in turns]
    framed_points = sm.point(np.concatenate([homogeneous(batch).reshape(-1, 3) for batch in points]))
    places = np.concatenate([solids.View(camera, np.ones(2), N_PIXELS).across(framed_points) for camera in cameras])
    half = np.maximum(np.abs(places).max(axis=0), 1.0)
    return [solids.View(camera, half + MARGIN * half.max(), N_PIXELS) for camera in cameras]


def animate_rays(controls: core.Point, curves: core.Point, complements: core.Point, shapes: core.Conic) -> Iterator[np.ndarray]:
    """Yield the rendered frame for each conic in the batch as the camera orbits once around the origin."""
    camera_views = views([curves, complements], AZIMUTH + np.linspace(0, 360, curves.shape[0], endpoint=False))
    for index in range(curves.shape[0]):
        yield draw_rays(controls[index], curves[index], complements[index], shapes[index], camera_views[index])
