"""GPU rendering for the rational Bézier curve example draws the rays of a quadratic curve and its complement in homogeneous space, the cone they sweep, and their projection onto the unit sphere."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from matplotlib.colors import to_rgb

from numga import concatenate, stack, where
from examples.geometry.bezier import core
from examples.util.wgsl import meshes, solids, strokes

sm = solids.mv
curve_colour = np.array(to_rgb("#287fa0"))
complement_colour = np.array(to_rgb("#d06c36"))
control_colour, origin_colour = np.full(3, 0.55), np.full(3, 0.3)
equator_colour, loop_colour, ray_colour = np.full(3, 0.55), np.full(3, 0.68), np.full(3, 0.75)
# The sphere's colour and how much of what lies behind it each of its surfaces hides, and the swept rays' opacity.
sphere_colour, sphere_opacity, fan_opacity = np.full(3, 0.85), 0.11, 0.22
# The direction homogeneous space is viewed from, in degrees; the margin as a share of the larger side, and the
# pixels across a frame.
elevation, azimuth = 18.0, -62.0
margin, n_pixels = 0.06, 300
# The widths of the equator, the cone's loops and the control rays, of the curves and of the curves on the sphere;
# the spacing and radius of the dots along a control polygon, and the radius of a control point; all in pixels.
line_width, curve_width, sphere_curve_width = 1.8, 2.2, 3.5
dot_spacing, dot_radius, point_radius = 4.0, 1.0, 2.2


# --- plumbing -------------------------------------------------------------------------
def dotted(polygon: core.Point, spacing: float) -> core.Point:
    """The dots along an open polyline are spaced evenly along each side, as many as fit at the spacing, from its
    start up to the next side's; the last corner closes it."""
    sides = []
    for start, end in zip(polygon[:-1], polygon[1:]):
        count = max(1, round(float((end - start).norm().to_array() / spacing)))
        sides.append(start + (end - start) * np.linspace(0, 1, count, endpoint=False))
    return concatenate(sides + [polygon[-1:]])                                # [dots] Point-


def fan(n_samples: int, offset: int) -> np.ndarray:
    """The triangle fan connects the origin at vertex `offset` to each step along a curve of `n_samples` points."""
    steps = offset + 1 + np.arange(n_samples - 1)
    return np.stack([np.full_like(steps, offset), steps, steps + 1], axis=-1)


def draw_rays(controls: core.Point, curve: core.Point, complement: core.Point, shape: core.Conic,
              view: solids.View) -> np.ndarray:
    """Render a curve and its complement in homogeneous space with their weighted control points, the ray fans from the origin, and their projection onto the unit sphere."""
    arcs = [curve, complement]                                                # [samples] Point- each
    colours = [curve_colour, complement_colour]
    # Each weighted control point on its ray from the origin, the ray reaching out to the sphere where the point lies inside it.
    corners = controls.reshape(-1)                                            # [corners] Point-
    reaches = where(corners.norm() < 1, corners.normalized(), corners)        # [corners] Point-
    rays = solids.rods(solids.origin, solids.origin + solids.context.include_from_algebra(reaches),
                       line_width / 2 * view.pixel)                            # [corners, 3] Quadric
    # The curves in the renderer's space, each point a ray's place, and where each ray meets the unit sphere.
    places = [solids.origin + solids.context.include_from_algebra(arc) for arc in arcs]        # [samples] Point- each
    on_sphere_places = [solids.origin + solids.context.include_from_algebra(arc.normalized()) for arc in arcs]   # [samples] Point- each
    arc_strokes = strokes.polylines(list(zip(places, colours)), line_width, view)
    on_sphere = strokes.polylines(list(zip(on_sphere_places, colours)), sphere_curve_width, view)
    # The cone of the curve's rays about the origin reads each point of the renderer's space by its offset from the
    # origin, its part along the degenerate blades.
    cone = solids.context.include_from_algebra(shape)(solids.Point.restrict.degenerate())      # [] Plane+ <- Point-
    # The control polygons dotted, their corners and the origin as points.
    dots = concatenate([dotted(polygon, dot_spacing * view.pixel) for polygon in controls])   # [dots] Point-
    parts = [(rays, ray_colour),
             (solids.ball(solids.origin + solids.context.include_from_algebra(dots), dot_radius * view.pixel)[:, None], control_colour),
             (solids.ball(solids.origin + solids.context.include_from_algebra(corners), point_radius * view.pixel)[:, None], control_colour),
             (solids.ball(solids.origin, point_radius * view.pixel)[None, None], origin_colour)]
    sphere = [(solids.ball(solids.origin, 1.0)[None, None], sphere_colour, sphere_opacity)]
    # Inked on the sphere, the first translucent part after the opaque ones: the equator, where the plane of zero
    # weight cuts it, and the cone's two loops.
    inks = [solids.Ink(len(parts), solids.half_space(sm.z)[None], solids.everywhere[None], equator_colour, line_width, 1.0),
            solids.Ink(len(parts), cone[None], solids.everywhere[None], loop_colour, line_width, 1.0)]
    # The rays from the origin to each curve: a fan of triangles, the origin and two neighbouring samples each.
    vertices = concatenate([concatenate([solids.origin[None], arc]) for arc in places])   # [vertices] Point-
    faces = np.concatenate([fan(len(arcs[0]), 0), fan(len(arcs[1]), len(arcs[0]) + 1)])
    face_colours = np.concatenate([np.broadcast_to(colour, (len(arc) - 1, 3)) for arc, colour in zip(arcs, colours)])
    fans = meshes.samples(vertices, faces, face_colours[:, None], meshes.flat(vertices, faces), view,
                          face_values=np.zeros(4), colour=meshes.plain, gloss=solids.gloss)
    return solids.draw(parts, view, np.ones(3), inks, translucent=sphere,
                       meshes=[(fans, fan_opacity), (arc_strokes, 1.0), (on_sphere, 1.0)])


def views(points: list[core.Point], azimuths: np.ndarray) -> list[solids.View]:
    """The camera views orbit the origin across the given azimuths at a shared framing."""
    turns = np.radians(azimuths + 90)
    cameras = [solids.aim(solids.origin, np.radians(elevation), turn) for turn in turns]
    framed_points = solids.origin + solids.context.include_from_algebra(concatenate([batch.reshape(-1) for batch in points]))   # [points] Point-
    screens = stack(cameras)[None, :] >> solids.axes[:2, None]                # [2, views] Plane+
    readings = (screens[..., None] & framed_points).abs().reshape(2, -1)
    half = readings.max(axis=-1).clip(1, np.inf)
    half = half + margin * half.max()
    return [solids.View(camera, half.to_array(), n_pixels) for camera in cameras]


def animate_rays(controls: core.Point, curves: core.Point, complements: core.Point, shapes: core.Conic) -> Iterator[np.ndarray]:
    """Yield the rendered frame for each conic in the batch as the camera orbits once around the origin."""
    camera_views = views([curves, complements], azimuth + np.linspace(0, 360, curves.shape[0], endpoint=False))
    for index in range(curves.shape[0]):
        yield draw_rays(controls[index], curves[index], complements[index], shapes[index], camera_views[index])
