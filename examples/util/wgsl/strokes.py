"""Strokes: straight segments and polylines drawn as round capsules, rasterized on the GPU, among the meshes.

A segment's capsule is three solids bounded by quadrics, as `solids` traces them: the rod along it, its cylinder
between the half-spaces past its ends, and the balls about its ends. The vertex stage places a box on the image around
the capsule's outline, a radius beyond the segment every way; every sample of the box follows its ray, parallel to the
camera's, to where it first crosses the capsule's surface, and writes the depth it meets it at, so strokes cross each
other and the meshes where they truly do. Consecutive segments of a polyline share their ends, and the balls there
round its joints. The surface is lit as a mesh's is, by the lamp of `solids`, through the quadric's tangent plane.
"""

from __future__ import annotations

from functools import cache

import numpy as np

from numga import concatenate
from examples.util.wgsl import Shader
from examples.util.wgsl.meshes import Samples
from examples.util.wgsl.solids import (Motor, Plane, Point, Quadric, View, axes, ball, everywhere, ga, gloss,
                                       half_space, lamp_shading, lighting, mv, origin, quadric_crossings)

source = """
// Each segment's capsule, every part a quadric, `Plane <- Point`, about the segment moved to start at the origin: the
// rod along it, its cylinder and the half-spaces past its ends, and the ball about its end; the ball about its start is
// `ball`, about the origin, the same for all.
struct Capsule {
    tube: Quadric,
    beyond_end: Quadric,
    before_start: Quadric,
    end: Quadric,
}

// Each segment's start, and the direction from it to its end, as long as the segment.
var<storage, read> starts: array<Point>;
var<storage, read> spans: array<Point>;
var<storage, read> capsules: array<Capsule>;
var<storage, read> colours: array<vec4<f32>>;
var<uniform> radius: f32;
var<uniform> ball: Quadric;
// The camera of `solids`: looking along its own -z, its rays parallel, the image half as wide and high as these; its
// frame, `axes`, given from Python.
var<uniform> camera: Motor;
var<uniform> half_size: vec2<f32>;
var<uniform> depth: f32;

struct Corner {
    @builtin(position) position: vec4<f32>,
    // Where the sample's ray starts, in front of the capsule, about the segment moved to start at the origin, as its
    // capsule is; the segment; and how far toward the viewer its rays start.
    @location(0) @interpolate(linear, sample) start: Point,
    @location(1) @interpolate(flat) segment: u32,
    @location(2) @interpolate(flat) front: f32,
}

// A sample's colour, its depth along the view in the depth target's first channel, and its depth for the test.
struct Shade {
    @location(0) colour: vec4<f32>,
    @location(1) depth: vec4<f32>,
    @builtin(frag_depth) nearness: f32,
}

// A point in the camera's frame, at unit weight.
fn seen(point: Point) -> Point {
    let framed = camera << point;
    return framed / (mv.w & framed);
}

// A corner of the box around a segment's capsule on the image, six to a segment, two triangles.
@vertex
fn place(@builtin(vertex_index) index: u32) -> Corner {
    let segment = index / 6u;
    let start = seen(starts[segment]);
    let span = camera << spans[segment];
    let end = start + span;
    // Along the segment's image and across it, directions in the image: the segment's direction less its part along
    // the view, at unit length, and that turned a quarter about the view; any way along for a segment seen end on.
    let imaged = span - mv.z.dual() * (mv.z & span);
    let extent = imaged.dual().norm();
    let along = select(Point(mv.x.dual()), imaged / extent, extent > 0);
    let across = quarter_turn >> along;
    var corners = array<vec2<f32>, 6>(vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
                                      vec2<f32>(0.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0));
    let corner = corners[index % 6u];
    let base = select(start - along * radius, end + along * radius, corner.x > 0.5);
    let place = base + across * (radius * corner.y);
    // The capsule's nearest reach toward the viewer, and the plane square to the view a radius nearer, where its rays
    // start.
    let nearest = max(mv.z & start, mv.z & end) + radius;
    let front = nearest + radius;
    let framed = axes & place;
    let position = vec4<f32>(framed.xy / half_size, 0.5 - nearest / depth, 1.0);
    // The ray's start at the corner: its offset from the segment's start, carried out of the camera's frame.
    let ray_start = origin + (camera >> (place + mv.z.dual() * (front - (mv.z & place)) - start));
    return Corner(position, ray_start, segment, front);
}

// Where a ray meets a surface: how far along it, and the surface's tangent plane there.
struct Met {
    distance: f32,
    tangent: Plane,
}

// The nearer of a surface met and where a ray crosses a quadric's surface ahead of its start, inside two other
// quadrics, a rod's half-spaces or all of space: the quadric's plane at a point of its surface is the tangent plane.
fn nearer(met: Met, quadric: Quadric, first_bound: Quadric, second_bound: Quadric, start: Point,
          direction: Point) -> Met {
    let along = along_ray(quadric, start, direction);
    if (along.b * along.b - along.a * along.c <= 0) {
        return met;
    }
    var nearest = met;
    let roots = crossings(along);
    for (var which = 0u; which < 2u; which++) {
        let distance = roots[which];
        let point = start + direction * distance;
        let inside = (first_bound(point) & point) < 0 && (second_bound(point) & point) < 0;
        if (distance > 0 && distance < nearest.distance && inside) {
            nearest = Met(distance, quadric(point));
        }
    }
    return nearest;
}

// A sample of the box: where its ray first meets the capsule, lit there, or nothing where it passes by.
@fragment
fn shade(corner: Corner) -> Shade {
    let capsule = capsules[corner.segment];
    let direction = camera >> -mv.z.dual();
    // The nearest of the rod's cylinder, between its ends, and the balls at the ends that the ray meets.
    var met = nearer(Met(far, mv.z), capsule.tube, capsule.beyond_end, capsule.before_start, corner.start, direction);
    met = nearer(met, ball, everywhere, everywhere, corner.start, direction);
    met = nearer(met, capsule.end, everywhere, everywhere, corner.start, direction);
    if (met.distance >= far) {
        discard;
    }
    // Its depth along the view: how far along the ray it lies, less how far toward the viewer the ray starts.
    let along_view = met.distance - corner.front;
    return Shade(vec4<f32>(lit(colours[corner.segment].rgb, met.tangent, direction), 1.0),
                 vec4<f32>(along_view, 0.0, 0.0, 1.0), 0.5 + along_view / depth);
}
"""


# --- plumbing -------------------------------------------------------------------------
# The quarter turn about the view, carrying the image's x onto its y.
quarter_turn = (mv.xy * (-np.pi / 4)).exp()                                  # [] Motor


@cache
def shader() -> Shader:
    # The colour a byte a channel, and the depth alone at full precision, as meshes write them.
    return Shader(quadric_crossings() + lamp_shading() + source, ga,
                  dict(Point=Point, Plane=Plane, Motor=Motor, Quadric=Quadric),
                  dict(axes=axes, origin=origin, everywhere=everywhere, quarter_turn=quarter_turn),
                  targets=("rgba8unorm", "r32float"))


def segments(starts: Point, ends: Point, colours: np.ndarray, width: float, view: View) -> Samples:
    """Straight segments between points, of any weight, `[segments]`, coloured `[segments, 3]` or one for all, as
    capsules `width` pixels across, at every sample of the view's pixels, for `solids.layer` and `Scene` to compose."""
    radius = width / 2 * view.pixel
    # The direction from each segment's start to its end, as long as the segment.
    spans = ends / (mv.w & ends) - starts / (mv.w & starts)                   # [segments] Point
    # Each capsule about its segment moved to start at the origin, where its quadrics keep their precision however far
    # the segment lies from it: where each segment reaches from the origin, so moved.
    reaches = origin + spans                                                  # [segments] Point
    # The plane through the origin square to each segment, as long as the segment is: a point's pairing with it is
    # how far along the segment the point lies, times the segment's length.
    square = -spans.dual()                                                    # [segments] Plane
    # The rod's cylinder: a point's squared distance from the segment's line, against the radius squared, both
    # scaled by the segment's length squared, so that a segment of no length has no cylinder and its capsule is its
    # ball; and the half-spaces past its ends, which then hold no point.
    tube = ball(origin, radius) * square.norm_squared() - square * (square & Point) / radius**2   # [segments] Quadric
    capsules = dict(tube=tube, beyond_end=half_space(square - mv.w * (square & reaches)),
                    before_start=half_space(-square), end=ball(reaches, radius))
    shades = np.concatenate([np.broadcast_to(colours, starts.shape + (3,)), np.ones(starts.shape + (1,))], axis=-1)
    return Samples(shader(), 6 * len(starts), dict(
        starts=starts, spans=spans, capsules=capsules, colours=shades, radius=radius, ball=ball(origin, radius),
        camera=view.camera, half_size=view.half, depth=8 * view.half.max(), **lighting(view.camera, gloss)))


def polylines(curves: list[tuple[Point, np.ndarray]], width: float, view: View) -> Samples:
    """Polylines of points `[n]`, each with its colour `[3]`, as strokes `width` pixels across: a capsule along each of
    their segments, the balls at the shared ends rounding the joints."""
    starts = concatenate([points[:-1] for points, _ in curves])
    ends = concatenate([points[1:] for points, _ in curves])
    colours = np.concatenate([np.repeat(np.asarray(colour)[None], len(points) - 1, axis=0) for points, colour in curves])
    return segments(starts, ends, colours, width, view)
