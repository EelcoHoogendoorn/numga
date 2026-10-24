"""Strokes: straight segments and polylines drawn as round capsules, rasterized on the GPU, among the meshes.

Only each segment's two ends and its colour go to the GPU. The vertex stage places a box on the image around the
capsule's outline, a radius beyond the segment every way; every sample of the box follows its ray, parallel to the
camera's, to the capsule's surface, the cylinder about the segment or the ball about either end, and writes the depth
it meets it at, so strokes cross each other and the meshes where they truly do. Consecutive segments of a polyline
share their ends, and the balls there round its joints. The surface is lit as a mesh's is, by the lamp of `solids`.
"""

from __future__ import annotations

from functools import cache

import numpy as np

from examples.util.wgsl import Shader
from examples.util.wgsl.meshes import Samples
from examples.util.wgsl.solids import Motor, Plane, Point, View, axes, ga, gloss, lamp_shading, lighting, mv

source = """
var<storage, read> starts: array<Point>;
var<storage, read> ends: array<Point>;
var<storage, read> colours: array<vec4<f32>>;
var<uniform> radius: f32;
// The camera of `solids`: looking along its own -z, its rays parallel, the image half as wide and high as these; its
// frame, `axes`, given from Python.
var<uniform> camera: Motor;
var<uniform> half_size: vec2<f32>;
var<uniform> depth: f32;

struct Corner {
    @builtin(position) position: vec4<f32>,
    // Where the sample lies across the camera's view, and the segment's ends and colour, in the camera's frame.
    @location(0) @interpolate(linear, sample) place: vec2<f32>,
    @location(1) @interpolate(flat) start: vec3<f32>,
    @location(2) @interpolate(flat) end: vec3<f32>,
    @location(3) @interpolate(flat) colour: vec4<f32>,
}

// A sample's colour, its depth along the view in the depth target's first channel, and its depth for the test.
struct Shade {
    @location(0) colour: vec4<f32>,
    @location(1) depth: vec4<f32>,
    @builtin(frag_depth) nearness: f32,
}

// A point in the camera's frame: its pairings with the camera's planes, over its weight.
fn framed(point: Point) -> vec3<f32> {
    let seen = camera << point;
    return (axes & seen) / (mv.w & seen);
}

// A corner of the box around a segment's capsule on the image, six to a segment, two triangles.
@vertex
fn place(@builtin(vertex_index) index: u32) -> Corner {
    let segment = index / 6u;
    let start = framed(starts[segment]);
    let end = framed(ends[segment]);
    // Along the segment's image and across it; any way along for a segment seen end on.
    let span = end.xy - start.xy;
    let extent = sqrt(dot(span, span));
    let along = select(vec2<f32>(1.0, 0.0), span / extent, extent > 0.0);
    let across = vec2<f32>(-along.y, along.x);
    var corners = array<vec2<f32>, 6>(vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
                                      vec2<f32>(0.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0));
    let corner = corners[index % 6u];
    let base = select(start.xy - along * radius, end.xy + along * radius, corner.x > 0.5);
    let place = base + across * (radius * corner.y);
    let nearest = max(start.z, end.z) + radius;
    let position = vec4<f32>(place.x / half_size.x, place.y / half_size.y, 0.5 - nearest / depth, 1.0);
    return Corner(position, place, start, end, colours[segment]);
}

// How far along a ray, from `origin` along `heading`, it meets the ball of the radius about a centre: negative where
// it passes by.
fn ball(origin: vec3<f32>, heading: vec3<f32>, centre: vec3<f32>) -> f32 {
    let offset = origin - centre;
    let half_b = dot(heading, offset);
    let discriminant = half_b * half_b - dot(offset, offset) + radius * radius;
    return select(-1.0, -half_b - sqrt(max(discriminant, 0.0)), discriminant > 0.0);
}

// A sample of the box: where its ray first meets the capsule, lit there, or nothing where it passes by.
@fragment
fn shade(corner: Corner) -> Shade {
    // The ray through the sample, from in front of the capsule along the camera's -z.
    let origin = vec3<f32>(corner.place, max(corner.start.z, corner.end.z) + 2.0 * radius);
    let heading = vec3<f32>(0.0, 0.0, -1.0);
    let axis = corner.end - corner.start;
    let offset = origin - corner.start;
    let axis_squared = dot(axis, axis);
    let axis_heading = dot(axis, heading);
    let axis_offset = dot(axis, offset);
    // The cylinder about the segment: where the ray's distance from its line is the radius, between its ends.
    let a = axis_squared - axis_heading * axis_heading;
    let b = axis_squared * dot(heading, offset) - axis_offset * axis_heading;
    let c = axis_squared * dot(offset, offset) - axis_offset * axis_offset - radius * radius * axis_squared;
    let discriminant = b * b - a * c;
    let side = (-b - sqrt(max(discriminant, 0.0))) / max(a, 1e-30);
    let along = axis_offset + side * axis_heading;
    let on_side = discriminant > 0.0 && a > 1e-12 * axis_squared && along > 0.0 && along < axis_squared;
    // The nearest of the cylinder and the balls at the ends that the ray meets.
    var reach = select(1e30, side, on_side);
    let first = ball(origin, heading, corner.start);
    let second = ball(origin, heading, corner.end);
    reach = select(reach, min(reach, first), first > 0.0);
    reach = select(reach, min(reach, second), second > 0.0);
    if (reach >= 1e30) {
        discard;
    }
    let hit = origin + heading * reach;
    // The normal: from the nearest point of the segment out to the surface.
    let share = clamp(dot(hit - corner.start, axis) / max(axis_squared, 1e-30), 0.0, 1.0);
    let outward = normalize(hit - (corner.start + axis * share));
    // Lit as a mesh is, in the world: the normal as the plane square to it through the origin, carried out of the
    // camera's frame.
    let normal = camera >> (mv.x * outward.x + mv.y * outward.y + mv.z * outward.z);
    let direction = camera >> -mv.z.dual();
    let shaded = corner.colour.rgb * brightness(normal, direction) + gloss * glint(normal, direction);
    return Shade(vec4<f32>(shaded, 1.0), vec4<f32>(-hit.z, 0.0, 0.0, 1.0), 0.5 - hit.z / depth);
}
"""


# --- plumbing -------------------------------------------------------------------------
@cache
def shader() -> Shader:
    # The colour a byte a channel, and the depth alone at full precision, as meshes write them.
    return Shader(lamp_shading() + source, ga, dict(Point=Point, Plane=Plane, Motor=Motor), dict(axes=axes),
                  targets=("rgba8unorm", "r32float"))


def segments(starts: np.ndarray, ends: np.ndarray, colours: np.ndarray, width: float, view: View) -> Samples:
    """Straight segments from starts to ends `[segments, 3]`, coloured `[segments, 3]`, as capsules `width` pixels
    across, at every sample of the view's pixels, for `solids.layer` and `Scene` to compose."""
    shades = np.concatenate([np.broadcast_to(colours, starts.shape), np.ones((len(starts), 1))], axis=-1)
    return Samples(shader(), 6 * len(starts), dict(
        starts=mv.point(starts), ends=mv.point(ends), colours=shades, radius=width / 2 * view.pixel,
        camera=view.camera, half_size=view.half, depth=8 * view.half.max(), **lighting(view.camera, gloss)))


def polylines(curves: list[tuple[np.ndarray, np.ndarray]], width: float, view: View) -> Samples:
    """Polylines `[n, 3]`, each with its colour `[3]`, as strokes `width` pixels across: a capsule along each of
    their segments, the balls at the shared ends rounding the joints."""
    starts = np.concatenate([points[:-1] for points, _ in curves])
    ends = np.concatenate([points[1:] for points, _ in curves])
    colours = np.concatenate([np.repeat(np.asarray(colour)[None], len(points) - 1, axis=0) for points, colour in curves])
    return segments(starts, ends, colours, width, view)
