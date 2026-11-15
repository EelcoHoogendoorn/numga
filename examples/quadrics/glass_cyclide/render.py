"""A WGSL shader written in numga ray-traces conformal quadric solids with refraction, Fresnel reflection, and glass absorption.

A point lies outside an intersected solid whenever the largest of its quadric pairings `quadric(point) & point`
is positive. Each camera ray steps across a solid's bounding sphere to bracket the first sign change, refines
the crossing by bisection, reads the outward normal from the active quadric's tangent sphere, and traces
refracted and reflected paths through tinted glass.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.quadrics.glass_cyclide import core
from numga.wgsl import Shader, supersampled

mv = core.mv
types = dict(
    Sphere=core.Sphere,
    OriginPlane=core.OriginPlane,
    Point=core.Point,
    Direction=core.Direction,
    Rotation=core.Rotation,
    Quadric=core.Quadric,
)
# Horizontal and vertical directions of the reference camera frame, together with its forward sight line and floor plane.
screen_axes = stack((mv.x.dual(), mv.y.dual()))                               # [2] Direction-
view_axis = -mv.z.dual()                                                      # [] Direction-
zenith_plane = mv.z                                                           # [] OriginPlane+
gamma = 1.2

source = """
// Each solid is the intersection of quadrics[first .. end] inside a bounding sphere around centre,
// carrying optical material parameters and two origin planes that lay a unit checker pattern across it.
struct Solid {
    first: u32,
    end: u32,
    centre: Point,
    radius: f32,
    colour: vec3<f32>,
    transparency: f32,
    refraction: f32,
    tile_x: OriginPlane,
    tile_y: OriginPlane,
}

var<storage, read> solids: array<Solid>;
var<storage, read> quadrics: array<Quadric>;
var<uniform> turn: Rotation;
var<uniform> eye: Point;
var<uniform> lamp: Direction;
var<uniform> half_view: f32;
alias Sample = vec3<f32>;

const far = 1e6;
const clearance = 1e-3;
const n_steps = 40u;
const n_halvings = 18u;
const n_bounces = 8u;
const ambient = 0.3;
const glass_shadow = 0.75;
const checker_contrast = 0.18;
const horizon_colour = vec3<f32>(0.86, 0.88, 0.92);
const zenith_colour = vec3<f32>(0.36, 0.5, 0.72);
const sun_brightness = 6.0;
const sun_sharpness = 2000.0;

// Sphere around centre whose pairing with an interior point is positive.
fn sphere(centre: Point, radius: f32) -> Sphere {
    return centre.dual_inverse() - radius * radius / 2 * horizon;
}

// Rate of change of a conformal point per unit step along a unit direction, given by its commutator with the translator generator.
fn velocity(start: Point, direction: Direction) -> Point {
    return start.commutator((infinity & direction).dual());
}

// Point at a given distance along a ray from a unit-weight start, expanded so single-precision start weight never multiplies distance squared.
fn ray_point(start: Point, direction: Direction, distance: f32) -> Point {
    return start + velocity(start, direction) * distance + infinity * (distance * distance / 2);
}

// Largest quadric form of the solid at point, negative strictly inside all of the solid's quadrics.
fn signed_outside(solid: Solid, point: Point) -> f32 {
    var largest = -far;
    for (var index = solid.first; index < solid.end; index++) {
        largest = max(largest, quadrics[index](point) & point);
    }
    return largest;
}

// First positive ray distance where the solid's surface is crossed, or far when the ray misses.
fn crossing(solid: Solid, origin: Point, direction: Direction) -> f32 {
    // Pairing the bounding sphere with the ray start and its velocity gives the chord midpoint and half-chord length squared.
    let bounding = sphere(solid.centre, solid.radius);
    let axial_distance = bounding & velocity(origin, direction);
    let half_chord_squared = axial_distance * axial_distance + 2 * (bounding & origin);
    let start = max(axial_distance - sqrt(max(half_chord_squared, 0.0)), clearance);
    let end = axial_distance + sqrt(max(half_chord_squared, 0.0));
    if (half_chord_squared <= 0 || end <= start) {
        return far;
    }
    let started_outside = signed_outside(solid, ray_point(origin, direction, start)) > 0;
    var before = start;
    for (var step = 1u; step <= n_steps; step++) {
        let after = start + (end - start) * f32(step) / f32(n_steps);
        if ((signed_outside(solid, ray_point(origin, direction, after)) > 0) != started_outside) {
            var low = before;
            var high = after;
            for (var halving = 0u; halving < n_halvings; halving++) {
                let middle = (low + high) / 2;
                if ((signed_outside(solid, ray_point(origin, direction, middle)) > 0) == started_outside) {
                    low = middle;
                } else {
                    high = middle;
                }
            }
            return high;
        }
        before = after;
    }
    return far;
}

struct Hit {
    distance: f32,
    solid: u32,
}

fn nearest(origin: Point, direction: Direction) -> Hit {
    var hit = Hit(far, 0u);
    for (var index = 0u; index < arrayLength(&solids); index++) {
        let distance = crossing(solids[index], origin, direction);
        if (distance < hit.distance) {
            hit = Hit(distance, index);
        }
    }
    return hit;
}

// Unit origin plane parallel to the solid's tangent plane at point, flattened from the active quadric's tangent sphere.
fn surface_plane(solid: Solid, point: Point) -> OriginPlane {
    var active = solid.first;
    for (var index = solid.first; index < solid.end; index++) {
        if ((quadrics[index](point) & point) > (quadrics[active](point) & point)) {
            active = index;
        }
    }
    let tangent = quadrics[active](point);
    return (tangent + (tangent | horizon) * point.dual_inverse()).cast(OriginPlane).normalized();
}

// Sky radiance from vertical elevation above the floor plane and alignment with the solar lamp direction.
fn sky(direction: Direction) -> vec3<f32> {
    let elevation = zenith_plane & direction;
    let sun_alignment = lamp.dual_inverse() & direction;
    return mix(horizon_colour, zenith_colour, clamp(elevation, 0.0, 1.0)) + sun_brightness * pow(max(sun_alignment, 0.0), sun_sharpness);
}

// Direct lamp transmittance reaching point, attenuated partially by glass solids and blocked completely by opaque ones.
fn sunlight(point: Point) -> f32 {
    var light = 1.0;
    for (var index = 0u; index < arrayLength(&solids); index++) {
        if (crossing(solids[index], point, lamp) < far) {
            light *= solids[index].transparency * glass_shadow;
        }
    }
    return light;
}

// Opaque diffuse shading with lamp shadow and planar checker modulation.
fn matte_colour(solid: Solid, point: Point, facing: OriginPlane) -> vec3<f32> {
    let lit = max(facing & lamp, 0.0) * sunlight(ray_point(point, facing.dual(), clearance));
    let parity = abs((floor(solid.tile_x & point) + floor(solid.tile_y & point)) % 2.0);
    return solid.colour * (1 - checker_contrast * parity) * (ambient + (1 - ambient) * lit);
}

// Single-bounce reflection lookup returning either the sky or an unrefracted surface glimpse.
fn glimpse(origin: Point, direction: Direction) -> vec3<f32> {
    let hit = nearest(origin, direction);
    if (hit.distance >= far) {
        return sky(direction);
    }
    let solid = solids[hit.solid];
    let point = ray_point(origin, direction, hit.distance);
    let facing = surface_plane(solid, point);
    return mix(matte_colour(solid, point, facing), sky(direction), solid.transparency);
}

// Radiance along the camera ray through a normalized image sample location.
fn sample(place: vec2<f32>) -> Sample {
    let screen_offset = screen(place) * half_view;
    let ray_vector = turn >> (view_axis + (screen_axes * screen_offset).sum());
    var direction = ray_vector / ray_vector.antinorm();
    var position = eye;
    var colour = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    for (var bounce = 0u; bounce < n_bounces; bounce++) {
        let hit = nearest(position, direction);
        if (hit.distance >= far) {
            colour += throughput * sky(direction);
            break;
        }
        let solid = solids[hit.solid];
        position = ray_point(position, direction, hit.distance);
        let incident_normal = surface_plane(solid, position);
        let entering = (incident_normal & direction) < 0;
        // Rays exiting tinted glass attenuate exponentially per unit path length inside the solid.
        throughput *= select(pow(solid.colour, vec3<f32>(hit.distance)), vec3<f32>(1.0), entering || solid.transparency == 0);
        let facing = select(-incident_normal, incident_normal, entering);
        colour += throughput * (1 - solid.transparency) * matte_colour(solid, position, facing);
        // Snell refraction and plane reflection across the interface, falling back to total internal reflection past the critical angle.
        let cosine = -(facing & direction);
        let ratio = select(solid.refraction, 1 / solid.refraction, entering);
        let sine_squared = ratio * ratio * (1 - cosine * cosine);
        let reflected_ray = facing >> direction;
        let refracted_ray = direction * ratio + facing.dual() * (ratio * cosine - sqrt(max(1 - sine_squared, 0.0)));
        // Schlick approximation to the Fresnel reflectance at dielectric interfaces.
        let normal_reflectance = ((solid.refraction - 1) / (solid.refraction + 1)) * ((solid.refraction - 1) / (solid.refraction + 1));
        let reflectance = normal_reflectance + (1 - normal_reflectance) * pow(1 - cosine, 5.0);
        let total_reflection = sine_squared > 1;
        colour += throughput * solid.transparency * select(reflectance, 0.0, total_reflection) * glimpse(ray_point(position, reflected_ray, clearance), reflected_ray);
        throughput *= solid.transparency * select(1 - reflectance, 1.0, total_reflection);
        direction = select(refracted_ray, reflected_ray, total_reflection);
        position = ray_point(position, direction, clearance);
        if (max(throughput.x, max(throughput.y, throughput.z)) < 0.01) {
            break;
        }
    }
    return colour;
}
"""


# --- plumbing -------------------------------------------------------------------------
def shader() -> Shader:
    """The supersampled WGSL ray tracer binds conformal infinity, the horizon plane, and the reference camera axes as constants."""
    return Shader(
        supersampled() + source,
        core.ga,
        types,
        dict(
            infinity=core.infinity,
            horizon=core.horizon,
            screen_axes=screen_axes,
            view_axis=view_axis,
            zenith_plane=zenith_plane,
        ),
    )


def camera_frames(
    tracer: Shader,
    quadrics: core.Quadric,
    solids: dict,
    turns: core.Rotation,
    eyes: core.Point,
    lamp: core.Direction,
    half_view: float,
    shape: tuple[int, int],
) -> Iterator[np.ndarray]:
    """Yields gamma-encoded RGB frames (`[n_rows, n_columns, 3]`) traced from each camera pose."""
    for turn, eye in zip(turns, eyes):
        raw_image = tracer(
            shape, quadrics=quadrics, solids=solids, turn=turn, eye=eye, lamp=lamp, half_view=half_view,
        )["image"]
        yield np.round(np.clip(raw_image, 0, 1) ** (1 / gamma) * 255).astype(np.uint8)
