"""The solids ray traced on the GPU through glass, by a shader whose geometry is written in numga.

How far a point lies outside a solid is the largest of its quadrics' forms, `point & quadric(point)`,
negative inside all of them; a ray's points are its start carried by translators along its direction, and the ray
meets the solid where that form changes sign, found by stepping across the solid's bounding sphere and halving the
step that changed. Each pixel follows one path: an opaque
surface shows its colour, lit by the lamp and shadowed by what lies towards it, glass shadowing less; at
glass the path refracts, by Snell's law, and adds what the reflected ray glimpses, by Fresnel's share,
and inside glass the light fades by the glass's tint per unit length.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.quadrics.glass_cyclide import core
from examples.util.wgsl import Shader, supersampled

mv = core.mv
types = dict(Sphere=core.Sphere, OriginPlane=core.OriginPlane, Point=core.Point, Direction=core.Direction,
             Rotation=core.Rotation, Quadric=core.Quadric)
# The directions across and up the camera's own frame, which a place on the image weighs, `(axes * across).sum()`.
axes = stack((mv.x.dual(), mv.y.dual()))                                      # [2] Direction-
# The look: light where the lamp does not reach, the light glass lets into its shadow, the checker's darkening, and
# the sky from the horizon up; the sun seen in a reflection, its brightness, and how tightly it gathers about the
# lamp's direction; and the gamma the image is shown at.
ambient = 0.3
glass_shadow = 0.75
checker = 0.18
horizon_colour = np.array([0.86, 0.88, 0.92])
zenith_colour = np.array([0.36, 0.5, 0.72])
sun = 6.0
sun_size = 2000.0
gamma = 1.2

source = """
// A solid: the intersection of its quadrics, `quadrics[first]` up to `quadrics[end]`, each negative
// inside it, within a bounding sphere about `centre`. Its colour, the tint glass takes on per unit
// length; the share of light it lets through, and its refractive index; two planes through the origin
// whose pairings with a point, in unit steps, lay a checker over it.
struct Solid {
    first: u32,
    end: u32,
    centre: Point,
    radius: f32,
    colour: vec3<f32>,
    transparency: f32,
    refraction: f32,
    across: OriginPlane,
    along: OriginPlane,
}

var<storage, read> solids: array<Solid>;
var<storage, read> quadrics: array<Quadric>;
var<uniform> turn: Rotation;
var<uniform> eye: Point;
var<uniform> lamp: Direction;
var<uniform> half_view: f32;
alias Sample = vec3<f32>;

const far = 1e6;
// The step a ray takes off a surface before it looks again; the steps across a bounding sphere, and the
// halvings of the one that crosses.
const clearance = 1e-3;
const n_steps = 40u;
const n_halvings = 18u;
const n_bounces = 8u;

// The sphere about a centre: positive inside, paired with a point.
fn sphere(centre: Point, radius: f32) -> Sphere {
    return centre.dual_inverse() - radius * radius / 2 * horizon;
}

// How a point moves per unit distance as the translator along a unit direction carries it: its commutator with the
// translator's generator, the dual of the line joining the point at infinity with the direction.
fn velocity(start: Point, direction: Direction) -> Point {
    return start.commutator((infinity & direction).dual());
}

// The point a distance along a ray: its start carried by the translator along its unit direction,
// `translator(direction * distance) >> start`, written out for a start of unit weight: the start, plus the distance
// times its velocity, plus half the distance squared times the point at infinity. So written, the start's weight,
// which single precision holds only roughly for a far point, never meets the distance squared.
fn ahead(start: Point, direction: Direction, distance: f32) -> Point {
    return start + velocity(start, direction) * distance + infinity * (distance * distance / 2);
}

// How far a point lies outside a solid: the largest of its quadrics' forms, negative inside them all.
fn outside(solid: Solid, point: Point) -> f32 {
    var largest = -far;
    for (var index = solid.first; index < solid.end; index++) {
        largest = max(largest, point & quadrics[index](point));
    }
    return largest;
}

// The first distance along a ray at which it crosses a solid's surface, or `far`.
fn crossing(solid: Solid, origin: Point, direction: Direction) -> f32 {
    // The stretch of the ray inside the bounding sphere: how far along the ray the sphere's centre lies, the rate at
    // which the sphere's pairing with the start grows as the start moves along the ray, and the square of half the
    // chord, that squared plus twice the sphere's pairing with the start, which is the squared radius less the start's
    // squared distance from the centre.
    let bounding = sphere(solid.centre, solid.radius);
    let along = bounding & velocity(origin, direction);
    let half_chord_squared = along * along + 2 * (bounding & origin);
    let start = max(along - sqrt(max(half_chord_squared, 0.0)), clearance);
    let end = along + sqrt(max(half_chord_squared, 0.0));
    if (half_chord_squared <= 0 || end <= start) {
        return far;
    }
    let started_outside = outside(solid, ahead(origin, direction, start)) > 0;
    var before = start;
    for (var step = 1u; step <= n_steps; step++) {
        let after = start + (end - start) * f32(step) / f32(n_steps);
        if ((outside(solid, ahead(origin, direction, after)) > 0) != started_outside) {
            var low = before;
            var high = after;
            for (var halving = 0u; halving < n_halvings; halving++) {
                let middle = (low + high) / 2;
                if ((outside(solid, ahead(origin, direction, middle)) > 0) == started_outside) {
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

// A solid's outward normal at a point on it, as the unit plane through the origin parallel to the tangent plane, whose
// dual is the outward direction: from the tangent sphere of the quadric whose surface the point is on, the one the
// point lies least inside, that sphere flattened at the point, `tangent + (tangent | horizon) * point.dual_inverse()`,
// is the tangent plane, and its part within x y z the plane through the origin.
fn normal(solid: Solid, point: Point) -> OriginPlane {
    var surface = solid.first;
    for (var index = solid.first; index < solid.end; index++) {
        if ((point & quadrics[index](point)) > (point & quadrics[surface](point))) {
            surface = index;
        }
    }
    let tangent = quadrics[surface](point);
    return (tangent + (tangent | horizon) * point.dual_inverse()).cast(OriginPlane).normalized();
}

// The sky along a direction: by how far it looks up, its pairing with the floor's plane, and how squarely it looks
// into the sun, its pairing with the plane square to the lamp's direction.
fn sky(direction: Direction) -> vec3<f32> {
    let up = mv.z & direction;
    let sunward = lamp.dual_inverse() & direction;
    return mix(horizon_colour, zenith_colour, clamp(up, 0.0, 1.0)) + sun * pow(max(sunward, 0.0), sun_size);
}

// The light towards the lamp from a point: dimmed by each solid in the way, opaque ones fully.
fn sunlight(point: Point) -> f32 {
    var light = 1.0;
    for (var index = 0u; index < arrayLength(&solids); index++) {
        if (crossing(solids[index], point, lamp) < far) {
            light *= solids[index].transparency * glass_shadow;
        }
    }
    return light;
}

// An opaque surface's colour at a point: lit by the lamp, shadowed, and checkered.
fn surface(solid: Solid, point: Point, facing: OriginPlane) -> vec3<f32> {
    let lit = max(facing & lamp, 0.0) * sunlight(ahead(point, facing.dual(), clearance));
    let parity = abs((floor(solid.across & point) + floor(solid.along & point)) % 2.0);
    return solid.colour * (1 - checker * parity) * (ambient + (1 - ambient) * lit);
}

// What a ray glimpses without passing through glass: the surface it meets, or the sky.
fn glimpse(origin: Point, direction: Direction) -> vec3<f32> {
    let hit = nearest(origin, direction);
    if (hit.distance >= far) {
        return sky(direction);
    }
    let solid = solids[hit.solid];
    let point = ahead(origin, direction, hit.distance);
    let facing = normal(solid, point);
    return mix(surface(solid, point, facing), sky(direction), solid.transparency);
}

// The light along the ray through a place on the image: from the camera's own -z, turned by its rotor.
fn sample(place: vec2<f32>) -> Sample {
    let across = screen(place) * half_view;
    let toward = turn >> (-mv.z.dual() + (axes * across).sum());
    // A direction's length squared is its pairing with the plane square to it.
    var direction = toward / sqrt(toward.dual_inverse() & toward);
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
        position = ahead(position, direction, hit.distance);
        let outward = normal(solid, position);
        let entering = (outward & direction) < 0;
        // Light that crossed glass to get here faded by its tint, per unit length.
        throughput *= select(pow(solid.colour, vec3<f32>(hit.distance)), vec3<f32>(1.0), entering || solid.transparency == 0);
        let facing = select(-outward, outward, entering);
        colour += throughput * (1 - solid.transparency) * surface(solid, position, facing);
        // At glass the ray refracts, by Snell's law, and reflects in the plane along the surface; past the critical
        // angle all of it reflects.
        let cosine = -(facing & direction);
        let ratio = select(solid.refraction, 1 / solid.refraction, entering);
        let sine_squared = ratio * ratio * (1 - cosine * cosine);
        let reflected = facing >> direction;
        let refracted = direction * ratio + facing.dual() * (ratio * cosine - sqrt(max(1 - sine_squared, 0.0)));
        // Schlick's approximation to Fresnel's share of reflected light.
        let normal_reflectance = ((solid.refraction - 1) / (solid.refraction + 1)) * ((solid.refraction - 1) / (solid.refraction + 1));
        let reflectance = normal_reflectance + (1 - normal_reflectance) * pow(1 - cosine, 5.0);
        let total = sine_squared > 1;
        colour += throughput * solid.transparency * select(reflectance, 0.0, total) * glimpse(ahead(position, reflected, clearance), reflected);
        throughput *= solid.transparency * select(1 - reflectance, 1.0, total);
        direction = select(refracted, reflected, total);
        position = ahead(position, direction, clearance);
        if (max(throughput.x, max(throughput.y, throughput.z)) < 0.01) {
            break;
        }
    }
    return colour;
}
"""


# --- plumbing -------------------------------------------------------------------------
def shader() -> Shader:
    """The tracer: the point and the plane at infinity, the camera's frame and the look given as its constants."""
    return Shader(supersampled() + source, core.ga, types,
                  dict(infinity=core.infinity, horizon=core.horizon, axes=axes, ambient=ambient,
                       glass_shadow=glass_shadow, checker=checker, horizon_colour=horizon_colour,
                       zenith_colour=zenith_colour, sun=sun, sun_size=sun_size))


def frames(tracer: Shader, quadrics: core.Quadric, solids: dict, turns: core.Rotation, eyes: core.Point,
           lamp: core.Direction, half_view: float, shape: tuple[int, int]) -> Iterator[np.ndarray]:
    """The scene from each camera, its turn and its eye, `[rows, columns, 3]` per camera."""
    for turn, eye in zip(turns, eyes):
        traced = tracer(shape, quadrics=quadrics, solids=solids, turn=turn, eye=eye, lamp=lamp, half_view=half_view)["image"]
        yield np.round(np.clip(traced, 0, 1) ** (1 / gamma) * 255).astype(np.uint8)

