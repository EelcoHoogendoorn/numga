"""The solids ray traced on the GPU through glass, by a shader whose geometry is written in numga.

How far a point lies outside a solid is the largest of its quadrics' forms, `point & quadric(point)`,
negative inside all of them; a ray meets the solid where that changes sign, found by stepping across
the solid's bounding sphere and halving the step that changed. Each pixel follows one path: an opaque
surface shows its colour, lit by the lamp and shadowed by what lies towards it, glass shadowing less; at
glass the path refracts, by Snell's law, and adds what the reflected ray glimpses, by Fresnel's share,
and inside glass the light fades by the glass's tint per unit length.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.quadrics.glass_cyclide import core
from examples.util.wgsl import Shader, supersampled

types = dict(Sphere=core.Sphere, Euclidean=core.Euclidean, Point=core.Point, Rotation=core.Rotation, Quadric=core.Quadric)

source = """
// A solid: the intersection of its quadrics, `quadrics[first]` up to `quadrics[end]`, each negative
// inside it, within a bounding sphere about `centre`. Its colour, the tint glass takes on per unit
// length; the share of light it lets through, and its refractive index; two directions whose unit steps
// lay a checker over it.
struct Solid {
    first: u32,
    end: u32,
    centre: Euclidean,
    radius: f32,
    colour: vec3<f32>,
    transparency: f32,
    refraction: f32,
    across: Euclidean,
    along: Euclidean,
}

var<storage, read> solids: array<Solid>;
var<storage, read> quadrics: array<Quadric>;
var<uniform> turn: Rotation;
var<uniform> eye: Euclidean;
var<uniform> lamp: Euclidean;
var<uniform> half_view: f32;
alias Sample = vec3<f32>;

const origin = (mv.w + mv.e) / 2;
const infinity = mv.e - mv.w;
const far = 1e6;
// The step a ray takes off a surface before it looks again; the steps across a bounding sphere, and the
// halvings of the one that crosses.
const clearance = 1e-3;
const n_steps = 40u;
const n_halvings = 18u;
const n_bounces = 8u;
// The look: light where the lamp does not reach, the light glass lets into its shadow, the checker's
// darkening, and the sky from the horizon up.
const ambient = 0.3;
const glass_shadow = 0.75;
const checker = 0.18;
const horizon = vec3<f32>(0.86, 0.88, 0.92);
const zenith = vec3<f32>(0.36, 0.5, 0.72);
// The sun seen in a reflection: its brightness, and how tightly it gathers about the lamp's direction.
const sun = 6.0;
const sun_size = 2000.0;

fn up(position: Euclidean) -> Point {
    return (origin + position + (position | position) / 2 * infinity).dual();
}

// How far a position lies outside a solid: the largest of its quadrics' forms, negative inside them all.
fn outside(solid: Solid, position: Euclidean) -> f32 {
    let point = up(position);
    var largest = -far;
    for (var index = solid.first; index < solid.end; index++) {
        largest = max(largest, point & quadrics[index](point));
    }
    return largest;
}

// The first distance along a ray at which it crosses a solid's surface, or `far`.
fn crossing(solid: Solid, origin: Euclidean, direction: Euclidean) -> f32 {
    // The stretch of the ray inside the bounding sphere.
    let along = (solid.centre - origin) | direction;
    let aside = solid.centre - origin - direction * along;
    let half_chord_squared = solid.radius * solid.radius - (aside | aside);
    let start = max(along - sqrt(max(half_chord_squared, 0.0)), clearance);
    let end = along + sqrt(max(half_chord_squared, 0.0));
    if (half_chord_squared <= 0 || end <= start) {
        return far;
    }
    let started_outside = outside(solid, origin + direction * start) > 0;
    var before = start;
    for (var step = 1u; step <= n_steps; step++) {
        let after = start + (end - start) * f32(step) / f32(n_steps);
        if ((outside(solid, origin + direction * after) > 0) != started_outside) {
            var low = before;
            var high = after;
            for (var halving = 0u; halving < n_halvings; halving++) {
                let middle = (low + high) / 2;
                if ((outside(solid, origin + direction * middle) > 0) == started_outside) {
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

fn nearest(origin: Euclidean, direction: Euclidean) -> Hit {
    var hit = Hit(far, 0u);
    for (var index = 0u; index < arrayLength(&solids); index++) {
        let distance = crossing(solids[index], origin, direction);
        if (distance < hit.distance) {
            hit = Hit(distance, index);
        }
    }
    return hit;
}

// The outward unit direction square to a solid at a position on it, from the tangent sphere of the
// quadric whose surface the position is on: the one the position lies least inside.
fn normal(solid: Solid, position: Euclidean) -> Euclidean {
    let point = up(position);
    var surface = solid.first;
    for (var index = solid.first; index < solid.end; index++) {
        if ((point & quadrics[index](point)) > (point & quadrics[surface](point))) {
            surface = index;
        }
    }
    let tangent = quadrics[surface](point);
    let direction = tangent.cast(Euclidean) + (tangent | infinity) * position;
    return direction / (direction | direction).square_root();
}

fn sky(direction: Euclidean) -> vec3<f32> {
    return mix(horizon, zenith, clamp(mv.z | direction, 0.0, 1.0)) + sun * pow(max(lamp | direction, 0.0), sun_size);
}

// The light towards the lamp from a position: dimmed by each solid in the way, opaque ones fully.
fn sunlight(position: Euclidean) -> f32 {
    var light = 1.0;
    for (var index = 0u; index < arrayLength(&solids); index++) {
        if (crossing(solids[index], position, lamp) < far) {
            light *= solids[index].transparency * glass_shadow;
        }
    }
    return light;
}

// An opaque surface's colour at a position: lit by the lamp, shadowed, and checkered.
fn surface(solid: Solid, position: Euclidean, facing: Euclidean) -> vec3<f32> {
    let lit = max(facing | lamp, 0.0) * sunlight(position + facing * clearance);
    let parity = abs((floor(solid.across | position) + floor(solid.along | position)) % 2.0);
    return solid.colour * (1 - checker * parity) * (ambient + (1 - ambient) * lit);
}

// What a ray glimpses without passing through glass: the surface it meets, or the sky.
fn glimpse(origin: Euclidean, direction: Euclidean) -> vec3<f32> {
    let hit = nearest(origin, direction);
    if (hit.distance >= far) {
        return sky(direction);
    }
    let solid = solids[hit.solid];
    let position = origin + direction * hit.distance;
    let facing = normal(solid, position);
    return mix(surface(solid, position, facing), sky(direction), solid.transparency);
}

fn unit(direction: Euclidean) -> Euclidean {
    return direction / (direction | direction).square_root();
}

// The light along the ray through a place on the image: from the camera's own -z, turned by its rotor.
fn sample(place: vec2<f32>) -> Sample {
    let across = screen(place) * half_view;
    var direction = unit(turn >> (-mv.z + mv.x * across.x + mv.y * across.y));
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
        position = position + direction * hit.distance;
        let outward = normal(solid, position);
        let entering = (outward | direction) < 0;
        // Light that crossed glass to get here faded by its tint, per unit length.
        throughput *= select(pow(solid.colour, vec3<f32>(hit.distance)), vec3<f32>(1.0), entering || solid.transparency == 0);
        let facing = select(-outward, outward, entering);
        colour += throughput * (1 - solid.transparency) * surface(solid, position, facing);
        // At glass the ray refracts, by Snell's law; past the critical angle all of it reflects.
        let cosine = -(facing | direction);
        let ratio = select(solid.refraction, 1 / solid.refraction, entering);
        let sine_squared = ratio * ratio * (1 - cosine * cosine);
        let reflected = direction + facing * (2 * cosine);
        let refracted = direction * ratio + facing * (ratio * cosine - sqrt(max(1 - sine_squared, 0.0)));
        // Schlick's approximation to Fresnel's share of reflected light.
        let normal_reflectance = ((solid.refraction - 1) / (solid.refraction + 1)) * ((solid.refraction - 1) / (solid.refraction + 1));
        let reflectance = normal_reflectance + (1 - normal_reflectance) * pow(1 - cosine, 5.0);
        let total = sine_squared > 1;
        colour += throughput * solid.transparency * select(reflectance, 0.0, total) * glimpse(position + reflected * clearance, reflected);
        throughput *= solid.transparency * select(1 - reflectance, 1.0, total);
        direction = select(refracted, reflected, total);
        position = position + direction * clearance;
        if (max(throughput.x, max(throughput.y, throughput.z)) < 0.01) {
            break;
        }
    }
    return colour;
}
"""


# --- plumbing -------------------------------------------------------------------------
def shader() -> Shader:
    return Shader(supersampled() + source, core.ga, types, {})


def frames(tracer: Shader, quadrics: core.Quadric, solids: dict, turns: core.Rotation, eyes: core.Euclidean,
           lamp: core.Euclidean, half_view: float, shape: tuple[int, int]) -> Iterator[np.ndarray]:
    """The scene from each camera, its turn and its eye, `[rows, columns, 3]` per camera."""
    for turn, eye in zip(turns, eyes):
        traced = tracer(shape, quadrics=quadrics, solids=solids, turn=turn, eye=eye, lamp=lamp, half_view=half_view)["image"]
        yield np.round(np.clip(traced, 0, 1) ** (1 / 1.2) * 255).astype(np.uint8)

