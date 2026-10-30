"""The solids ray traced on the GPU, by a shader whose geometry is written in numga.

One invocation per pixel follows a few camera rays through it to the first point on any solid's surface: on one of
its quadrics and inside all its others. That point is lit by the lamp through a shadow ray, laid with
its solid's checker and glint, and mixed with what its reflection sees, by its solid's reflectivity.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from numga import stack
from examples.quadrics.csg_raytracer import core
from examples.util.wgsl import Shader, supersampled
from examples.util.wgsl.solids import quadric_regions

types = dict(Point=core.Point, Plane=core.Plane, Motor=core.Motor, Quadric=core.Quadric)

source = """
// A solid: the intersection of its quadrics, `quadrics[first]` up to `quadrics[end]`, each negative
// inside it; its colour, the share of light it reflects as a mirror, and two planes whose unit steps
// lay a checker over it.
struct Solid {
    first: u32,
    end: u32,
    colour: vec3<f32>,
    reflectivity: f32,
    across: Plane,
    along: Plane,
}

var<storage, read> solids: array<Solid>;
var<uniform> camera: Motor;
var<uniform> eye: Point;
var<uniform> lamp: Point;
var<uniform> half_view: f32;
alias Sample = vec3<f32>;

// Distances beyond the scene, and the step a ray takes off a surface before it looks again.
const scene_far = 1e6;
const clearance = 1e-3;
// How scene_far a point may lie outside a quadric and still count as inside, for points on two surfaces.
const tolerance = 1e-4;
// The look: light where the lamp does not reach, a shadow's light, the checker's darkening, and the glint.
const ambient = 0.28;
const shadow = 0.3;
const checker = 0.15;
const shininess = 40.0;
const gloss = 0.3;
const horizon = vec3<f32>(0.80, 0.84, 0.90);
const zenith = vec3<f32>(0.38, 0.52, 0.72);

struct Hit {
    distance: f32,
    tangent: Plane,
    solid: u32,
}

// Each solid supplies its own look; its surface comes from the shared intersection of quadrics.
fn first_hit(origin: Point, direction: Point) -> Hit {
    var hit = Hit(scene_far, mv.z, 0u);
    for (var index = 0u; index < arrayLength(&solids); index++) {
        let solid = solids[index];
        let surface = region_hit(solid.first, solid.end, origin, direction, clearance, hit.distance, tolerance);
        if (surface.distance < hit.distance) {
            hit = Hit(surface.distance, surface.tangent, index);
        }
    }
    return hit;
}

struct Sight {
    colour: vec3<f32>,
    point: Point,
    normal: Plane,
    reflectivity: f32,
}

// What a ray sees: the solid it meets, lit by the lamp, shadowed, checkered and glinting; or the sky.
fn sight(origin: Point, direction: Point) -> Sight {
    let hit = first_hit(origin, direction);
    if (hit.distance >= scene_far) {
        return Sight(mix(horizon, zenith, clamp(mv.z & direction, 0.0, 1.0)), origin, mv.z, 0.0);
    }
    let solid = solids[hit.solid];
    let point = origin + direction * hit.distance;
    let normal = hit.tangent.normalized();
    // The lamp's light falls by the cosine of its direction off the normal, a plane paired with a direction.
    let lit = max(normal & lamp, 0.0) * select(1.0, shadow, first_hit(point + lamp * clearance, lamp).distance < scene_far);
    let parity = abs((floor(solid.across & point) + floor(solid.along & point)) % 2.0);
    // The view ray reflected in the tangent plane, against the lamp: `-d.dual()` is the plane square to d.
    let glint = pow(max(-((normal >> direction).dual() & lamp), 0.0), shininess);
    let colour = solid.colour * (1 - checker * parity) * (ambient + (1 - ambient) * lit) + gloss * glint * lit;
    return Sight(colour, point, normal, solid.reflectivity);
}

// What the ray through a place on the image sees: along the camera's own -z, turned by its motor.
fn sample(place: vec2<f32>) -> Sample {
    let across = screen(place) * half_view;
    let toward = camera >> (frame * vec3<f32>(across, -1.0)).sum();
    let direction = toward / toward.dual().norm();
    let seen = sight(eye, direction);
    // A mirror shows what its reflection sees: the view ray reflected in the tangent plane.
    let reflected = seen.normal >> direction;
    let mirrored = sight(seen.point + reflected * clearance, reflected);
    return mix(seen.colour, mirrored.colour, seen.reflectivity);
}
"""


# --- plumbing -------------------------------------------------------------------------
def shader() -> Shader:
    return Shader(supersampled() + quadric_regions() + source, core.ga, types,
                  dict(frame=stack((core.mv.x, core.mv.y, core.mv.z)).dual()))


def frames(tracer: Shader, quadrics: core.Quadric, solids: dict, cameras: core.Motor, lamp: core.Point,
           half_view: float, shape: tuple[int, int]) -> Iterator[np.ndarray]:
    """The scene as each camera sees it, `[rows, columns, 3]` per camera."""
    for camera in cameras:
        traced = tracer(shape, quadrics=quadrics, solids=solids, camera=camera, eye=camera >> core.origin,
                        lamp=lamp, half_view=half_view)["image"]
        yield np.round(np.clip(traced, 0, 1) * 255).astype(np.uint8)

