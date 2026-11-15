"""Supersampled GPU ray tracing of composite quadric solids in projective 3D space.

Each pixel invocation traces supersampled camera rays to the nearest boundary crossing of any solid
that lies inside all sibling quadrics of that solid. Surface hits are shaded by key-lamp cosine, a
hard shadow ray, planar checker steps, specular glint, and one mirror bounce reflected across the
tangent plane.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np

from numga import stack
from numga.render.solids import quadric_regions
from numga.wgsl import Shader, supersampled
from examples.quadrics.csg_raytracer import core

types = dict(Point=core.Point, Plane=core.Plane, Motor=core.Motor, Quadric=core.Quadric)

source = '''
// Each solid intersects `quadrics[first]` up to `quadrics[end]`, carries its RGB colour and mirror
// reflectivity, and tiles a checker pattern from unit steps of `first_stripe` and `second_stripe`.
struct Solid {
    first: u32,
    end: u32,
    colour: vec3<f32>,
    reflectivity: f32,
    first_stripe: Plane,
    second_stripe: Plane,
}

var<storage, read> solids: array<Solid>;
var<uniform> camera: Motor;
var<uniform> eye: Point;
var<uniform> lamp: Point;
var<uniform> half_view: f32;
alias Sample = vec3<f32>;

// Sentinel distance beyond the scene and ray offset off a hit surface before secondary rays start.
const scene_far = 1e6;
const clearance = 1e-3;
// Inside-test slack for boundary points shared by two intersecting quadric surfaces.
const tolerance = 1e-4;
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

// Nearest boundary hit across all solids, lying on one quadric and inside all sibling quadrics.
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

// Direct illumination, shadow test, planar checker, and specular glint at a ray's first hit, or sky.
fn sight(origin: Point, direction: Point) -> Sight {
    let hit = first_hit(origin, direction);
    if (hit.distance >= scene_far) {
        return Sight(mix(horizon, zenith, clamp(mv.z & direction, 0.0, 1.0)), origin, mv.z, 0.0);
    }
    let solid = solids[hit.solid];
    let point = origin + direction * hit.distance;
    let normal = hit.tangent.normalized();
    // Pairing the unit tangent plane with the unit lamp direction yields the incidence cosine.
    let lit = max(normal & lamp, 0.0) * select(1.0, shadow, first_hit(point + lamp * clearance, lamp).distance < scene_far);
    let parity = abs((floor(solid.first_stripe & point) + floor(solid.second_stripe & point)) % 2.0);
    // Reflect the view direction in the tangent plane and pair its dual plane against the lamp direction.
    let glint = pow(max(-((normal >> direction).dual() & lamp), 0.0), shininess);
    let colour = solid.colour * (1 - checker * parity) * (ambient + (1 - ambient) * lit) + gloss * glint * lit;
    return Sight(colour, point, normal, solid.reflectivity);
}

// Trace one supersampled screen ray along the camera's local -z axis and blend one mirror reflection.
fn sample(place: vec2<f32>) -> Sample {
    let offset = screen(place) * half_view;
    let toward = camera >> (frame * vec3<f32>(offset, -1.0)).sum();
    let direction = toward / toward.dual().norm();
    let seen = sight(eye, direction);
    // Sandwiching the ray direction by the unit tangent plane reflects the view ray across the surface.
    let reflected = seen.normal >> direction;
    let mirrored = sight(seen.point + reflected * clearance, reflected);
    return mix(seen.colour, mirrored.colour, seen.reflectivity);
}
'''


# --- plumbing -------------------------------------------------------------------------
def shader() -> Shader:
    """Compiles the supersampled quadric-intersection WGSL ray-tracing shader."""
    return Shader(supersampled() + quadric_regions() + source, core.ga, types,
                  dict(frame=stack((core.mv.x, core.mv.y, core.mv.z)).dual()))


def frames(tracer: Shader, quadrics: core.Quadric, solids: dict, cameras: Iterable[core.Motor], lamp: core.Point,
           half_view: float, shape: tuple[int, int]) -> Iterator[np.ndarray]:
    """Yields `[rows, columns, 3]` RGB `uint8` images traced from each pose motor in `cameras`."""
    for camera in cameras:
        traced = tracer(shape, quadrics=quadrics, solids=solids, camera=camera, eye=camera >> core.origin,
                        lamp=lamp, half_view=half_view)["image"]
        yield np.round(np.clip(traced, 0, 1) * 255).astype(np.uint8)
