"""WebGPU ray tracing of quadric ellipsoids on the 3-sphere.

Each pixel's ray from eye `e` along equatorial direction `d` traces a great circle parameterized by
cotangent depth `u`, where a primal quadric evaluates to a scalar quadratic `A * u**2 + 2 * B * u + C`
with `A = e & Q(e)`, `B = e & Q(d)`, and `C = d & Q(d)`. Selecting the largest real root across the
scene finds the smallest positive arc angle, reconstructs the hit `e * u + d`, and shades its polar
plane against a spherical point light.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import cache

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.quadrics.s3_raytracer.core import Motor, Plane, Point, Quadric, ga, mv, origin
from numga.wgsl import Shader, supersampled
from examples.util.panels import image_figure

background = 0.02

source = """
// Quadrics and colours of the bodies, the motor placing the eye frame, the screen half-height, and the light point.
var<storage, read> quadrics: array<Quadric>;
var<storage, read> colours: array<vec3<f32>>;
var<uniform> eye_frame: Motor;
var<uniform> half_height: f32;
var<uniform> light: Point;

alias Sample = vec3<f32>;

const ambient = 0.15;
// Specular highlight weight and exponent.
const gloss = 0.35;
const shininess = 40.0;
// Arc offset from the hit before testing shadow occlusion toward the light.
const clearance = 1e-4;

// Unit tangent point orthogonal to `place` pointing toward `target` along the 3-sphere.
fn tangent_toward(place: Point, target: Point) -> Point {
    return (target - place * place.dot(target)).normalized();
}

// Tests whether the short great-circle arc from `hit` to `light` enters any quadric interior.
fn in_shadow(hit: Point) -> bool {
    let start = hit + (light - hit) * clearance;
    for (var index = 0u; index < arrayLength(&quadrics); index++) {
        let polar = quadrics[index](light);
        let at_light = light & polar;
        let at_start = start & quadrics[index](start);
        let across = start & polar;
        if (at_start < 0 || at_light < 0 || (across < 0 && across * across > at_start * at_light)) {
            return true;
        }
    }
    return false;
}

fn sample(place: vec2<f32>) -> Sample {
    let across = screen(place) * half_height;
    // Eye point and the ray direction a quarter turn ahead, offset across the image plane.
    let eye = eye_frame >> origin;
    let ray_direction = eye_frame >> (frame * vec3<f32>(1.0, across.x, across.y)).sum();
    var depth = -1e30;
    var body = 0u;
    for (var index = 0u; index < arrayLength(&quadrics); index++) {
        let quadric = quadrics[index];
        let a = eye & quadric(eye);
        let b = eye & quadric(ray_direction);
        let c = ray_direction & quadric(ray_direction);
        let discriminant = b * b - a * c;
        let nearest = (-b + sqrt(max(discriminant, 0.0))) / a;
        if (discriminant >= 0 && nearest > depth) {
            depth = nearest;
            body = index;
        }
    }
    if (depth <= -1e30) {
        return vec3<f32>(background);
    }
    let hit = (eye * depth + ray_direction).normalized();
    // Pairing the hit's polar plane with `light` determines front-facing illumination with spherical sin^-2 falloff.
    let polar = quadrics[body](hit);
    let arc = acos(clamp(hit.dot(light) / light.dot(light), -1.0, 1.0));
    let cosine = (polar & light) / (polar.norm() * sin(arc));
    let lit = cosine < 0 && !in_shadow(hit);
    // Soft hyperbolic rolloff preserves highlight detail on faces close to the point source.
    let lambert = select(0.0, tanh(-cosine / (sin(arc) * sin(arc))), lit);
    // Blinn-Phong glint where the outward spherical normal bisects the tangent directions to the eye and light.
    let to_eye = tangent_toward(hit, eye);
    let normal = tangent_toward(hit, polar.dual());
    let facing = normal * select(-1.0, 1.0, normal.dot(to_eye) >= 0);
    let halfway = tangent_toward(hit, to_eye + tangent_toward(hit, light));
    let glint = select(0.0, pow(max(facing.dot(halfway), 0.0), shininess), lit);
    return colours[body] * (ambient + (1 - ambient) * lambert) + gloss * glint;
}
"""


# --- plumbing -------------------------------------------------------------------------
@cache
def shader() -> Shader:
    """Compiles and caches the supersampled WebGPU ray-tracing shader on the 3-sphere."""
    # Equatorial points a quarter turn from `origin` along x, y, and z define forward, right, and up in the origin frame.
    frame = stack((mv.x, mv.y, mv.z)).dual()                                   # [3] Point-
    return Shader(
        supersampled() + source,
        ga,
        dict(Point=Point, Plane=Plane, Quadric=Quadric, Motor=Motor),
        dict(origin=origin, frame=frame, background=background),
    )


def picture(
    eye_frame: Motor,
    surfaces: Quadric,
    colors: np.ndarray,
    light: Point,
    fov: float,
    shape: tuple[int, int],
) -> np.ndarray:
    """Renders one `[n_rows, n_columns, 3]` `uint8` RGB image of `surfaces` from `eye_frame`."""
    n_rows, n_columns = shape
    traced = shader()(
        shape,
        quadrics=surfaces,
        colours=colors,
        eye_frame=eye_frame,
        half_height=np.tan(fov / 2) * n_rows / n_columns,
        light=light,
    )["image"]
    return np.round(np.clip(traced, 0, 1) * 255).astype(np.uint8)


def walk_animation(
    surfaces: Quadric,
    colors: np.ndarray,
    light: Point,
    fov: float,
    shape: tuple[int, int],
    eye_frames: Iterable[Motor],
) -> Iterator[np.ndarray]:
    """Yields one `[n_rows, n_columns, 3]` `uint8` RGB frame for each motor in `eye_frames`."""
    for eye_frame in eye_frames:
        yield picture(eye_frame, surfaces, colors, light, fov, shape)


def walk_figure(
    eye_frame: Motor,
    surfaces: Quadric,
    colors: np.ndarray,
    light: Point,
    fov: float,
    shape: tuple[int, int],
) -> plt.Figure:
    """Wraps the rendered view from `eye_frame` in a borderless image figure."""
    return image_figure(picture(eye_frame, surfaces, colors, light, fov, shape))
