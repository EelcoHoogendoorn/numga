"""Ray tracing on the 3-sphere, on the GPU: each pixel's great circle meets the bodies' quadrics, then shading.

The ray from the eye `e` along the direction `d` of a pixel is the great circle `e * cos(t) + d * sin(t)`. A body's
quadric bound to it is `A * u**2 + 2 * B * u + C` in `u = cot(t)`, with `A = e & Q(e)`, `B = e & Q(d)` and
`C = d & Q(d)`, so the nearest hit, the smallest angle, is the largest root: the depth the screen conics give,
taken here straight from the quadric. The point it reaches is `e * u + d`, kept with its own sign.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import cache

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.quadrics.s3_raytracer.core import Motor, Plane, Point, Quadric, ga, mv, origin
from examples.util.wgsl import Shader, image_figure, supersampled

background = 0.02

source = """
// The bodies' quadrics and colours; the motor carrying the eye's frame out from the origin's, the eye looking along
// its own x, right along y and up along z; how far up the image's top edge looks, and the light.
var<storage, read> quadrics: array<Quadric>;
var<storage, read> colours: array<vec3<f32>>;
var<uniform> eye_frame: Motor;
var<uniform> half_height: f32;
var<uniform> light: Point;

alias Sample = vec3<f32>;

const ambient = 0.15;
// The glint's share and sharpness.
const gloss = 0.35;
const shininess = 40.0;
// How far along the arc to the light a shadow test starts.
const clearance = 1e-4;

// A point's direction along the sphere at a place on it, unit: its part square to the place.
fn along(place: Point, toward: Point) -> Point {
    return (toward - place * place.dot(toward)).normalized();
}

// Whether the short arc from a hit to the light enters any body, without roots: it does when either end lies
// inside, or the pairing with the light's polar plane falls below zero between them.
fn shadowed(hit: Point) -> bool {
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
    // The eye, and the direction of the pixel's ray from it: a point a quarter turn away, straight ahead, and as far
    // right and up as the pixel lies across the image.
    let eye = eye_frame >> origin;
    let direction = eye_frame >> (frame * vec3<f32>(1.0, across.x, across.y)).sum();
    var depth = -1e30;
    var body = 0u;
    for (var index = 0u; index < arrayLength(&quadrics); index++) {
        let quadric = quadrics[index];
        let a = eye & quadric(eye);
        let b = eye & quadric(direction);
        let c = direction & quadric(direction);
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
    let hit = (eye * depth + direction).normalized();
    // The great circle out of the light reaches the hit along an arc; the surface is lit where that arc
    // arrives from outside, which the pairing of its polar plane with the light decides, with the falloff
    // of a point source on the sphere.
    let polar = quadrics[body](hit);
    let arc = acos(clamp(hit.dot(light) / light.dot(light), -1.0, 1.0));
    let cosine = (polar & light) / (polar.norm() * sin(arc));
    let lit = cosine < 0 && !shadowed(hit);
    // Rolled off softly where the light is strongest, rather than clipped, so the brightest faces keep their form.
    let lambert = select(0.0, tanh(-cosine / (sin(arc) * sin(arc))), lit);
    // A glint where the normal, the polar plane's dual along the sphere and facing the eye, bisects the ways
    // back to the eye and on to the light.
    let to_eye = along(hit, eye);
    let normal = along(hit, polar.dual());
    let facing = normal * select(-1.0, 1.0, normal.dot(to_eye) >= 0);
    let halfway = along(hit, to_eye + along(hit, light));
    let glint = select(0.0, pow(max(facing.dot(halfway), 0.0), shininess), lit);
    return colours[body] * (ambient + (1 - ambient) * lambert) + gloss * glint;
}
"""


# --- plumbing -------------------------------------------------------------------------
@cache
def shader() -> Shader:
    # The points a quarter turn from the origin along x, y and z, ahead, right and up from the origin's own view.
    frame = stack((mv.x, mv.y, mv.z)).dual()                                   # [3] Point
    return Shader(supersampled() + source, ga, dict(Point=Point, Plane=Plane, Quadric=Quadric, Motor=Motor),
                  dict(origin=origin, frame=frame, background=background))


def picture(eye_frame: Motor, surfaces: Quadric, colors: np.ndarray, light: Point, fov: float,
            shape: tuple[int, int]) -> np.ndarray:
    """One RGB image, `[rows, columns, 3]` bytes, through a pinhole of the given field of view looking along the
    eye frame's x."""
    traced = shader()(shape, quadrics=surfaces, colours=colors, eye_frame=eye_frame,
                      half_height=np.tan(fov / 2) * shape[0] / shape[1], light=light)["image"]
    return np.round(np.clip(traced, 0, 1) * 255).astype(np.uint8)


def frames(eye_frames: Motor, surfaces: Quadric, colors: np.ndarray, light: Point, fov: float,
           shape: tuple[int, int]) -> Iterator[np.ndarray]:
    """One RGB frame per eye frame."""
    for eye in eye_frames:
        yield picture(eye, surfaces, colors, light, fov, shape)


def draw_walk(eye_frames: Motor, surfaces: Quadric, colors: np.ndarray, light: Point, fov: float,
              shape: tuple[int, int]) -> plt.Figure:
    """The view from the eye's first frame."""
    return image_figure(picture(eye_frames[0], surfaces, colors, light, fov, shape))
