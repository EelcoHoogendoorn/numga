"""Ray tracing on the 3-sphere, on the GPU: each pixel's great circle meets the bodies' quadrics, then shading.

The ray from the eye `e` along the direction `d` of a pixel is the great circle `e * cos(t) + d * sin(t)`. A body's
quadric bound to it is `A * u**2 + 2 * B * u + C` in `u = cot(t)`, with `A = e & Q(e)`, `B = e & Q(d)` and
`C = d & Q(d)`, so the nearest hit, the smallest angle, is the largest root: the depth the screen conics give,
taken here straight from the quadric. The point it reaches is `e * u + d`, kept with its own sign.
"""

from __future__ import annotations

from functools import cache

import matplotlib.pyplot as plt
import numpy as np

from examples.quadrics.s3_raytracer.core import Motor, Plane, Point, Quadric, ScreenPoint, ga, mv, origin
from examples.wgsl import Shader, image_figure, supersampled

background = 0.02

source = """
// The bodies' quadrics and colours; the eye, its view straight ahead, right and up, each a point a quarter turn
// from the eye; how far up the image's top edge looks, and the light.
var<storage, read> quadrics: array<Quadric>;
var<storage, read> colours: array<vec3<f32>>;
var<uniform> eye: Point;
var<uniform> ahead: Point;
var<uniform> right: Point;
var<uniform> up: Point;
var<uniform> half_height: f32;
var<uniform> light: Point;

alias Sample = vec3<f32>;

const ambient = 0.15;
// The glint's share and sharpness.
const gloss = 0.35;
const shininess = 40.0;
const background = BACKGROUND;
// How far along the arc to the light a shadow test starts.
const clearance = 1e-4;

// A point's direction along the sphere at a place on it, unit: its part square to the place.
fn along(place: Point, toward: Point) -> Point {
    let tangent = toward - place * place.dot(toward);
    return tangent / sqrt(tangent.dot(tangent));
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
    let direction = ahead + right * across.x + up * across.y;
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
    let reached = eye * depth + direction;
    let hit = reached / sqrt(-(reached | reached));
    // The great circle out of the light reaches the hit along an arc; the surface is lit where that arc
    // arrives from outside, which the pairing of its polar plane with the light decides, with the falloff
    // of a point source on the sphere.
    let polar = quadrics[body](hit);
    let arc = acos(clamp((hit | light) / (light | light), -1.0, 1.0));
    let cosine = (polar & light) / ((polar | polar).square_root() * sin(arc));
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
""".replace("BACKGROUND", repr(background))


# --- plumbing -------------------------------------------------------------------------
@cache
def shader(samples_per_side: int) -> Shader:
    return Shader(supersampled(samples_per_side) + source, ga, dict(Point=Point, Plane=Plane, Quadric=Quadric))


def picture(eye_frame: Motor, surfaces: Quadric, colors: np.ndarray, light: Point, fov: float,
            shape: tuple[int, int], supersample: int) -> np.ndarray:
    """One RGB image, `[rows, columns, 3]` bytes, through a pinhole of the given field of view looking along the
    eye frame's x."""
    ahead, right, up = eye_frame >> mv(ScreenPoint, np.eye(3))                 # [3] Point
    traced = shader(supersample)(shape, quadrics=surfaces, colours=colors, eye=eye_frame >> origin, ahead=ahead,
                                 right=right, up=up, half_height=np.tan(fov / 2) * shape[0] / shape[1],
                                 light=light)["image"]
    return np.round(np.clip(traced, 0, 1) * 255).astype(np.uint8)


def frames(eye_frames: Motor, surfaces: Quadric, colors: np.ndarray, light: Point, fov: float,
           shape: tuple[int, int], supersample: int) -> list[np.ndarray]:
    """One RGB frame per eye frame."""
    return [picture(eye, surfaces, colors, light, fov, shape, supersample) for eye in eye_frames]


def draw_walk(eye_frames: Motor, surfaces: Quadric, colors: np.ndarray, light: Point, fov: float,
              shape: tuple[int, int], supersample: int) -> plt.Figure:
    """The view from the eye's first frame."""
    return image_figure(picture(eye_frames[0], surfaces, colors, light, fov, shape, supersample))
