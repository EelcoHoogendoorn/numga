"""Renders of the top on its ground, ray traced on the GPU: every ray meets the nearest of the quadrics.

Each shape is one solid quadric, `quadric(point) & point` negative inside. Along a ray the pairing is a
quadratic in the distance, and the ray enters the solid at the root where it falls through zero; the quadric's
plane there is the tangent plane. The ground carries a checkerboard, and each part alternating sectors about the
top's axis, read in the top's own frame, `motor << point`.
"""

from __future__ import annotations

from functools import cache

import numpy as np

from numga import concatenate
from examples.mechanics.spinning_top import core
from examples.wgsl import Shader, solids, supersampled

mv = core.mv
# Colours of the disc, the stem and the tip, and the contrast of the sectors drawn on each to show its spin.
part_colours = np.array([[0.85, 0.30, 0.25], [0.35, 0.35, 0.40], [0.80, 0.70, 0.35]])
sector_contrast = np.array([0.25, 0.0, 0.0])

source = supersampled() + """
// The ground, then the top's parts, placed; each part's colour and the contrast of its sectors.
struct Paint {
    colour: vec3<f32>,
    contrast: f32,
}

var<storage, read> quadrics: array<Quadric>;
var<storage, read> paints: array<Paint>;
var<uniform> motor: Motor;
// A camera looking along its own -z, its rays parallel, starting `back` behind the middle of the view; the image
// half as high as this; the lamp, toward it.
var<uniform> camera: Motor;
var<uniform> back: f32;
var<uniform> half_height: f32;
var<uniform> lamp: Point;

alias Sample = vec3<f32>;

const origin = mv.w.dual();
const far = 1e30;
const ground_colour = vec3<f32>(0.95, 0.93, 0.88);
const square = 0.15;
const sector_angle = 0.7853982;

fn sample(place: vec2<f32>) -> Sample {
    let across = screen(place) * half_height;
    let start = camera >> (origin + mv.z.dual() * back + mv.x.dual() * across.x + mv.y.dual() * across.y);
    let direction = camera >> -mv.z.dual();
    var nearest = far;
    var shape = 0u;
    for (var index = 0u; index < arrayLength(&quadrics); index++) {
        let quadric = quadrics[index];
        // Along the ray the quadric is a t^2 + 2 b t + c; it enters the solid where the pairing falls through
        // zero, the root c / (sqrt(b^2 - a c) - b), which holds where the square term vanishes as well.
        let a = quadric(direction) & direction;
        let b = quadric(direction) & start;
        let c = quadric(start) & start;
        let discriminant = b * b - a * c;
        let distance = c / (sqrt(max(discriminant, 0.0)) - b);
        if (discriminant >= 0 && distance > 0 && distance < nearest) {
            nearest = distance;
            shape = index;
        }
    }
    if (nearest >= far) {
        return vec3<f32>(1.0);
    }
    let point = start + direction * nearest;
    // The quadric's plane at its surface is the tangent plane; lit on either side.
    let tangent = quadrics[shape](point);
    let facing = abs(tangent & lamp) / (tangent | tangent).square_root();
    if (shape == 0) {
        let spot = floor(vec2<f32>(mv.x & point, mv.y & point) / square);
        let checker = spot.x + spot.y - 2 * floor((spot.x + spot.y) / 2);
        return (0.62 + 0.18 * checker) * ground_colour * (0.45 + 0.55 * facing);
    }
    // Alternating sectors about the top's axis, in its own frame.
    let local = motor << point;
    let turns = floor(atan2(mv.y & local, mv.x & local) / sector_angle);
    let sector = turns - 2 * floor(turns / 2);
    let paint = paints[shape - 1];
    return paint.colour * (1 - paint.contrast * (1 - sector)) * (0.35 + 0.65 * facing);
}
"""


# --- plumbing -------------------------------------------------------------------------
@cache
def shader() -> Shader:
    return Shader(source, core.ga, dict(Point=core.Point, Plane=core.Plane, Motor=core.Motor, Quadric=core.Quadric))


def frame(motor: core.Motor, parts: core.Quadric, ground: core.Quadric, view: tuple[float, float], centre: np.ndarray,
          extent: float, pixels: int) -> np.ndarray:
    """One RGB image of the top placed by the motor on the ground, square, of half-width extent about the centre,
    seen from an elevation and azimuth in degrees, the azimuth measured from x."""
    camera = solids.aim(mv.point(centre), np.radians(view[0]), np.radians(view[1] + 90))         # [] Motor
    # The lamp above the viewer's left shoulder.
    lamp = camera >> mv.direction([-0.3, 0.8, 1.0])
    lamp = lamp / (-(lamp.dual() & lamp)).square_root()
    placed = motor >> parts(motor << core.Point)                                                # [parts] Quadric
    traced = shader()((pixels, pixels), quadrics=concatenate([ground[None], placed]),
                      paints=dict(colour=part_colours, contrast=sector_contrast), motor=motor, camera=camera,
                      back=20 * extent, half_height=extent, lamp=lamp)["image"]
    return np.round(np.clip(traced, 0, 1) * 255).astype(np.uint8)
