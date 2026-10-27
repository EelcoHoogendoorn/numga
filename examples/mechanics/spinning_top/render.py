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
from examples.util.wgsl import Shader, solids, supersampled
from examples.util.wgsl.solids import lamp_shading, quadric_crossings

mv = core.mv
# Colours of the disc, the stem and the tip, and the contrast of the sectors drawn on each to show its spin, a sector
# an eighth of a turn about the top's axis.
part_colours = np.array([[0.85, 0.30, 0.25], [0.35, 0.35, 0.40], [0.80, 0.70, 0.35]])
sector_contrast = np.array([0.25, 0.0, 0.0])
sector_angle = np.pi / 4
# The ground's colour, its checkerboard's squares and their two tones, and the background.
ground_colour = np.array([0.95, 0.93, 0.88])
square = 0.15
checker_tones = np.array([0.62, 0.80])
background = np.ones(3)
# The share of the light that falls everywhere; the surfaces have no glint.
ambient, gloss, shininess = 0.35, 0.0, 32.0

source = """
// The ground, then the top's parts, placed; each part's colour and the contrast of its sectors.
struct Paint {
    colour: vec3<f32>,
    contrast: f32,
}

var<storage, read> quadrics: array<Quadric>;
var<storage, read> paints: array<Paint>;
var<uniform> motor: Motor;
// A camera looking along its own -z, its rays parallel, starting `back` behind the middle of the view; the image
// half as high as this. Its frame, `axes`, is given from Python.
var<uniform> camera: Motor;
var<uniform> back: f32;
var<uniform> half_height: f32;

alias Sample = vec3<f32>;

const origin = mv.w.dual();

fn sample(place: vec2<f32>) -> Sample {
    // The ray's start in the camera's frame: across and up the image, and `back` toward the viewer.
    let across = screen(place) * half_height;
    let start = camera >> (origin + (axes.dual() * vec3<f32>(across, back)).sum());
    let direction = camera >> -mv.z.dual();
    // The nearest solid the ray enters.
    var nearest = far;
    var shape = 0u;
    for (var index = 0u; index < arrayLength(&quadrics); index++) {
        let distance = entry(along_ray(quadrics[index], start, direction));
        if (distance > 0 && distance < nearest) {
            nearest = distance;
            shape = index;
        }
    }
    if (nearest >= far) {
        return background;
    }
    let point = start + direction * nearest;
    // The quadric's plane at its surface is the tangent plane.
    let tangent = quadrics[shape](point);
    if (shape == 0) {
        let spot = floor(vec2<f32>(mv.x & point, mv.y & point) / square);
        let checker = spot.x + spot.y - 2 * floor((spot.x + spot.y) / 2);
        return lit(mix(checker_tones.x, checker_tones.y, checker) * ground_colour, tangent, direction);
    }
    // Alternating sectors about the top's axis, in its own frame.
    let local = motor << point;
    let turns = floor(atan2(mv.y & local, mv.x & local) / sector_angle);
    let sector = turns - 2 * floor(turns / 2);
    let paint = paints[shape - 1];
    return lit(paint.colour * (1 - paint.contrast * (1 - sector)), tangent, direction);
}
"""


# --- plumbing -------------------------------------------------------------------------
@cache
def shader() -> Shader:
    return Shader(supersampled() + quadric_crossings() + lamp_shading() + source, core.ga,
                  dict(Point=core.Point, Plane=core.Plane, Motor=core.Motor, Quadric=core.Quadric),
                  dict(axes=solids.axes, background=background, ground_colour=ground_colour, square=square,
                       checker_tones=checker_tones, sector_angle=sector_angle))


def frame(motor: core.Motor, parts: core.Quadric, ground: core.Quadric, view: tuple[float, float], centre: core.Point,
          extent: float, pixels: int) -> np.ndarray:
    """One RGB image of the top placed by the motor on the ground, square, of half-width extent about the centre,
    seen from an elevation and azimuth in degrees, the azimuth measured from x."""
    camera = solids.aim(centre, np.radians(view[0]), np.radians(view[1] + 90))                  # [] Motor+
    # The lamp above the viewer's left shoulder.
    toward = camera >> mv.direction([-0.3, 0.8, 1.0])                                         # [] Point-
    lighting = dict(lamp=toward / toward.dual().norm(), ambient=ambient, fill_tilt=0.0, headlight=0.0, gloss=gloss,
                    shininess=shininess)
    placed_parts = motor >> parts(motor << core.Point)                                          # [parts] Plane+ <- Point-
    traced = shader()((pixels, pixels), quadrics=concatenate([ground[None], placed_parts]),
                      paints=dict(colour=part_colours, contrast=sector_contrast), motor=motor, camera=camera,
                      back=20 * extent, half_height=extent, **lighting)["image"]
    return np.round(np.clip(traced, 0, 1) * 255).astype(np.uint8)


def row(motors: core.Motor, parts: core.Quadric, ground: core.Quadric, view: tuple[float, float], centre: core.Point,
        extent: float, pixels: int) -> np.ndarray:
    """Tops side by side in one image, each placed by its motor `[cases]` with its parts `[cases, parts]`, as `frame`
    draws one."""
    return np.concatenate([frame(motor, case_parts, ground, view, centre, extent, pixels)
                           for motor, case_parts in zip(motors, parts)], axis=1)
