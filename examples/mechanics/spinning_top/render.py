"""GPU ray tracing renders quadric tops on a quadric ground by intersecting each ray with the nearest solid.

Each shape is modelled as one solid quadric whose pairing `quadric(point) & point` is negative inside the
solid. Along any camera ray that pairing is quadratic in the ray distance, entering the solid where the
quadratic crosses downward through zero, and the quadric's image plane at that entry point is the surface
tangent plane. The ground carries a checkerboard and each part carries alternating spin sectors evaluated
in the top's body frame `motor << point`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import cache

import numpy as np

from numga import concatenate
from numga.render import solids
from numga.render.solids import lamp_shading, quadric_crossings
from numga.wgsl import Shader, supersampled
from examples.mechanics.spinning_top import core

# Colours of the disc, stem, and tip, with alternating eighth-turn sectors painted on the disc to show spin.
part_colours = np.array([[0.85, 0.30, 0.25], [0.35, 0.35, 0.40], [0.80, 0.70, 0.35]])
sector_contrast = np.array([0.25, 0.0, 0.0])
sector_angle = np.pi / 4
# Ground colour, checkerboard square width, alternating checker tones, and white background.
ground_colour = np.array([0.95, 0.93, 0.88])
square_size = 0.15
checker_tones = np.array([0.62, 0.80])
background = np.ones(3)
ambient, gloss, shininess = 0.35, 0.0, 32.0

ray_source = '''
// Placed ground followed by the top's placed parts, with each part's base colour and sector contrast.
struct Paint {
    colour: vec3<f32>,
    contrast: f32,
}

var<storage, read> quadrics: array<Quadric>;
var<storage, read> paints: array<Paint>;
var<uniform> motor: Motor;
// Orthographic camera looking along its local -z axis from `back` behind the view centre.
var<uniform> camera: Motor;
var<uniform> back: f32;
var<uniform> half_height: f32;

alias Sample = vec3<f32>;

const origin = mv.w.dual();

fn sample(place: vec2<f32>) -> Sample {
    // Ray origin in the camera frame across and up the image plane, `back` toward the viewer.
    let across = screen(place) * half_height;
    let start = camera >> (origin + (axes.dual() * vec3<f32>(across, back)).sum());
    let direction = camera >> -mv.z.dual();
    // Nearest solid entered along the ray.
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
    // Evaluating the quadric at a surface point gives its outward tangent plane.
    let tangent = quadrics[shape](point);
    if (shape == 0) {
        let spot = floor(vec2<f32>(mv.x & point, mv.y & point) / square_size);
        let checker = spot.x + spot.y - 2 * floor((spot.x + spot.y) / 2);
        return lit(mix(checker_tones.x, checker_tones.y, checker) * ground_colour, tangent);
    }
    // Alternating sectors around the symmetry axis in the top's body frame.
    let local = motor << point;
    let turns = floor(atan2(mv.y & local, mv.x & local) / sector_angle);
    let sector = turns - 2 * floor(turns / 2);
    let paint = paints[shape - 1];
    return lit(paint.colour * (1 - paint.contrast * (1 - sector)), tangent);
}
'''


# --- plumbing -------------------------------------------------------------------------
@cache
def quadric_shader() -> Shader:
    """Caches the compiled WGSL ray-tracing shader for the ground and ellipsoid quadrics."""
    return Shader(supersampled() + quadric_crossings() + lamp_shading() + solids.tangent_seen() + ray_source, core.ga,
                  dict(Point=core.Point, Plane=core.Plane, Tangent=core.Plane, Motor=core.Motor, Quadric=core.Quadric),
                  dict(axes=solids.axes, background=background, ground_colour=ground_colour, square_size=square_size,
                       checker_tones=checker_tones, sector_angle=sector_angle))


def frame(motor: core.Motor, parts: core.Quadric, ground: core.Quadric, view: tuple[float, float], centre: core.Point,
          extent: float, n_pixels: int) -> np.ndarray:
    """Renders one `[n_pixels, n_pixels, 3]` RGB image of `parts` placed by `motor` on `ground`."""
    camera = solids.aim(centre, np.radians(view[0]), np.radians(view[1] + 90))                  # [] Rotor+
    # Key lamp above the viewer's left shoulder in camera coordinates.
    lighting = dict(lamp=np.array([-0.3, 0.8, 1.0]) / np.linalg.norm([-0.3, 0.8, 1.0]), ambient=ambient, fill_tilt=0.0,
                    headlight=0.0, gloss=gloss, shininess=shininess)
    placed_parts = motor >> parts(motor << core.Point)                                          # [...] Plane+ <- Point-
    traced = quadric_shader()((n_pixels, n_pixels), quadrics=concatenate([ground[None], placed_parts]),
                              paints=dict(colour=part_colours, contrast=sector_contrast), motor=motor, camera=camera,
                              back=20 * extent, half_height=extent, **lighting,
                              **solids.steps(solids.View(camera, np.full(2, extent), n_pixels)))["image"]
    return np.round(np.clip(traced, 0, 1) * 255).astype(np.uint8)


def row(motors: core.Motor, parts: core.Quadric, ground: core.Quadric, view: tuple[float, float], centre: core.Point,
        extent: float, n_pixels: int) -> np.ndarray:
    """Places multiple tops side by side into one horizontal RGB strip."""
    return np.concatenate([frame(motor, case_parts, ground, view, centre, extent, n_pixels)
                           for motor, case_parts in zip(motors, parts)], axis=1)


def row_frames(parts: core.Quadric, ground: core.Quadric, view: tuple[float, float], centre: core.Point,
               extent: float, n_pixels: int, states: Iterable[tuple[core.Motor, core.Rate]]) -> Iterator[np.ndarray]:
    """Yields side-by-side RGB frames for a streamed sequence of `(motors, rates)` states."""
    for motors, _ in states:
        yield row(motors, parts, ground, view, centre, extent, n_pixels)
