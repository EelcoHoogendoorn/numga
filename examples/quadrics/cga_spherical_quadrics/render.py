"""Spherical quadrics on S², painted on the GPU by a shader whose geometry is written in numga.

Each pixel of the front hemisphere, seen along z, is a point of S², the dual of its null vector,
`(mv.x * u + mv.y * v + mv.z * height + mv.w).dual()`, and it takes the colour of the last body whose
quadric's form, `quadrics[body](point) & point`, is negative there.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import cache

import numpy as np
from matplotlib.colors import to_rgb

from examples.quadrics.cga_spherical_quadrics.core import Plane, Point, Quadric, ga
from examples.quadrics.elliptic_physics.render import BACKGROUND, DISK, RIM
from examples.util.wgsl import Shader, supersampled

source = """
var<storage, read> quadrics: array<Quadric>;
var<storage, read> colours: array<vec3<f32>>;
var<uniform> disk: vec3<f32>;
var<uniform> rim: vec3<f32>;
var<uniform> background: vec3<f32>;

alias Sample = vec3<f32>;

// The share of the disk's radius inside its rim.
const rim_start = 0.985;

fn sample(place: vec2<f32>) -> Sample {
    let across = screen(place);
    let squared = dot(across, across);
    if (squared > 1) {
        return background;
    }
    if (squared > rim_start) {
        return rim;
    }
    // The point of S² under the pixel, on the front hemisphere.
    let point = (mv.x * across.x + mv.y * across.y + mv.z * sqrt(1 - squared) + mv.w).dual();
    var colour = disk;
    for (var body = 0u; body < arrayLength(&quadrics); body++) {
        if ((quadrics[body](point) & point) < 0) {
            colour = colours[body];
        }
    }
    return colour;
}
"""


# --- plumbing -------------------------------------------------------------------------
@cache
def shader() -> Shader:
    return Shader(supersampled() + source, ga, dict(Point=Point, Plane=Plane, Quadric=Quadric), {})


def vortex_frames(world: Quadric, colors: list[str], resolution: int) -> Iterator[np.ndarray]:
    """One frame per row of world quadrics: a pixel takes the colour of the last quadric whose form is
    negative there, averaged over its samples."""
    rgb = np.array([to_rgb(color) for color in colors])
    painter = shader()
    for quadrics in world:
        yield np.round(np.clip(painter((resolution, resolution), quadrics=quadrics, colours=rgb, disk=to_rgb(DISK),
                                       rim=to_rgb(RIM), background=to_rgb(BACKGROUND))["image"], 0, 1) * 255).astype(np.uint8)
