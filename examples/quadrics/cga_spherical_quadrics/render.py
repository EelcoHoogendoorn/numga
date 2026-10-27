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

from numga import stack
from examples.quadrics.cga_spherical_quadrics.core import Plane, Point, Quadric, ga, mv
from examples.util.wgsl import Shader, supersampled

# The colours of the background, of the disk and of its rim; where the rim begins: the squared distance from the
# disk's centre, in its radii, past which a pixel is rim.
BACKGROUND, DISK, RIM = "#0a0f1d", "#111827", "#334155"
RIM_START = 0.985
# The vectors x, y and z, whose sum with w, each weighted by a unit coordinate, is a point's null vector.
axes = stack((mv.x, mv.y, mv.z))                                              # [3] Plane+

source = """
var<storage, read> quadrics: array<Quadric>;
var<storage, read> colours: array<vec3<f32>>;

alias Sample = vec3<f32>;

fn sample(place: vec2<f32>) -> Sample {
    let across = screen(place);
    let squared = dot(across, across);
    if (squared > 1) {
        return background;
    }
    if (squared > rim_start) {
        return rim;
    }
    // The point of S² under the sample, on the front hemisphere: the dual of its null vector.
    let point = ((axes * vec3<f32>(across, sqrt(1 - squared))).sum() + mv.w).dual();
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
    return Shader(supersampled() + source, ga, dict(Point=Point, Plane=Plane, Quadric=Quadric),
                  dict(axes=axes, rim_start=RIM_START, disk=to_rgb(DISK), rim=to_rgb(RIM), background=to_rgb(BACKGROUND)))


def vortex_frames(world: Quadric, colors: list[str], resolution: int) -> Iterator[np.ndarray]:
    """One frame per row of world quadrics: a pixel takes the colour of the last quadric whose form is
    negative there, averaged over its samples."""
    rgb = np.array([to_rgb(color) for color in colors])
    painter = shader()
    for quadrics in world:
        yield np.round(np.clip(painter((resolution, resolution), quadrics=quadrics, colours=rgb)["image"], 0, 1)
                       * 255).astype(np.uint8)
