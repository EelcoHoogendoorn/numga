"""Implicit GPU rendering of spherical quadrics on the front hemisphere of the unit sphere.

Each pixel on the orthographic front hemisphere lifts to a null conformal point equal to the sphere's centre plus its
unit spatial direction, and takes the colour of the last quadric in the stack whose quadratic form
`quadrics[body](point) & point` is negative there.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import cache

import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.quadrics.cga_spherical_quadrics.core import Plane, Point, Quadric, ga, mv
from numga.wgsl import Shader, supersampled

# Colours of the background, sphere disk, and rim, and the squared screen radius where the rim begins.
background_hex, disk_hex, rim_hex = "#0a0f1d", "#111827", "#334155"
rim_start = 0.985
# The sphere's centre plus a unit combination of the three spatial directions forms a null point on the unit sphere.
centre = mv.w.dual()                                                           # [] Point-
directions = stack((mv.x.dual(), mv.y.dual(), mv.z.dual()))                    # [3] Point-

source = """
var<storage, read> quadrics: array<Quadric>;
var<storage, read> colours: array<vec3<f32>>;

alias Sample = vec3<f32>;

fn sample(place: vec2<f32>) -> Sample {
    let across = screen(place);
    let foot = (directions * vec3<f32>(across, 0.0)).sum();
    let squared = foot.dual().scalar_norm_squared();
    if (squared > 1) {
        return background;
    }
    if (squared > rim_start) {
        return rim;
    }
    // Lift the screen sample to the front hemisphere as the sphere's centre plus a unit direction.
    let point = centre + foot + toward * sqrt(1 - squared);
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
def hemisphere_shader() -> Shader:
    """Compiles and caches the supersampled WebGPU shader that shades quadric regions on the front hemisphere."""
    return Shader(
        supersampled() + source,
        ga,
        dict(Point=Point, Plane=Plane, Quadric=Quadric),
        dict(
            centre=centre,
            directions=directions,
            toward=directions[2],
            rim_start=rim_start,
            disk=to_rgb(disk_hex),
            rim=to_rgb(rim_hex),
            background=to_rgb(background_hex),
        ),
    )


def vortex_animation(colors: list[str], resolution: int, states: Iterable[Quadric]) -> Iterator[np.ndarray]:
    """Renders each state of `[n_quadrics] Plane+ <- Point-` into a `[resolution, resolution, 3]` `uint8` RGB frame on
    the front hemisphere."""
    rgb = np.array([to_rgb(color) for color in colors])
    painter = hemisphere_shader()
    for quadrics in states:
        yield np.round(np.clip(painter((resolution, resolution), quadrics=quadrics, colours=rgb)["image"], 0, 1)
                       * 255).astype(np.uint8)
