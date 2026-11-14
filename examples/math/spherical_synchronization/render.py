"""Orthographic GPU views render a population of directions and its carried circles on the translucent
unit sphere directly in the sphere's conformal algebra.

Each viewing ray crosses the unit sphere at a near point and a far point, each represented as a null ray
`direction + time`. A circle is inked wherever `cut & point` nearly vanishes relative to the rate at
which that pairing changes across the screen. Particles appear as dots with the far hemisphere faded
behind the near one.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import cache

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from numga.wgsl import Shader, supersampled
from examples.util.panels import image_figure
from examples.math.spherical_synchronization.core import Cut, Point, Rotor, ga, mv

size = 300
# Camera elevation and azimuth in radians (azimuth zero looks along +y toward the gathering side) and half-width.
elevation, azimuth = np.radians([20.0, -45.0])
half_width = 1.07
particle_colours = np.array([
    [0.90, 0.36, 0.20], [0.91, 0.59, 0.16], [0.24, 0.55, 0.61],
    [0.32, 0.43, 0.68], [0.62, 0.39, 0.61],
])
# Circle stroke width, near-side dot radius, white dot rim, and sphere outline width in pixels.
line_width = 1.0
dot_radius = 2.0
rim_width = 0.4
outline_width = 0.8

source = '''
// A camera turned to look along its own -z at the sphere's centre; the image half as high as this.
var<uniform> camera: Rotor;
var<uniform> half_height: f32;
var<storage, read> particles: array<Particle>;
var<storage, read> cuts: array<Cut>;
// Explicit cut count avoids inlining an arrayLength-bounded loop twice on Mesa's software Vulkan driver.
var<uniform> n_cuts: u32;
// Widths across the view, in sphere radii.
var<uniform> line_half_width: f32;
var<uniform> dot_radius: f32;
var<uniform> rim_width: f32;
var<uniform> outline_half_width: f32;

alias Sample = vec3<f32>;

struct Particle {
    place: Point,
    colour: vec3<f32>,
}

const time = -mv.t.dual();
const cut_colour = vec3<f32>(0.12, 0.42, 0.46);
const outline_colour = vec3<f32>(0.76, 0.78, 0.78);
const white = vec3<f32>(1.0);
// Depth toward the viewer needed for full opacity, and minimum strengths on the far hemisphere.
const fade_depth = 1.0 / 12;
const far_cuts = 0.18;
const far_particles = 0.25;

// Null ray of the sphere point at screen offset `across` and signed `depth` toward the viewer.
fn on_sphere(across: vec2<f32>, depth: f32) -> Point {
    return camera >> (time + (frame * vec3<f32>(across, depth)).sum());
}

// Opacity transition from `floor` on the far hemisphere to 1 on the near hemisphere.
fn showing(depth: f32, floor: f32) -> f32 {
    return floor + (1 - floor) * clamp(depth / fade_depth + 0.5, 0.0, 1.0);
}

// Blends `colour` toward white according to `strength`.
fn faded(colour: vec3<f32>, strength: f32) -> vec3<f32> {
    return white - (white - colour) * strength;
}

// Inks every circle whose cut pairing `cut & point` lies within `line_half_width` of zero in screen units.
fn inked_cuts(beneath: vec3<f32>, across: vec2<f32>, depth: f32) -> vec3<f32> {
    let right = camera >> mv.x.dual();
    let up = camera >> mv.y.dual();
    let toward = camera >> mv.z.dual();
    let point = on_sphere(across, depth);
    // Moving across the screen keeps the point on the sphere by shifting along the viewing ray as well.
    let rightward = right - toward * (across.x / depth);
    let upward = up - toward * (across.y / depth);
    var colour = beneath;
    for (var index = 0u; index < n_cuts; index++) {
        let cut = cuts[index];
        let change = right.dual_inverse() * (cut & rightward) + up.dual_inverse() * (cut & upward);
        if (abs(cut & point) < line_half_width * change.norm()) {
            colour = faded(cut_colour, showing(depth, far_cuts));
        }
    }
    return colour;
}

// Shades the nearest particle dot on one hemisphere covering screen offset `across`.
fn dots(beneath: vec3<f32>, across: vec2<f32>, near_side: bool) -> vec3<f32> {
    var colour = beneath;
    var nearest = -2.0;
    for (var index = 0u; index < arrayLength(&particles); index++) {
        let particle = particles[index];
        let depth = (camera >> mv.z) & particle.place;
        let strength = showing(depth, far_particles);
        let radius = dot_radius * (0.75 + 0.25 * strength);
        let offset = on_sphere(across, 0.0) - particle.place;
        let screen_offset = offset - (camera >> mv.z.dual()) * ((camera >> mv.z) & offset);
        let distance = screen_offset.dual().norm();
        if (distance < radius && depth > nearest && (depth >= 0) == near_side) {
            nearest = depth;
            colour = select(white, faded(particle.colour, strength), distance < radius - rim_width);
        }
    }
    return colour;
}

fn sample(place: vec2<f32>) -> Sample {
    let across = screen(place) * half_height;
    let foot = (frame * vec3<f32>(across, 0.0)).sum();
    let distance = foot.dual().norm();
    let depth = sqrt(max(1 - distance * distance, 0.0));
    // Softly lit sphere interior over a white background with a thin silhouette outline.
    let shade = 0.92 + 0.07 * clamp(lamp & on_sphere(across, depth), 0.0, 1.0);
    var colour = select(white, shade * vec3<f32>(1.0, 1.005, 1.01), distance <= 1);
    colour = select(colour, outline_colour, abs(distance - 1) < outline_half_width);
    colour = dots(colour, across, false);
    if (distance < 1) {
        colour = inked_cuts(inked_cuts(colour, across, -depth), across, depth);
    }
    return dots(colour, across, true);
}
'''


# --- plumbing -------------------------------------------------------------------------
@cache
def shader() -> Shader:
    """Compiles and caches the supersampled WGSL sphere shader."""
    return Shader(
        supersampled() + source,
        ga,
        dict(Point=Point, Cut=Cut, Rotor=Rotor),
        dict(lamp=lamp, frame=stack((mv.x, mv.y, mv.z)).dual()),
    )


azimuth_turn = (mv.xy * (-azimuth / 2)).exp()                               # [] Rotor+
elevation_tilt = (mv.yz * (-(np.pi / 2 - elevation) / 2)).exp()             # [] Rotor+
camera = azimuth_turn * elevation_tilt                                       # [] Rotor+
# Directional lamp cut through the origin, up and to the viewer's front-left.
lamp = camera >> (mv.y * 0.35 + mv.z * 0.9 - mv.x * 0.25)                    # [] Cut+


def coordinates(points: Point) -> np.ndarray:
    """Pairing `points` (`[...] Point-`) with the three coordinate planes reads out `[..., 3]` Cartesian
    positions."""
    basis_mv = points.context.multivector
    return (stack((basis_mv.x, basis_mv.y, basis_mv.z), axis=-1) & points[..., None]).to_array()


def picture(population: Point, cuts: Cut) -> np.ndarray:
    """Renders `population` and `cuts` on the translucent sphere as a `[size, size, 3]` `uint8` image."""
    pixels = size
    pixel = 2 * half_width / pixels
    colours = particle_colours[np.arange(population.shape[-1]) % len(particle_colours)]
    traced = shader()(
        (pixels, pixels), camera=camera, half_height=half_width,
        particles=dict(place=population, colour=colours), cuts=cuts, n_cuts=np.uint32(cuts.shape[0]),
        line_half_width=line_width / 2 * pixel, dot_radius=dot_radius * pixel,
        rim_width=rim_width * pixel, outline_half_width=outline_width / 2 * pixel,
    )["image"]
    return np.round(np.clip(traced, 0, 1) * 255).astype(np.uint8)


def sphere_figure(population: Point, cuts: Cut) -> plt.Figure:
    """Wraps a rendered sphere frame of `population` and `cuts` in a matplotlib figure."""
    return image_figure(picture(population, cuts))


def frames(states: Iterable[tuple[Point, Cut]]) -> Iterator[np.ndarray]:
    """Yields `[size, size, 3]` `uint8` frames of `states` from the fixed orthographic camera."""
    for population, cuts in states:
        yield picture(population, cuts)
