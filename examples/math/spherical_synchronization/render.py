"""Orthographic views of a population and its carried circles on the see-through unit sphere, on the GPU, in the
sphere's own algebra.

A ray along the view crosses the sphere at two points, one on its near side and one on its far side, each a null
ray `direction + time`. A circle is inked at either where its cut nearly pairs to nothing with the point,
`cut & point`, the shortfall measured across the view by how fast the pairing changes there. The population are
dots about their places on the screen. The far side shows faintly through the near one.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import cache

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.util.wgsl import Shader, image_figure, supersampled
from examples.math.spherical_synchronization.core import Cut, Point, Rotor, ga, mv

size = 300
# The view, elevation and azimuth in radians, azimuth zero looking along +y, facing the side the population
# gathers on; and the half width it shows.
elevation, azimuth = np.radians([20.0, -45.0])
half_width = 1.07
particle_colours = np.array([
    [0.90, 0.36, 0.20], [0.91, 0.59, 0.16], [0.24, 0.55, 0.61],
    [0.32, 0.43, 0.68], [0.62, 0.39, 0.61],
])
# The circles' width, the dots' radius on the near side and their white rim, and the sphere's outline, in pixels.
line_width = 1.0
dot_radius = 2.0
rim_width = 0.4
outline_width = 0.8

source = """
// A camera turned to look along its own -z at the sphere's centre; the image half as high as this.
var<uniform> camera: Rotor;
var<uniform> half_height: f32;
var<storage, read> particles: array<Particle>;
var<storage, read> cuts: array<Cut>;
// Widths across the view, in the sphere's radii.
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
// How deep a point must lie toward the viewer to show at full strength, and the strength of the far side.
const fade_depth = 1.0 / 12;
const far_cuts = 0.18;
const far_particles = 0.25;

// The point of the sphere at a place across the view and a depth toward the viewer.
fn on_sphere(across: vec2<f32>, depth: f32) -> Point {
    return camera >> (time + (frame * vec3<f32>(across, depth)).sum());
}

// How strongly a point at a depth toward the viewer shows, from a floor on the far side to one on the near.
fn showing(depth: f32, floor: f32) -> f32 {
    return floor + (1 - floor) * clamp(depth / fade_depth + 0.5, 0.0, 1.0);
}

// A colour seen with a strength, washed toward white.
fn faded(colour: vec3<f32>, strength: f32) -> vec3<f32> {
    return white - (white - colour) * strength;
}

// The colour of the circles at a point of the sphere, a place across the view and a depth toward the viewer,
// over a colour beneath: a cut's pairing with the point, over how fast it changes across the view, is how far
// across the view the point lies from that cut's circle.
fn inked_cuts(beneath: vec3<f32>, across: vec2<f32>, depth: f32) -> vec3<f32> {
    let right = camera >> mv.x.dual();
    let up = camera >> mv.y.dual();
    let toward = camera >> mv.z.dual();
    let point = on_sphere(across, depth);
    // Moving across the view, the point keeps to the sphere by moving along the view as well.
    let rightward = right - toward * (across.x / depth);
    let upward = up - toward * (across.y / depth);
    var colour = beneath;
    for (var index = 0u; index < arrayLength(&cuts); index++) {
        let cut = cuts[index];
        let change = right.dual_inverse() * (cut & rightward) + up.dual_inverse() * (cut & upward);
        if (abs(cut & point) < line_half_width * change.norm()) {
            colour = faded(cut_colour, showing(depth, far_cuts));
        }
    }
    return colour;
}

// The nearest dot covering a place across the view, among those on one side of the sphere, over a colour beneath.
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
    // The sphere, softly lit by the lamp, white beyond it, its outline drawn.
    let shade = 0.92 + 0.07 * clamp(lamp & on_sphere(across, depth), 0.0, 1.0);
    var colour = select(white, shade * vec3<f32>(1.0, 1.005, 1.01), distance <= 1);
    colour = select(colour, outline_colour, abs(distance - 1) < outline_half_width);
    colour = dots(colour, across, false);
    if (distance < 1) {
        colour = inked_cuts(inked_cuts(colour, across, -depth), across, depth);
    }
    return dots(colour, across, true);
}
"""


# --- plumbing -------------------------------------------------------------------------
@cache
def shader() -> Shader:
    return Shader(supersampled() + source, ga, dict(Point=Point, Cut=Cut, Rotor=Rotor), dict(lamp=lamp, frame=stack((mv.x, mv.y, mv.z)).dual()))


azimuth_turn = (mv.xy * (-azimuth / 2)).exp()                               # [] Rotor+
elevation_tilt = (mv.yz * (-(np.pi / 2 - elevation) / 2)).exp()             # [] Rotor+
camera = azimuth_turn * elevation_tilt                                       # [] Rotor+
# The lamp, up and to the left of the viewer and in front, a cut through the centre: its pairing with a point of the
# sphere's near side is how squarely the point faces it.
lamp = camera >> (mv.y * 0.35 + mv.z * 0.9 - mv.x * 0.25)                    # [] Cut+


def coordinates(points: Point) -> np.ndarray:
    """Positions of directions on the last axis: each paired with the planes across x, y and z."""
    mv = points.context.multivector
    return (stack((mv.x, mv.y, mv.z), axis=-1) & points[..., None]).to_array()


def picture(population: Point, cuts: Cut) -> np.ndarray:
    """One population and its circles, `[size, size, 3]` bytes."""
    pixels = size
    pixel = 2 * half_width / pixels
    colours = particle_colours[np.arange(population.shape[-1]) % len(particle_colours)]
    traced = shader()((pixels, pixels), camera=camera, half_height=half_width,
                      particles=dict(place=population, colour=colours), cuts=cuts,
                      line_half_width=line_width / 2 * pixel, dot_radius=dot_radius * pixel,
                      rim_width=rim_width * pixel, outline_half_width=outline_width / 2 * pixel)["image"]
    return np.round(np.clip(traced, 0, 1) * 255).astype(np.uint8)


def draw(population: Point, cuts: Cut) -> plt.Figure:
    """Draw one population and the circles carried alongside it."""
    return image_figure(picture(population, cuts))


def frames(states: Iterable[tuple[Point, Cut]]) -> Iterator[np.ndarray]:
    """Animate a fixed camera while the common motion gathers the population."""
    for population, cuts in states:
        yield picture(population, cuts)
