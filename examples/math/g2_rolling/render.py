"""A marked ball rolling over a sphere, with its contact route painted on the sphere, ray traced on the GPU.

Each ray meets the sphere and the ball, both found by where the ray's squared distance from their centres equals
their squared radius. The ball's paint is fixed to its material: the normal where the ray meets it, carried back by
the attitude, `attitude << normal`, picks its octant. The routes are inked onto the sphere wherever a point of it
lies, as the view sees it, within a line's half width of a segment of a route.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import cache

import numpy as np

from examples.util.wgsl import Shader, supersampled
from numga import concatenate

from .core import Rotor, Vector, ga, mv

route_colours = np.array([[27, 111, 152], [213, 99, 36], [113, 67, 151]]) / 255
# The route lines' width and their end dots' radius, in pixels; the share of the view left around the scene.
line_width = 3
dot_radius = 4
margin = 0.07

source = """
// A camera turned to look along its own -z at the middle of the view, its rays parallel, starting `back` behind
// it; the image half as high as this; the lamp, toward it.
var<uniform> camera: Rotor;
var<uniform> middle: Vector;
var<uniform> back: f32;
var<uniform> half_height: f32;
var<uniform> lamp: Vector;
// The sphere about the origin, and the ball touching it along `contact`, turned by its attitude.
var<uniform> sphere_radius: f32;
var<uniform> ball_radius: f32;
var<uniform> contact: Vector;
var<uniform> attitude: Rotor;
// The routes' segments and their ends, and the width they are inked at, in the scene's lengths.
var<storage, read> segments: array<Segment>;
var<storage, read> ends: array<End>;
var<uniform> line_half_width: f32;
var<uniform> dot_radius: f32;
var<uniform> rim_width: f32;

alias Sample = vec3<f32>;

struct Segment {
    start: Vector,
    end: Vector,
    colour: vec3<f32>,
}

struct End {
    place: Vector,
    colour: vec3<f32>,
}

const background = vec3<f32>(0.98, 0.98, 0.96);
const surface = vec3<f32>(0.85, 0.87, 0.86);
const seams = vec3<f32>(0.13, 0.15, 0.18);
const rim = vec3<f32>(0.96, 0.96, 0.93);
const seam_width = 0.035;
const octants = array<vec3<f32>, 8>(
    vec3<f32>(0.19, 0.44, 0.66), vec3<f32>(0.93, 0.48, 0.19),
    vec3<f32>(0.94, 0.75, 0.32), vec3<f32>(0.19, 0.64, 0.61),
    vec3<f32>(0.81, 0.35, 0.39), vec3<f32>(0.40, 0.68, 0.84),
    vec3<f32>(0.61, 0.47, 0.69), vec3<f32>(0.91, 0.89, 0.76));
const far = 1e30;

// How far along a unit direction a ray from a start first meets a ball, or far if it misses.
fn meet(start: Vector, direction: Vector, centre: Vector, radius: f32) -> f32 {
    let offset = start - centre;
    let along = offset | direction;
    let discriminant = along * along - (offset | offset) + radius * radius;
    return select(far, -along - sqrt(discriminant), discriminant >= 0);
}

// A difference of places as the view sees it: its part along the view taken away.
fn across_view(offset: Vector, direction: Vector) -> Vector {
    return offset - direction * (offset | direction);
}

// The sphere's paint at a point on it: the nearest route's colour within a line of it, the ends' dots over them,
// all measured across the view, and only on the sphere's half facing it.
fn route_paint(point: Vector, direction: Vector) -> vec3<f32> {
    var paint = surface;
    var nearest = line_half_width;
    for (var index = 0u; index < arrayLength(&segments); index++) {
        let segment = segments[index];
        let span = across_view(segment.end - segment.start, direction);
        let offset = across_view(point - segment.start, direction);
        let along = clamp((offset | span) / (span | span), 0.0, 1.0);
        let gap = offset - span * along;
        let distance = sqrt(gap | gap);
        if (distance < nearest && ((segment.start + segment.end) | direction) < 0) {
            nearest = distance;
            paint = segment.colour;
        }
    }
    for (var index = 0u; index < arrayLength(&ends); index++) {
        let gap = across_view(point - ends[index].place, direction);
        let distance = sqrt(gap | gap);
        if (distance < dot_radius && (ends[index].place | direction) < 0) {
            paint = select(rim, ends[index].colour, distance < dot_radius - rim_width);
        }
    }
    return paint;
}

fn sample(place: vec2<f32>) -> Sample {
    let across = screen(place) * half_height;
    let start = middle + (camera >> (mv.z * back + mv.x * across.x + mv.y * across.y));
    let direction = camera >> -mv.z;
    let centre = contact * (sphere_radius + ball_radius);
    let to_sphere = meet(start, direction, mv.x * 0, sphere_radius);
    let to_ball = meet(start, direction, centre, ball_radius);
    if (to_ball < to_sphere) {
        let normal = (start + direction * to_ball - centre) / ball_radius;
        // The normal in the ball's own frame, its signs picking the octant, and the seams where one is small.
        let body = attitude << normal;
        let x = mv.x | body;
        let y = mv.y | body;
        let z = mv.z | body;
        let octant = select(0u, 1u, x >= 0) + select(0u, 2u, y >= 0) + select(0u, 4u, z >= 0);
        let seam = clamp(1 - min(abs(x), min(abs(y), abs(z))) / seam_width, 0.0, 1.0);
        return mix(octants[octant], seams, seam) * (0.47 + 0.53 * max(normal | lamp, 0.0));
    }
    if (to_sphere < far) {
        let point = start + direction * to_sphere;
        return route_paint(point, direction) * (0.58 + 0.42 * max((point / sphere_radius) | lamp, 0.0));
    }
    return background;
}
"""


# --- plumbing -------------------------------------------------------------------------
@cache
def shader() -> Shader:
    return Shader(supersampled() + source, ga, dict(Vector=Vector, Rotor=Rotor))


class SphereScene:
    """The fixed sphere and routes, seen by one camera framing the sphere and the ball along all the fitted
    routes; elevation and azimuth in degrees, the azimuth measured from x."""

    def __init__(self, paths: tuple[Vector, ...], fit_paths: tuple[Vector, ...],
                 sphere_radius: float, ball_radius: float,
                 elevation: float, azimuth: float, size: int) -> None:
        elevation, azimuth = np.radians(elevation), np.radians(azimuth + 90)
        self.camera = (mv.xy * (-azimuth / 2)).exp() * (mv.yz * (-(np.pi / 2 - elevation) / 2)).exp()   # [] Rotor
        # The ball's centres along every fitted route, across and up the view, and the sphere's outline about
        # the middle.
        centres = concatenate(fit_paths) * (sphere_radius + ball_radius)                      # [samples] Vector
        right, up = (((self.camera >> axis) | centres).to_array() for axis in (mv.x, mv.y))
        low = np.minimum([right.min() - ball_radius, up.min() - ball_radius], -sphere_radius)
        high = np.maximum([right.max() + ball_radius, up.max() + ball_radius], sphere_radius)
        middle = (low + high) / 2
        self.middle = self.camera >> (mv.x * middle[0] + mv.y * middle[1])                    # [] Vector
        self.half = np.max(high - low) / 2 * (1 + margin)
        self.size = size
        self.sphere_radius, self.ball_radius = sphere_radius, ball_radius
        # The routes on the sphere: each leg's samples joined in segments, their colours by leg.
        places = [path * sphere_radius for path in paths]                                      # [legs][samples] Vector
        self.segments = dict(start=concatenate([place[:-1] for place in places]),
                             end=concatenate([place[1:] for place in places]),
                             colour=np.concatenate([np.repeat(route_colours[[index]], len(place) - 1, axis=0)
                                                    for index, place in enumerate(places)]))
        self.ends = dict(place=concatenate([place[[0, -1]] for place in places]),
                         colour=np.repeat(route_colours[:len(places)], 2, axis=0))
        self.lamp = self.camera >> (mv.x * -0.40 + mv.y * 0.48 + mv.z * 0.78)

    def draw(self, contact: Vector, attitude: Rotor) -> np.ndarray:
        """The ball touching the sphere along a contact direction, turned by its attitude."""
        pixel = 2 * self.half / self.size
        traced = shader()((self.size, self.size), camera=self.camera, middle=self.middle, back=4 * self.half,
                          half_height=self.half, lamp=self.lamp, sphere_radius=self.sphere_radius,
                          ball_radius=self.ball_radius, contact=contact, attitude=attitude, segments=self.segments,
                          ends=self.ends, line_half_width=line_width / 2 * pixel, dot_radius=dot_radius * pixel,
                          rim_width=pixel)["image"]
        return np.round(np.clip(traced, 0, 1) * 255).astype(np.uint8)


def frames(paths: tuple[Vector, ...], rolling: Iterable[tuple[Vector, Rotor]],
           sphere_radius: float, ball_radius: float, elevation: float, azimuth: float,
           size: int) -> Iterator[np.ndarray]:
    """Render successive contact directions and ball attitudes with one fixed camera."""
    scene = SphereScene(paths, paths, sphere_radius, ball_radius, elevation, azimuth, size)
    for contact, attitude in rolling:
        yield scene.draw(contact, attitude)


def comparison_frames(paths: tuple[Vector, ...], transformed_paths: tuple[Vector, ...],
                      rolling: Iterable[tuple[Vector, Rotor]],
                      transformed_rolling: Iterable[tuple[Vector, Rotor]],
                      sphere_radius: float, ball_radius: float, elevation: float,
                      azimuth: float, size: int) -> Iterator[np.ndarray]:
    """Two rolling routes in identical camera frames, synchronized by their supplied states."""
    fit_paths = paths + transformed_paths
    scene = SphereScene(paths, fit_paths, sphere_radius, ball_radius, elevation, azimuth, size)
    transformed_scene = SphereScene(transformed_paths, fit_paths, sphere_radius, ball_radius,
                                    elevation, azimuth, size)
    for state, transformed_state in zip(rolling, transformed_rolling):
        yield np.concatenate([scene.draw(*state), transformed_scene.draw(*transformed_state)], axis=1)
