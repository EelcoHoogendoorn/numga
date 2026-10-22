"""Drawing for the XPBD chain: every pixel's rays meet the nearest link, a solid ellipsoid placed by its motor,
over a floor that catches the links' shadows.

The links are traced on the GPU by a shader whose geometry is written in numga. Every link is the same solid
about its own origin, so a ray is carried into each link's frame by its motor, `motor << direction`, and meets
the one quadric there; where it hits, the hit's place in the link's frame shows the octant it lies in, and the
tangent plane carried back out, `motor >> link(local)`, lights it.
"""

from __future__ import annotations

from functools import cache

import numpy as np

from examples.mechanics.xpbd.core import Motor
from examples.wgsl import Shader, solids, supersampled
from examples.wgsl.solids import Plane, Point, Quadric, ga, mv, origin

# Two alternating link colours, the pin's colour, the floor's two tiles and the background.
link_colours = np.array([[0.86, 0.42, 0.20], [0.20, 0.52, 0.66]])
pin_colour = np.array([0.25, 0.25, 0.28])
# The darker tone of a link's alternate octants, which shows it turning about its own axes.
octant_contrast = 0.22
# The view: its elevation and azimuth, and its margin around the whole swing and its shadows, as a fraction of
# the larger side; the floor's depth below the swing's lowest point and its reach beyond the swing, and a floor
# tile's side, in link spacings; the pin's radius as a fraction of the spacing, and the pixels across.
elevation, azimuth = np.radians(14.0), np.radians(32.0)
margin = 0.08
floor_drop, floor_reach, tile = 1.0, 12.0, 3.0
pin_radius = 0.18
pixels = 360
lamp = mv.direction([0.35, -0.45, 0.82])
lamp = lamp / (-(lamp.dual() & lamp)).square_root()

source = supersampled(2) + """
// Each link's motor and colour; a link's solid about its own origin; the pin, placed.
var<storage, read> motors: array<Motor>;
var<storage, read> colours: array<vec3<f32>>;
var<uniform> link: Quadric;
var<uniform> pin: Quadric;
var<uniform> pin_colour: vec3<f32>;
var<uniform> octant_contrast: f32;
// The floor: its plane, the square it covers, and its tiles' side.
var<uniform> floor_plane: Plane;
var<uniform> floor_low: vec2<f32>;
var<uniform> floor_high: vec2<f32>;
var<uniform> tile: f32;
// A camera looking along its own -z, its rays parallel, starting `back` behind the scene's middle; the image
// half as high as this.
var<uniform> camera: Motor;
var<uniform> back: f32;
var<uniform> half_height: f32;
var<uniform> lamp: Point;

alias Sample = vec3<f32>;

const origin = mv.w.dual();
const far = 1e6;
const clearance = 1e-6;
const ambient = 0.3;
const gloss = 0.3;
const shininess = 40.0;
const shadow = 0.25;
const background = vec3<f32>(0.97, 0.97, 0.96);
const floor_colours = array<vec3<f32>, 2>(vec3<f32>(0.86, 0.85, 0.82), vec3<f32>(0.76, 0.75, 0.72));

// Where a ray enters a solid quadric, or `far`: along it the quadric is a quadratic in the distance.
fn entry(quadric: Quadric, origin: Point, direction: Point) -> f32 {
    let a = quadric(direction) & direction;
    let b = quadric(direction) & origin;
    let c = quadric(origin) & origin;
    let discriminant = b * b - a * c;
    let distance = (-b - sqrt(max(discriminant, 0.0))) / a;
    return select(far, distance, discriminant >= 0 && a > 0 && distance > clearance);
}

// The nearest body a ray meets: the pin, numbered after the links, or a link, met in its own frame.
struct Hit {
    distance: f32,
    body: u32,
}

fn nearest(origin: Point, direction: Point) -> Hit {
    var hit = Hit(entry(pin, origin, direction), arrayLength(&motors));
    for (var body = 0u; body < arrayLength(&motors); body++) {
        let distance = entry(link, motors[body] << origin, motors[body] << direction);
        if (distance < hit.distance) {
            hit = Hit(distance, body);
        }
    }
    return hit;
}

// The lamp's light on a surface with this tangent plane, shadowed by the bodies, with its glint.
fn lit(colour: vec3<f32>, tangent: Plane, point: Point, direction: Point) -> vec3<f32> {
    let normal = tangent / (tangent | tangent).square_root();
    let shaded = select(1.0, shadow, nearest(point + lamp * clearance, lamp).distance < far);
    let light = max(normal & lamp, 0.0) * shaded;
    // The view ray reflected in the tangent plane, against the lamp: `-d.dual()` is the plane square to d.
    let glint = pow(max(-((normal >> direction).dual() & lamp), 0.0), shininess) * select(0.0, 1.0, light > 0);
    return colour * (ambient + (1 - ambient) * light) + gloss * glint;
}

fn sample(place: vec2<f32>) -> Sample {
    let across = screen(place) * half_height;
    let start = camera >> (origin + mv.z.dual() * back + mv.x.dual() * across.x + mv.y.dual() * across.y);
    let direction = camera >> -mv.z.dual();
    let hit = nearest(start, direction);
    let to_floor = -(floor_plane & start) / (floor_plane & direction);
    let on_floor = start + direction * to_floor;
    let floor_place = vec2<f32>(mv.x & on_floor, mv.y & on_floor);
    let floor_seen = all(floor_place >= floor_low) && all(floor_place <= floor_high) && to_floor < hit.distance;
    if (floor_seen) {
        let parity = u32(abs(floor(floor_place.x / tile) + floor(floor_place.y / tile))) % 2;
        return lit(floor_colours[parity], floor_plane, on_floor, direction);
    }
    if (hit.distance >= far) {
        return background;
    }
    let point = start + direction * hit.distance;
    if (hit.body == arrayLength(&motors)) {
        return lit(pin_colour, pin(point), point, direction);
    }
    // The hit in the link's own frame: its octant, and its tangent plane there, carried back out.
    let motor = motors[hit.body];
    let local = motor << point;
    let octant = (select(0, 1, (mv.x & local) > 0) + select(0, 1, (mv.y & local) > 0) + select(0, 1, (mv.z & local) > 0)) % 2;
    let colour = colours[hit.body] * (1 - octant_contrast * f32(octant));
    return lit(colour, motor >> link(local), point, direction);
}
"""


# --- plumbing -------------------------------------------------------------------------
@cache
def shader() -> Shader:
    return Shader(source, ga, dict(Point=Point, Plane=Plane, Motor=Motor, Quadric=Quadric))


def framing(points: Point, floor: Plane, reach: float) -> solids.View:
    """A view framing points and their shadows on the floor, cast along the lamp, with `reach` to spare around each."""
    # Where the line through each point and the lamp meets the floor.
    shadows = (points & lamp) ^ floor                                                       # [points] Point
    return solids.view([points, shadows], reach, elevation, azimuth, margin, pixels)


def animate_chain(motors: Motor, semi_axes: np.ndarray, spacing: float) -> list[np.ndarray]:
    """One frame per step: the links swinging from the pin, over a tiled floor below the swing's lowest point,
    the view framed by the whole swing and its shadows."""
    centres = (motors >> origin).reshape(-1)                                                  # [steps * bodies] Point
    place = np.stack([(axis & centres).to_array() for axis in (mv.x, mv.y, mv.z)], axis=-1)  # [steps * bodies, 3]
    low, high = place.min(axis=0), place.max(axis=0)
    floor = mv.z - mv.w * (low[2] - floor_drop * spacing)                                   # [] Plane
    view = framing(centres, floor, semi_axes.max())
    # The pin sits at the joint of the fixed body and the first link.
    pin = solids.ball((motors[0, 0] >> origin) + mv.x.dual() * (spacing / 2), pin_radius * spacing)   # [] Quadric
    n_links = motors.shape[1] - 1
    return [np.round(np.clip(shader()(view.shape, motors=links, colours=link_colours[np.arange(n_links) % 2],
                                      link=solids.ellipsoid(semi_axes), pin=pin, pin_colour=pin_colour,
                                      octant_contrast=octant_contrast, floor_plane=floor,
                                      floor_low=low[:2] - floor_reach * spacing, floor_high=high[:2] + floor_reach * spacing,
                                      tile=tile * spacing, camera=view.camera, back=4 * view.half.max(),
                                      half_height=view.half[1], lamp=lamp)["image"],
                             0, 1) * 255).astype(np.uint8)
            for links in motors[:, 1:]]
