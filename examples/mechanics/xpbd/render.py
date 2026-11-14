"""Ray-traced rendering for the XPBD chain of solid ellipsoidal links over a shadowed floor.

The links are traced on the GPU by a shader whose geometry is written in numga. Every link shares one reference quadric in its own body frame, so each camera ray pulls into a link's frame via `motor << direction` and intersects that quadric there; at the hit point, the body-frame coordinates mark the octant tone so axial spin stays visible, and pushing the tangent plane forward via `motor >> link(local)` shades the surface.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import cache
from itertools import chain

import numpy as np

from numga import concatenate
from numga.render import solids
from numga.render.solids import (Motor, Plane, Point, Quadric, axes, ga, lamp_shading, mv, origin, quadric_crossings,
                                 seen_lamp, steps, tangent_seen)
from numga.wgsl import Shader, supersampled

# Alternating link colours, pin colour, checkerboard floor tones, and background tone.
link_colours = np.array([[0.86, 0.42, 0.20], [0.20, 0.52, 0.66]])
pin_colour = np.array([0.25, 0.25, 0.28])
floor_light, floor_dark = np.array([0.86, 0.85, 0.82]), np.array([0.76, 0.75, 0.72])
background = np.array([0.97, 0.97, 0.96])
# Tone contrast on alternate body-frame octants so axial twisting is visible.
octant_contrast = 0.22
# Camera angles, frame margin, floor placement and tile size (in link spacings), pin radius fraction, and image width.
elevation, azimuth = np.radians(14.0), np.radians(32.0)
margin = 0.08
floor_drop, floor_reach, tile = 1.0, 12.0, 3.0
pin_radius = 0.18
pixels = 360
# Unit lamp direction, Blinn-Phong lighting weights, and shadow attenuation.
toward_lamp = mv.direction([0.35, -0.45, 0.82])                                 # [] Point-
lamp = toward_lamp / toward_lamp.antinorm()                                     # [] Point-
lighting = dict(ambient=0.3, fill_tilt=0.0, headlight=0.0, gloss=0.3, shininess=40.0)
shadow = 0.25

source = '''
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
// Orthographic camera looking along its own -z, starting `back` behind the scene centre.
var<uniform> camera: Motor;
// Unit lamp direction in the scene along which shadow rays march.
var<uniform> lamp_in_scene: Point;
var<uniform> back: f32;
var<uniform> half_height: f32;

alias Sample = vec3<f32>;

const origin = mv.w.dual();
const clearance = 1e-6;

// Nearest body met along a ray: either a link index or `arrayLength(&motors)` for the pin.
struct Hit {
    distance: f32,
    body: u32,
}

fn entered(quadric: Quadric, start: Point, direction: Point) -> f32 {
    let distance = entry(along_ray(quadric, start, direction));
    return select(far, distance, distance > clearance);
}

fn nearest(start: Point, direction: Point) -> Hit {
    var hit = Hit(entered(pin, start, direction), arrayLength(&motors));
    for (var body = 0u; body < arrayLength(&motors); body++) {
        let distance = entered(link, motors[body] << start, motors[body] << direction);
        if (distance < hit.distance) {
            hit = Hit(distance, body);
        }
    }
    return hit;
}

fn lit_or_shadowed(colour: vec3<f32>, tangent: Plane, point: Point, direction: Point) -> vec3<f32> {
    let normal = turned(slopes(tangent));
    let lamp_share = select(1.0, shadow, nearest(point + lamp_in_scene * clearance, lamp_in_scene).distance < far);
    return colour * mix(ambient, brightness(normal), lamp_share) + gloss * lamp_share * glint(normal);
}

fn sample(place: vec2<f32>) -> Sample {
    let across = screen(place) * half_height;
    let start = camera >> (origin + (axes.dual() * vec3<f32>(across, back)).sum());
    let direction = camera >> -mv.z.dual();
    let hit = nearest(start, direction);
    let to_floor = -(floor_plane & start) / (floor_plane & direction);
    let on_floor = start + direction * to_floor;
    let floor_place = vec2<f32>(mv.x & on_floor, mv.y & on_floor);
    let floor_seen = all(floor_place >= floor_low) && all(floor_place <= floor_high) && to_floor < hit.distance;
    if (floor_seen) {
        let parity = u32(abs(floor(floor_place.x / tile) + floor(floor_place.y / tile))) % 2;
        return lit_or_shadowed(select(floor_dark, floor_light, parity == 0), floor_plane, on_floor, direction);
    }
    if (hit.distance >= far) {
        return background;
    }
    let point = start + direction * hit.distance;
    if (hit.body == arrayLength(&motors)) {
        return lit_or_shadowed(pin_colour, pin(point), point, direction);
    }
    let motor = motors[hit.body];
    let local = motor << point;
    let octant = (select(0, 1, (mv.x & local) > 0) + select(0, 1, (mv.y & local) > 0) + select(0, 1, (mv.z & local) > 0)) % 2;
    let colour = colours[hit.body] * (1 - octant_contrast * f32(octant));
    return lit_or_shadowed(colour, motor >> link(local), point, direction);
}
'''


# --- plumbing -------------------------------------------------------------------------
@cache
def shader() -> Shader:
    """Compiles and caches the supersampled WGSL ray-tracing shader for the chain scene."""
    return Shader(supersampled() + quadric_crossings() + lamp_shading() + tangent_seen() + source, ga,
                  dict(Point=Point, Plane=Plane, Tangent=Plane, Motor=Motor, Quadric=Quadric),
                  dict(axes=axes, shadow=shadow, floor_light=floor_light, floor_dark=floor_dark, background=background))


def framing(points: Point, floor: Plane, clearance: float) -> solids.View:
    """Orthographic camera view frames `points` (`[...] Point-`) together with their shadows cast onto `floor`."""
    shadows = (points & lamp) ^ floor                                           # [...] Point-
    return solids.view([points, shadows], clearance, elevation, azimuth, margin, pixels)


def animate_chain(semi_axes: np.ndarray, spacing: float, motors: Iterable) -> Iterator[np.ndarray]:
    """Yields one RGB frame per body-motor state (`[...] Rotor+`) for the swinging chain over a tiled floor."""
    stream = (solids.context.include_from_algebra(state) for state in motors)
    first = next(stream)                                                        # [...] Rotor+
    centres = first >> origin                                                   # [...] Point-
    # Total chain reach from the pin equals the sum of initial inter-centre link offsets.
    reach = (centres[1:] - centres[:-1]).antinorm().sum()                       # [] Scalar+
    turned_centres = (mv.xy * (np.pi / 2)).exp() >> centres                     # [...] Point-
    extremes = concatenate([centres, turned_centres, (origin - mv.z.dual() * reach)[None]])   # [...] Point-
    readings = axes[:, None] & extremes                                         # [3, ...] Scalar+
    low = readings.min(axis=-1)                                                 # [3] Scalar+
    high = readings.max(axis=-1)                                                # [3] Scalar+
    floor = mv.z - mv.w * (low[2] - floor_drop * spacing)                       # [] Plane+
    view = framing(extremes, floor, float(semi_axes.max()))
    # Place the spherical pin at the joint joining the fixed body 0 and the first moving link.
    pin = solids.ball(centres[0] + mv.x.dual() * (spacing / 2), pin_radius * spacing)   # [] Plane+ <- Point-
    n_links = centres.shape[0] - 1
    for bodies in chain([first], stream):
        yield np.round(np.clip(shader()(view.shape, motors=bodies[1:], colours=link_colours[np.arange(n_links) % 2],
                                        link=solids.ellipsoid(semi_axes), pin=pin, pin_colour=pin_colour,
                                        octant_contrast=octant_contrast, floor_plane=floor,
                                        floor_low=(low[:2] - floor_reach * spacing).to_array(),
                                        floor_high=(high[:2] + floor_reach * spacing).to_array(), tile=tile * spacing,
                                        camera=view.camera, back=4 * view.half.max(), half_height=view.half[1],
                                        lamp=seen_lamp(view, lamp), lamp_in_scene=lamp, **lighting, **steps(view))["image"],
                               0, 1) * 255).astype(np.uint8)
