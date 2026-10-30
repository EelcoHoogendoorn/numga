"""Drawing for the XPBD chain: every pixel's rays meet the nearest link, a solid ellipsoid placed by its motor,
over a floor that catches the links' shadows.

The links are traced on the GPU by a shader whose geometry is written in numga. Every link is the same solid
about its own origin, so a ray is carried into each link's frame by its motor, `motor << direction`, and meets
the one quadric there; where it hits, the hit's place in the link's frame shows the octant it lies in, and the
tangent plane carried back out, `motor >> link(local)`, lights it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import cache
from itertools import chain

import numpy as np

from numga import concatenate
from examples.mechanics.xpbd.core import Motor
from examples.util.wgsl import Shader, solids, supersampled
from examples.util.wgsl.solids import Plane, Point, Quadric, axes, ga, lamp_shading, mv, origin, quadric_crossings

# Two alternating link colours, the pin's colour, the floor's two tiles and the background.
link_colours = np.array([[0.86, 0.42, 0.20], [0.20, 0.52, 0.66]])
pin_colour = np.array([0.25, 0.25, 0.28])
floor_light, floor_dark = np.array([0.86, 0.85, 0.82]), np.array([0.76, 0.75, 0.72])
background = np.array([0.97, 0.97, 0.96])
# The darker tone of a link's alternate octants, which shows it turning about its own axes.
octant_contrast = 0.22
# The view: its elevation and azimuth, and its margin around where the swing can reach and its shadows, as a
# fraction of the larger side; the floor's depth below the lowest the chain can hang and its reach beyond the swing,
# and a floor tile's side, in link spacings; the pin's radius as a fraction of the spacing, and the pixels across.
elevation, azimuth = np.radians(14.0), np.radians(32.0)
margin = 0.08
floor_drop, floor_reach, tile = 1.0, 12.0, 3.0
pin_radius = 0.18
pixels = 360
# The lamp's direction at unit length; the light that falls everywhere, and the glint's share and sharpness, with the
# lamp as `lamp_shading` takes them; and how much of the lamp's light falls in a shadow.
toward_lamp = mv.direction([0.35, -0.45, 0.82])                                # [] Point-
lamp = toward_lamp / toward_lamp.dual().norm()                                 # [] Point-
lighting = dict(lamp=lamp, ambient=0.3, fill_tilt=0.0, headlight=0.0, gloss=0.3, shininess=40.0)
shadow = 0.25

source = """
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
// half as high as this. Its frame, `axes`, is given from Python.
var<uniform> camera: Motor;
var<uniform> back: f32;
var<uniform> half_height: f32;

alias Sample = vec3<f32>;

const origin = mv.w.dual();
const clearance = 1e-6;

// The nearest body a ray meets: the pin, numbered after the links, or a link, met in its own frame.
struct Hit {
    distance: f32,
    body: u32,
}

// Where a ray enters a solid quadric past its start, or `far`.
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

// A colour on a surface with this tangent plane, lit, with its glint; where the bodies stand between it and the lamp,
// only `shadow` of the lamp's light and glint reaches it, and all the light that falls everywhere.
fn lit_or_shadowed(colour: vec3<f32>, tangent: Plane, point: Point, direction: Point) -> vec3<f32> {
    let normal = facing(tangent, direction);
    let lamp_share = select(1.0, shadow, nearest(point + lamp * clearance, lamp).distance < far);
    return colour * mix(ambient, brightness(normal, direction), lamp_share) + gloss * lamp_share * glint(normal, direction);
}

fn sample(place: vec2<f32>) -> Sample {
    // The ray's start in the camera's frame: across and up the image, and `back` toward the viewer.
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
    // The hit in the link's own frame: its octant, and its tangent plane there, carried back out.
    let motor = motors[hit.body];
    let local = motor << point;
    let octant = (select(0, 1, (mv.x & local) > 0) + select(0, 1, (mv.y & local) > 0) + select(0, 1, (mv.z & local) > 0)) % 2;
    let colour = colours[hit.body] * (1 - octant_contrast * f32(octant));
    return lit_or_shadowed(colour, motor >> link(local), point, direction);
}
"""


# --- plumbing -------------------------------------------------------------------------
@cache
def shader() -> Shader:
    return Shader(supersampled() + quadric_crossings() + lamp_shading() + source, ga,
                  dict(Point=Point, Plane=Plane, Motor=Motor, Quadric=Quadric),
                  dict(axes=axes, shadow=shadow, floor_light=floor_light, floor_dark=floor_dark, background=background))


def framing(points: Point, floor: Plane, clearance: float) -> solids.View:
    """A view framing points and their shadows on the floor, cast along the lamp, with `clearance` to spare around
    each."""
    # Where the line through each point and the lamp meets the floor.
    shadows = (points & lamp) ^ floor                                                       # [points] Point-
    return solids.view([points, shadows], clearance, elevation, azimuth, margin, pixels)


def animate_chain(motors: Iterable[Motor], semi_axes: np.ndarray, spacing: float) -> Iterator[np.ndarray]:
    """One frame per step of the bodies' motors `[bodies]`: the links swinging from the pin, over a tiled floor below
    the lowest the chain can hang, the view framed by where the swing can reach and its shadows: the chain as it
    starts, the same half a turn about the vertical through the pin, and hung straight down from the pin."""
    motors = iter(motors)
    first = next(motors)
    centres = first >> origin                                                                # [bodies] Point-
    # How far the chain reaches along itself from the pin: the lengths of the offsets between neighbouring links.
    length = (centres[1:] - centres[:-1]).dual().norm().sum()                                # [] Scalar+
    extremes = concatenate([centres, (mv.xy * (np.pi / 2)).exp() >> centres, (origin - mv.z.dual() * length)[None]])
    readings = axes[:, None] & extremes
    low = readings.min(axis=-1)
    high = readings.max(axis=-1)
    floor = mv.z - mv.w * (low[2] - floor_drop * spacing)                                   # [] Plane+
    view = framing(extremes, floor, semi_axes.max())
    # The pin sits at the joint of the fixed body and the first link.
    pin = solids.ball(centres[0] + mv.x.dual() * (spacing / 2), pin_radius * spacing)        # [] Plane+ <- Point-
    n_links = centres.shape[0] - 1
    for bodies in chain([first], motors):
        yield np.round(np.clip(shader()(view.shape, motors=bodies[1:], colours=link_colours[np.arange(n_links) % 2],
                                        link=solids.ellipsoid(semi_axes), pin=pin, pin_colour=pin_colour,
                                        octant_contrast=octant_contrast, floor_plane=floor,
                                        floor_low=(low[:2] - floor_reach * spacing).to_array(),
                                        floor_high=(high[:2] + floor_reach * spacing).to_array(), tile=tile * spacing,
                                        camera=view.camera, back=4 * view.half.max(), half_height=view.half[1],
                                        **lighting)["image"], 0, 1) * 255).astype(np.uint8)
