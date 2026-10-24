"""A rattleback as built, a wooden boat with steel balls pressed into its sides, above its ground, ray traced on the
GPU.

The drawing traces the quadrics the mass is read from: each part an intersection of quadrics, `quadric(point) & point`
negative inside, a ray's first meeting with one part's quadric that lies inside all its others is what it sees, and
the quadric's plane there is the tangent plane. Each quadric colours the surface it bounds: the painted underside,
the bare wood of the deck and the cuts; the steel mirrors what lies around it.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import cache
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

from numga import concatenate
from numga.algebras import PGA3D
from examples import instantiate
from examples.util.wgsl import Shader, image_figure, solids, supersampled
from examples.util.wgsl.solids import lamp_shading, quadric_crossings

core = instantiate("examples.mechanics.rattleback.core", PGA3D)
mv = core.mv

n_pixels = 500
# The views, elevation and azimuth in radians, azimuth zero looking along +y; the margin around the subject as a
# share of its larger side.
view = np.radians([25.0, 0.0])
setup_view = np.radians([58.0, 22.0])
margin = 0.1
# The lamp's direction at unit length; the light that falls everywhere and the glint's sharpness, with the lamp as
# `lamp_shading` takes them, each finish carrying its own glint.
toward_lamp = mv.direction([-0.3, -0.6, 0.85])                                 # [] Point
lamp = toward_lamp / toward_lamp.dual().norm()                                 # [] Point
lighting = dict(lamp=lamp, ambient=0.38, fill_tilt=0.0, headlight=0.0, gloss=0.0, shininess=32.0)
# The surfaces' colours; the glint of the satin paint, the bare wood having none; and how much of what lies around
# it the steel mirrors.
paint = np.array([0.19, 0.52, 0.62])
deck_wood = np.array([0.91, 0.81, 0.60])
cut_wood = np.array([0.80, 0.63, 0.42])
steel = np.array([0.68, 0.68, 0.70])
paint_gloss = 0.08
steel_mirroring = 1.0
# The ground: tiles of warm grey stone, each a little lighter or darker than its neighbours and softly mottled, honed
# smooth enough to mirror a little of what stands on it, the more the more glancing the view.
stone = np.array([0.47, 0.44, 0.40])
tile_variation = 0.06
mottling = 0.07
mottle_scale = 1.5
stone_reflectance = 0.04
# The sky as a mirror shows it: bright at the horizon, deepening overhead, with the lamp a bright disc in it; and how
# much of the lamp's light the body's shadow takes.
horizon = np.array([0.86, 0.86, 0.86])
zenith = np.array([0.38, 0.39, 0.41])
lamp_size = 0.995
lamp_brightness = 5.0
shadow = 0.22
# The tiles' size, and the joints between them: their width and how much darker they are.
tile_size = 2.0
joint_width = 0.02
joint_darkness = 0.45
# The points of the body the framing follows; the turns of its first placement about the vertical it frames, every
# way the body may come to face as it spins, and the room around them for the body to wander as it rolls.
n_framing = 400
n_turns = 16
drift = 0.3

source = """
// A part: the intersection of its quadrics, `quadrics[first]` up to `quadrics[end]`; each quadric's finish, the
// colour of the surface it bounds and how much that mirrors.
struct Part {
    first: u32,
    end: u32,
}

struct Finish {
    colour: vec3<f32>,
    gloss: f32,
    mirroring: f32,
}

var<storage, read> parts: array<Part>;
var<storage, read> quadrics: array<Quadric>;
var<storage, read> finishes: array<Finish>;
// A camera looking along its own -z, its rays parallel, starting `back` behind the middle of the view; the image
// half as high as this. Its frame, `axes`, is given from Python.
var<uniform> camera: Motor;
var<uniform> back: f32;
var<uniform> half_height: f32;

alias Sample = vec3<f32>;

const origin = mv.w.dual();
const ground_plane = mv.z;
// How far a point may lie outside a quadric and still count as inside, for points on two surfaces; and how far
// past its start a ray leaving a surface looks.
const tolerance = 1e-4;
const clearance = 1e-3;

struct Hit {
    distance: f32,
    quadric: u32,
    tangent: Cut,
}

fn inside_others(part: Part, own: u32, point: Point) -> bool {
    for (var index = part.first; index < part.end; index++) {
        if (index != own && (quadrics[index](point) & point) > tolerance) {
            return false;
        }
    }
    return true;
}

// The first point past `nearest` along a ray on a part's surface: on one of its quadrics, inside all the others.
fn first_hit(start: Point, direction: Point, nearest: f32) -> Hit {
    var hit = Hit(far, 0u, mv.z);
    for (var index = 0u; index < arrayLength(&parts); index++) {
        let part = parts[index];
        for (var own = part.first; own < part.end; own++) {
            let quadric = quadrics[own];
            let along = along_ray(quadric, start, direction);
            if (along.b * along.b - along.a * along.c < 0) {
                continue;
            }
            let roots = crossings(along);
            for (var which = 0u; which < 2u; which++) {
                let distance = roots[which];
                let point = start + direction * distance;
                if (distance > nearest && distance < hit.distance && inside_others(part, own, point)) {
                    // The quadric's plane at a point of its surface is the tangent plane there.
                    hit = Hit(distance, own, quadric(point));
                }
            }
        }
    }
    return hit;
}

// A value for each cell of a lattice, and a smooth blend of them between the cells.
fn hash(cell: vec2<f32>) -> f32 {
    return fract(sin(dot(cell, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn noise(spot: vec2<f32>) -> f32 {
    let cell = floor(spot);
    let within = fract(spot);
    let blend = within * within * (3 - 2 * within);
    return mix(mix(hash(cell), hash(cell + vec2<f32>(1.0, 0.0)), blend.x),
               mix(hash(cell + vec2<f32>(0.0, 1.0)), hash(cell + vec2<f32>(1.0, 1.0)), blend.x), blend.y);
}

// The ground where a ray meets it, tiled, mottled and darker where the body stands between it and the lamp; the
// sky, with the lamp in it, where the ray rises.
fn surroundings(start: Point, direction: Point) -> vec3<f32> {
    let to_ground = -(ground_plane & start) / (ground_plane & direction);
    let on_ground = start + direction * to_ground;
    let spot = vec2<f32>(mv.x & on_ground, mv.y & on_ground) / tile_size;
    let off_joint = min(abs(spot.x - round(spot.x)), abs(spot.y - round(spot.y))) * tile_size;
    let joint = clamp(1 - off_joint / joint_width, 0.0, 1.0);
    let tone = 1 + tile_variation * (2 * hash(floor(spot)) - 1);
    let mottle = 1 + mottling * (noise(spot * mottle_scale) + noise(spot * mottle_scale * 2.3) / 2 - 0.75);
    let shaded = select(1.0, 1 - shadow, first_hit(on_ground, lamp, clearance).distance < far);
    let floor = stone * tone * mottle * (1 - joint_darkness * joint) * shaded;
    let rise = max(mv.z & direction, 0.0);
    let toward_lamp = -(direction.dual() & lamp);
    let heaven = mix(horizon, zenith, sqrt(rise)) + lamp_brightness * smoothstep(lamp_size, 1.0, toward_lamp);
    // Every ray starts on or above the ground, so it meets the ground only heading down.
    return select(heaven, floor, (ground_plane & direction) < 0);
}

// A surface lit by the lamp, with its finish's glint, unless the body stands between them: then none of the lamp's
// light and glint reaches it, only the light that falls everywhere.
fn lit_or_shadowed(hit: Hit, start: Point, direction: Point) -> vec3<f32> {
    let normal = facing(hit.tangent, direction);
    let lamp_share = select(1.0, 0.0, first_hit(start + direction * hit.distance, lamp, clearance).distance < far);
    let finish = finishes[hit.quadric];
    return finish.colour * mix(ambient, brightness(normal, direction), lamp_share)
        + finish.gloss * lamp_share * glint(normal, direction);
}

// What a ray sees, without looking into mirrors.
fn seen(start: Point, direction: Point) -> vec3<f32> {
    let hit = first_hit(start, direction, clearance);
    if (hit.distance < far) {
        return lit_or_shadowed(hit, start, direction);
    }
    return surroundings(start, direction);
}

fn sample(place: vec2<f32>) -> Sample {
    // The ray's start in the camera's frame: across and up the image, and `back` toward the viewer.
    let across = screen(place) * half_height;
    let start = camera >> (origin + (axes.dual() * vec3<f32>(across, back)).sum());
    let direction = camera >> -mv.z.dual();
    let hit = first_hit(start, direction, 0.0);
    let to_ground = -(ground_plane & start) / (ground_plane & direction);
    if (hit.distance >= far || hit.distance > to_ground) {
        // The honed stone mirrors a little of what the ray, reflected in the ground, goes on to see.
        let glancing = pow(1 - abs(ground_plane & direction), 5.0);
        let reflectance = stone_reflectance + (1 - stone_reflectance) * glancing;
        return mix(surroundings(start, direction), seen(start + direction * to_ground, ground_plane >> direction), reflectance);
    }
    // A mirroring surface shows what the ray, reflected in its tangent plane, goes on to see, tinted by its own
    // colour as a metal is, and less tinted the more glancing the ray.
    let point = start + direction * hit.distance;
    let normal = hit.tangent.normalized();
    let finish = finishes[hit.quadric];
    let glancing = pow(1 - abs(normal & direction), 5.0);
    let reflectance = finish.colour + (1 - finish.colour) * glancing;
    return mix(lit_or_shadowed(hit, start, direction), reflectance * seen(point, normal >> direction), finish.mirroring);
}
"""


# --- plumbing -------------------------------------------------------------------------
@cache
def shader() -> Shader:
    return Shader(supersampled() + quadric_crossings() + lamp_shading() + source, PGA3D,
                  dict(Point=core.Point, Plane=core.Cut, Cut=core.Cut, Motor=core.Motor, Quadric=core.Quadric),
                  dict(axes=solids.axes, stone=stone, tile_variation=tile_variation, mottling=mottling,
                       mottle_scale=mottle_scale, stone_reflectance=stone_reflectance, horizon=horizon, zenith=zenith,
                       lamp_size=lamp_size, lamp_brightness=lamp_brightness, shadow=shadow, tile_size=tile_size,
                       joint_width=joint_width, joint_darkness=joint_darkness))


def scene(built: core.Body, motor: core.Motor) -> dict:
    """The parts placed by a motor, their quadrics one after another, and each quadric's finish: the wood first,
    then each ball."""
    wood = built.wood()                                                         # [quadrics] Quadric
    quadrics = concatenate([wood, built.balls])                                 # [quadrics] Quadric
    n_wood, n_balls = wood.shape[0], built.balls.shape[0]
    ends = np.concatenate([[n_wood], n_wood + 1 + np.arange(n_balls)])
    starts = np.concatenate([[0], ends[:-1]])
    # The underside painted, the deck and the cuts bare wood; the pockets lie under the balls.
    colours = np.concatenate([[paint, deck_wood], np.repeat([cut_wood], n_wood - 2, axis=0),
                              np.repeat([steel], n_balls, axis=0)])
    gloss = np.concatenate([[paint_gloss], np.zeros(n_wood - 1 + n_balls)])
    mirroring = np.concatenate([np.zeros(n_wood), np.full(n_balls, steel_mirroring)])
    return dict(parts=dict(first=starts, end=ends), quadrics=solids.placed(quadrics, motor),
                finishes=dict(colour=colours, gloss=gloss, mirroring=mirroring))


def framing(motors: core.Motor, samples: core.Point, elevation: float, azimuth: float, reach: float,
            pixels: int) -> solids.View:
    """A view `pixels` across framing the body over every placement and its shadow on the ground, cast along the
    lamp, with `reach` to spare around each, the height rounded to whole pixels."""
    body = motors[..., None] >> samples[::max(1, samples.shape[0] // n_framing)]   # [..., points] Point
    shadows = (body & lamp) ^ mv.z                                                # [..., points] Point
    framed = solids.view([body, shadows], reach, elevation, azimuth, margin, pixels)
    n_rows = framed.shape[0]
    return solids.View(framed.camera, np.array([framed.half[0], framed.half[0] * n_rows / pixels]), pixels)


def picture(built: core.Body, motor: core.Motor, seen: solids.View) -> np.ndarray:
    """The body and its shadow, `[rows, columns, 3]` bytes."""
    traced = shader()(seen.shape, **scene(built, motor), camera=seen.camera, back=4 * seen.half.max(),
                      half_height=seen.half[1], **lighting)["image"]
    return np.round(np.clip(traced, 0, 1) * 255).astype(np.uint8)


def turned(motor: core.Motor) -> core.Motor:
    """A placement turned about the vertical line through the centre of mass, a whole turn in `n_turns` steps
    `[turns]`."""
    # The vertical line through the centre: the join of the centre with the upward direction, made extrinsic.
    vertical = ((motor >> mv.w.dual()) & mv.z.dual()) * solids.flip                   # [] Line
    return (vertical.normalized() * (np.pi * np.arange(n_turns) / n_turns)).exp() * motor


def frames(motors: Iterable[core.Motor], built: core.Body, samples: core.Point) -> Iterator[np.ndarray]:
    """The rolling body, as `sized_frames` draws it `n_pixels` across."""
    return sized_frames(motors, built, samples, n_pixels)


def sized_frames(motors: Iterable[core.Motor], built: core.Body, samples: core.Point,
                 pixels: int) -> Iterator[np.ndarray]:
    """The rolling body, `pixels` across, viewed from a fixed camera framing its first placement turned every way
    about the vertical, with room for it to wander."""
    motors = iter(motors)
    first = next(motors)
    seen = framing(turned(first), samples, *view, drift, pixels)
    for motor in chain([first], motors):
        yield picture(built, motor, seen)


def draw_setup(built: core.Body, samples: core.Point, motor: core.Motor) -> plt.Figure:
    """The body at rest, seen from above, its steel balls on opposite corners."""
    return image_figure(picture(built, motor, framing(motor, samples, *setup_view, 0.0, n_pixels)))
