"""GPU ray tracer for a rattleback hull of wood and inset steel balls rolling above a tiled stone table.

Each solid part is an intersection of quadrics whose pairing `quadric(point) & point` is negative inside and zero on
the boundary. A ray hits the nearest boundary point that lies inside every sibling quadric of that part, where
`quadric(point)` supplies the outward tangent plane. Each quadric carries its own surface finish for the painted hull,
the bare-wood deck and chamfers, and the mirroring steel balls.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import cache
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

from numga import concatenate
from numga.render import solids
from numga.render.solids import lamp_shading, quadric_regions
from numga.wgsl import Shader, supersampled
from examples.util.panels import image_figure

ga = solids.ga
mv = solids.mv
origin = solids.origin                                                # [] Point-

Point = solids.Point
Plane = solids.Plane
Line = solids.Line
Spin = ga.gatype.bivector().derive.nondegenerate
Motor = solids.Motor
Quadric = solids.Quadric

n_default_pixels = 300
camera_angles = np.radians([25.0, 0.0])
setup_angles = np.radians([58.0, 22.0])
margin = 0.1

lamp_offset = mv.direction([-0.3, -0.6, 0.85])                        # [] Point-
lamp = lamp_offset / lamp_offset.dual().norm()                        # [] Point-
lighting = dict(ambient=0.38, fill_tilt=0.0, headlight=0.0, gloss=0.0, shininess=32.0)

paint = np.array([0.19, 0.52, 0.62])
deck_wood = np.array([0.91, 0.81, 0.60])
cut_wood = np.array([0.80, 0.63, 0.42])
steel = np.array([0.68, 0.68, 0.70])
paint_gloss = 0.08
steel_mirroring = 1.0

stone = np.array([0.47, 0.44, 0.40])
tile_variation = 0.06
mottling = 0.07
mottle_scale = 1.5
stone_reflectance = 0.04

horizon = np.array([0.86, 0.86, 0.86])
zenith = np.array([0.38, 0.39, 0.41])
lamp_size = 0.995
lamp_brightness = 5.0
shadow = 0.22

tile_size = 2.0
joint_width = 0.02
joint_darkness = 0.45

n_framing_samples = 400
n_turns = 16
drift = 0.3


# --- plumbing -------------------------------------------------------------------------
@cache
def shader() -> Shader:
    """The cached shader compiles the supersampled WGSL ray tracer for quadric intersections over a reflective stone floor."""
    wgsl_source = """
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
var<storage, read> finishes: array<Finish>;
var<uniform> camera: Motor;
var<uniform> lamp_in_scene: Point;
var<uniform> back: f32;
var<uniform> half_height: f32;

alias Sample = vec3<f32>;

const tolerance = 1e-4;
const clearance = 1e-3;

fn first_hit(start: Point, direction: Point, nearest: f32) -> SurfaceHit {
    var hit = SurfaceHit(far, 0u, ground_plane);
    for (var index = 0u; index < arrayLength(&parts); index++) {
        let part = parts[index];
        let surface = region_hit(part.first, part.end, start, direction, nearest, hit.distance, tolerance);
        if (surface.distance < hit.distance) {
            hit = surface;
        }
    }
    return hit;
}

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

fn surroundings(start: Point, direction: Point) -> vec3<f32> {
    let to_ground = -(ground_plane & start) / (ground_plane & direction);
    let on_ground = start + direction * to_ground;
    let spot = vec2<f32>(plane_x & on_ground, plane_y & on_ground) / tile_size;
    let off_joint = min(abs(spot.x - round(spot.x)), abs(spot.y - round(spot.y))) * tile_size;
    let joint = clamp(1 - off_joint / joint_width, 0.0, 1.0);
    let tone = 1 + tile_variation * (2 * hash(floor(spot)) - 1);
    let mottle = 1 + mottling * (noise(spot * mottle_scale) + noise(spot * mottle_scale * 2.3) / 2 - 0.75);
    let shaded = select(1.0, 1 - shadow, first_hit(on_ground, lamp_in_scene, clearance).distance < far);
    let floor_colour = stone * tone * mottle * (1 - joint_darkness * joint) * shaded;
    let rise = max(ground_plane & direction, 0.0);
    let lamp_alignment = -(direction.dual() & lamp_in_scene);
    let sky_colour = mix(horizon, zenith, sqrt(rise)) + lamp_brightness * smoothstep(lamp_size, 1.0, lamp_alignment);
    return select(sky_colour, floor_colour, (ground_plane & direction) < 0);
}

fn lit_or_shadowed(hit: SurfaceHit, start: Point, direction: Point) -> vec3<f32> {
    let normal = turned(slopes(hit.tangent));
    let lamp_share = select(1.0, 0.0, first_hit(start + direction * hit.distance, lamp_in_scene, clearance).distance < far);
    let finish = finishes[hit.quadric];
    return finish.colour * mix(ambient, brightness(normal), lamp_share) + finish.gloss * lamp_share * glint(normal);
}

fn seen(start: Point, direction: Point) -> vec3<f32> {
    let hit = first_hit(start, direction, clearance);
    if (hit.distance < far) {
        return lit_or_shadowed(hit, start, direction);
    }
    return surroundings(start, direction);
}

fn sample(place: vec2<f32>) -> Sample {
    let screen_offset = screen(place) * half_height;
    let start = camera >> (origin + (axes.dual() * vec3<f32>(screen_offset, back)).sum());
    let direction = camera >> -ground_plane.dual();
    let hit = first_hit(start, direction, 0.0);
    let to_ground = -(ground_plane & start) / (ground_plane & direction);
    if (hit.distance >= far || hit.distance > to_ground) {
        let glancing = pow(1 - abs(ground_plane & direction), 5.0);
        let reflectance = stone_reflectance + (1 - stone_reflectance) * glancing;
        return mix(surroundings(start, direction), seen(start + direction * to_ground, ground_plane >> direction), reflectance);
    }
    let point = start + direction * hit.distance;
    let normal = hit.tangent.normalized();
    let finish = finishes[hit.quadric];
    let glancing = pow(1 - abs(normal & direction), 5.0);
    let reflectance = finish.colour + (1 - finish.colour) * glancing;
    return mix(lit_or_shadowed(hit, start, direction), reflectance * seen(point, normal >> direction), finish.mirroring);
}
"""
    return Shader(
        supersampled() + quadric_regions() + lamp_shading() + solids.tangent_seen() + wgsl_source,
        ga,
        dict(Point=Point, Plane=Plane, Tangent=Plane, Motor=Motor, Quadric=Quadric),
        dict(
            axes=solids.axes, origin=origin, plane_x=mv.x, plane_y=mv.y, ground_plane=mv.z,
            stone=stone, tile_variation=tile_variation, mottling=mottling, mottle_scale=mottle_scale,
            stone_reflectance=stone_reflectance, horizon=horizon, zenith=zenith, lamp_size=lamp_size,
            lamp_brightness=lamp_brightness, shadow=shadow, tile_size=tile_size,
            joint_width=joint_width, joint_darkness=joint_darkness,
        ),
    )


def scene(wood: Quadric, balls: Quadric, motor: Motor) -> dict:
    """Wood and steel quadrics placed by `motor` (`[n_quadrics] Plane+ <- Point-`) pair with their surface finishes."""
    n_wood, n_balls = wood.shape[0], balls.shape[0]
    n_quadrics = n_wood + n_balls
    quadrics = concatenate([wood, balls])                             # [n_quadrics] Plane+ <- Point-
    ends = np.concatenate([[n_wood], n_wood + 1 + np.arange(n_balls)])
    starts = np.concatenate([[0], ends[:-1]])
    colours = np.concatenate([[paint, deck_wood], np.repeat([cut_wood], n_wood - 2, axis=0),
                              np.repeat([steel], n_balls, axis=0)])
    gloss = np.concatenate([[paint_gloss], np.zeros(n_quadrics - 1)])
    mirroring = np.concatenate([np.zeros(n_wood), np.full(n_balls, steel_mirroring)])
    return dict(parts=dict(first=starts, end=ends), quadrics=solids.placed(quadrics, motor),
                finishes=dict(colour=colours, gloss=gloss, mirroring=mirroring))


def framing(motors: Motor, samples: Point, elevation: float, azimuth: float, clearance: float,
            n_pixels: int) -> solids.View:
    """Sampled body points and their ground shadows across `motors` set an orthographic camera view `n_pixels` wide."""
    stride = max(1, samples.shape[0] // n_framing_samples)
    framed_samples = samples[::stride]
    n_points = framed_samples.shape[0]
    body_points = motors[..., None] >> framed_samples                 # [..., n_points] Point-
    shadow_points = (body_points & lamp) ^ mv.z                       # [..., n_points] Point-
    camera_view = solids.view([body_points, shadow_points], clearance, elevation, azimuth, margin, n_pixels)
    n_rows = camera_view.shape[0]
    return solids.View(camera_view.camera, np.array([camera_view.half[0], camera_view.half[0] * n_rows / n_pixels]),
                       n_pixels)


def picture(wood: Quadric, balls: Quadric, motor: Motor, camera_view: solids.View) -> np.ndarray:
    """Ray tracing the placed quadrics produces one `[n_rows, n_columns, 3]` RGB byte image of the body and its shadow."""
    raw_image = shader()(
        camera_view.shape, **scene(wood, balls, motor), camera=camera_view.camera,
        back=4 * camera_view.half.max(), half_height=camera_view.half[1],
        lamp=solids.seen_lamp(camera_view, lamp), lamp_in_scene=lamp,
        **lighting, **solids.steps(camera_view),
    )["image"]
    return np.round(np.clip(raw_image, 0, 1) * 255).astype(np.uint8)


def spin_orbit(motor: Motor) -> Motor:
    """Rotations about the vertical line through the centre of mass sweep `motor` across `n_turns` yaw orientations (`[n_turns] Motor+`)."""
    vertical_line = ((motor >> origin) & mv.z.dual()) * solids.flip   # [] Line+
    turns = (vertical_line.normalized() * (np.pi * np.arange(n_turns) / n_turns)).exp()  # [n_turns] Motor+
    return turns * motor


def frames(wood: Quadric, balls: Quadric, samples: Point,
           states: Iterable[tuple[Motor, Spin]]) -> Iterator[np.ndarray]:
    """Yields rendered `[n_rows, n_default_pixels, 3]` frames as `states` stream from the integrator."""
    return sized_frames(wood, balls, samples, states, n_default_pixels)


def sized_frames(wood: Quadric, balls: Quadric, samples: Point,
                 states: Iterable[tuple[Motor, Spin]], n_pixels: int) -> Iterator[np.ndarray]:
    """Yields rendered `[n_rows, n_pixels, 3]` frames from a fixed camera framed around the initial placement."""
    state_stream = iter(states)
    first_state = next(state_stream)
    first_motor, _ = first_state
    camera_view = framing(spin_orbit(first_motor), samples, *camera_angles, drift, n_pixels)
    for motor, _ in chain([first_state], state_stream):
        yield picture(wood, balls, motor, camera_view)


def setup_figure(wood: Quadric, balls: Quadric, samples: Point, motor: Motor) -> plt.Figure:
    """An overhead view of the stationary rattleback displays the deck chamfers and opposite steel balls."""
    return image_figure(picture(wood, balls, motor, framing(motor, samples, *setup_angles, 0.0, n_default_pixels)))
