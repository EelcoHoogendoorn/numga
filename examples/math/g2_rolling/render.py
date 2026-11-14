"""A supersampled GPU ray tracer renders a painted ball rolling over a fixed sphere with inked contact routes.

Orthographic camera rays intersect both spheres where their squared distance from each centre equals the squared radius. Carrying the ball's surface normal back into the body frame via `attitude << normal` and pairing it with the three coordinate planes selects the octant colour and dark seam lines, while screen-space distances paint the sampled contact arcs and endpoint dots onto the facing hemisphere of the fixed sphere.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from functools import cache

import numpy as np

from numga import concatenate, stack
from numga.wgsl import Shader, supersampled

from .core import Plane, Point, Rotor, ga, mv

route_colours = np.array([[27, 111, 152], [213, 99, 36], [113, 67, 151]]) / 255
line_width = 3
dot_radius = 4
margin = 0.07
# Camera screen planes for horizontal and vertical readout, body octant planes, and camera-frame directions.
screen_planes = stack((mv.x, mv.y))                                           # [2] Plane+
octant_planes = stack((mv.x, mv.y, mv.z))                                     # [3] Plane+
view_offset = mv.z.dual()                                                     # [] Point-
lamp_direction = (mv.x * -0.40 + mv.y * 0.48 + mv.z * 0.78).dual()            # [] Point-

source = """
var<uniform> camera: Rotor;
var<uniform> middle: Point;
var<uniform> back: f32;
var<uniform> half_height: f32;
var<uniform> lamp: Point;
var<uniform> sphere_radius: f32;
var<uniform> ball_radius: f32;
var<uniform> contact: Point;
var<uniform> attitude: Rotor;
var<storage, read> segments: array<Segment>;
var<storage, read> ends: array<End>;
var<uniform> line_half_width: f32;
var<uniform> dot_radius: f32;
var<uniform> rim_width: f32;

alias Sample = vec3<f32>;

struct Segment {
    start: Point,
    end: Point,
    colour: vec3<f32>,
}

struct End {
    place: Point,
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

fn sphere_hit(offset: Point, direction: Point, radius: f32) -> f32 {
    let depth = direction.dual() & offset;
    let discriminant = depth * depth - offset.scalar_norm_squared() + radius * radius;
    return select(far, -depth - sqrt(discriminant), discriminant >= 0);
}

fn view_projection(offset: Point, direction: Point) -> Point {
    return offset - direction * (direction.dual() & offset);
}

fn route_paint(point: Point, direction: Point) -> vec3<f32> {
    var paint = surface;
    var nearest = line_half_width;
    for (var index = 0u; index < arrayLength(&segments); index++) {
        let segment = segments[index];
        let span = view_projection(segment.end - segment.start, direction);
        let offset = view_projection(point - segment.start, direction);
        let fraction = clamp((span.dual() & offset) / (span.dual() & span), 0.0, 1.0);
        let distance = (offset - span * fraction).norm();
        if (distance < nearest && (direction.dual() & (segment.start + segment.end)) < 0) {
            nearest = distance;
            paint = segment.colour;
        }
    }
    for (var index = 0u; index < arrayLength(&ends); index++) {
        let distance = view_projection(point - ends[index].place, direction).norm();
        if (distance < dot_radius && (direction.dual() & ends[index].place) < 0) {
            paint = select(rim, ends[index].colour, distance < dot_radius - rim_width);
        }
    }
    return paint;
}

fn sample(place: vec2<f32>) -> Sample {
    let across = screen(place) * half_height;
    let start = middle + (camera >> (view_offset * back + (screen_planes.dual() * across).sum()));
    let direction = camera >> -view_offset;
    let centre = contact * (sphere_radius + ball_radius);
    let to_sphere = sphere_hit(start, direction, sphere_radius);
    let to_ball = sphere_hit(start - centre, direction, ball_radius);
    if (to_ball < to_sphere) {
        let normal = (start + direction * to_ball - centre) / ball_radius;
        let body = attitude << normal;
        let coordinates = vec3<f32>(octant_planes & body);
        let octant = (select(0u, 1u, coordinates.x >= 0)
                      + select(0u, 2u, coordinates.y >= 0)
                      + select(0u, 4u, coordinates.z >= 0));
        let seam = clamp(
            1 - min(abs(coordinates.x), min(abs(coordinates.y), abs(coordinates.z))) / seam_width,
            0.0, 1.0,
        );
        return mix(octants[octant], seams, seam) * (0.47 + 0.53 * max(normal.dual() & lamp, 0.0));
    }
    if (to_sphere < far) {
        let point = start + direction * to_sphere;
        return route_paint(point, direction) * (0.58 + 0.42 * max((point / sphere_radius).dual() & lamp, 0.0));
    }
    return background;
}
"""


# --- plumbing -------------------------------------------------------------------------
@cache
def route_shader() -> Shader:
    """The scene shader binds the camera screen planes, body octant planes, and view offset into the supersampled WGSL ray tracer."""
    return Shader(
        supersampled() + source,
        ga,
        dict(Plane=Plane, Point=Point, Rotor=Rotor),
        dict(screen_planes=screen_planes, octant_planes=octant_planes, view_offset=view_offset),
    )


def scene_renderer(
    arcs: tuple[Point, ...], fit_arcs: tuple[Point, ...],
    sphere_radius: float, ball_radius: float,
    elevation: float, azimuth: float, size: int,
) -> Callable[[Point, Rotor], np.ndarray]:
    """Camera framing across `fit_arcs` constructs a fixed-view image renderer enclosing both spheres."""
    elevation_rad, azimuth_rad = np.radians(elevation), np.radians(azimuth + 90)
    camera = (mv.xy * (-azimuth_rad / 2)).exp() * (mv.yz * (-(np.pi / 2 - elevation_rad) / 2)).exp()   # [] Rotor+
    centres = concatenate(fit_arcs) * (sphere_radius + ball_radius)                                     # [...] Point-
    screen_axes = camera >> screen_planes                                                               # [2] Plane+
    readings = screen_axes[:, None] & centres                                                           # [2, ...] Scalar+
    low = ((screen_axes & centres[readings.argmin(axis=-1)]) - ball_radius).clip(-np.inf, -sphere_radius)    # [2] Scalar+
    high = ((screen_axes & centres[readings.argmax(axis=-1)]) + ball_radius).clip(sphere_radius, np.inf)     # [2] Scalar+
    middle = (screen_axes.dual() * ((low + high) / 2)).sum()                                            # [] Point-
    spans = high - low                                                                                  # [2] Scalar+
    half_height = float((spans.max() / 2 * (1 + margin)).to_array())
    arc_points = [arc * sphere_radius for arc in arcs]
    segments = dict(
        start=concatenate([points[:-1] for points in arc_points]),
        end=concatenate([points[1:] for points in arc_points]),
        colour=np.concatenate([
            np.repeat(route_colours[[index]], len(points) - 1, axis=0)
            for index, points in enumerate(arc_points)
        ]),
    )
    ends = dict(
        place=concatenate([points[[0, -1]] for points in arc_points]),
        colour=np.repeat(route_colours[:len(arc_points)], 2, axis=0),
    )
    scene_lamp = camera >> lamp_direction                                                               # [] Point-
    pixel = 2 * half_height / size

    def frame_image(contact: Point, attitude: Rotor) -> np.ndarray:
        """Orthographic ray tracing renders one `[size, size, 3]` RGB image for `contact` and `attitude`."""
        image = route_shader()(
            (size, size), camera=camera, middle=middle, back=4 * half_height,
            half_height=half_height, lamp=scene_lamp, sphere_radius=sphere_radius,
            ball_radius=ball_radius, contact=contact, attitude=attitude, segments=segments,
            ends=ends, line_half_width=line_width / 2 * pixel, dot_radius=dot_radius * pixel,
            rim_width=pixel,
        )["image"]
        return np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)

    return frame_image


def route_frames(
    arcs: tuple[Point, ...], sphere_radius: float, ball_radius: float,
    elevation: float, azimuth: float, size: int,
    placements: Iterable[tuple[Point, Rotor]],
) -> Iterator[np.ndarray]:
    """Yields `[size, size, 3]` RGB frames as the ball traverses `placements` along `arcs`."""
    frame_image = scene_renderer(arcs, arcs, sphere_radius, ball_radius, elevation, azimuth, size)
    for contact, attitude in placements:
        yield frame_image(contact, attitude)


def comparison_frames(
    arcs: tuple[Point, ...], transformed_arcs: tuple[Point, ...],
    sphere_radius: float, ball_radius: float, elevation: float, azimuth: float, size: int,
    paired_placements: Iterable[tuple[tuple[Point, Rotor], tuple[Point, Rotor]]],
) -> Iterator[np.ndarray]:
    """Yields `[size, 2 * size, 3]` side-by-side RGB frames comparing a route with its symmetry image."""
    fit_arcs = arcs + transformed_arcs
    left_image = scene_renderer(arcs, fit_arcs, sphere_radius, ball_radius, elevation, azimuth, size)
    right_image = scene_renderer(transformed_arcs, fit_arcs, sphere_radius, ball_radius, elevation, azimuth, size)
    for placement, transformed_placement in paired_placements:
        yield np.concatenate([left_image(*placement), right_image(*transformed_placement)], axis=1)
