"""Drawing for Terrell rotation, on the GPU: photographs of a passing sphere and cube, one lane per speed, on a
stereographic map of the camera's sky, which draws circles on the sky as circles. Each sample of the map is the ray
of light arriving from there; read in the body's frame, it runs straight back from where the camera is to the pair
at rest, and the first surface it meets colours it."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.relativity.terrell import core
from examples.util.wgsl import Shader, supersampled

mv = core.mv
Vector = core.ga.gatype.vector()
_, line_of_sight, _ = core.space
# The colours of the cube's faces: the top, the bottom, the face ahead, the face behind, and the faces towards and
# away from the camera. The sky, the sphere's fill, grid and outline, and the edges of the faces.
face_colours = ["#fde68a", "#a8a29e", "#fca5a5", "#93c5fd", "#c4b5fd", "#86efac"]
sky_colour = "#ffffff"
sphere_colour = "#e0f2fe"
grid_colour = "#0369a1"
outline_colour = "#0c4a6e"
edge_colour = "#334155"
# The slices of space the cube's faces face along, outward, in the order of their colours.
face_slices = stack((mv.z, -mv.z, mv.x, -mv.x, -mv.y, mv.y))                 # [6] Vector+
# The slices of space whose pairings with a place on the map's chart are how far across and up the map it lies: along
# the travel and up; the place so far along each is `(axes.dual() * at).sum()`.
axes = stack((mv.x, mv.z))                                                   # [2] Vector+
# The window of the map each lane shows, across and up, the pixels across it, and the pixels between the lanes; in
# pixels, the widths of the cube's edges, of the sphere's grid and of its outline.
window = ((-0.61, 0.61), (-0.39, 0.34))
n_pixels = 400
lane_gap = 5
edge_width, grid_width, outline_width = 1.1, 0.6, 1.0


source = """
alias Sample = vec3<f32>;
// The body's frame, where the camera is in it; the sphere's centre and radius, the cube's centre and half its width,
// the angles between the sphere's meridians and between its parallels, and how far its parallels reach from its
// equator; the middle of the window on the map, half its height, and a pixel's size on the map.
var<uniform> frame: Rotor;
var<uniform> camera: Antivector;
var<uniform> sphere: Antivector;
var<uniform> radius: f32;
var<uniform> cube: Antivector;
var<uniform> half_width: f32;
var<uniform> meridians: f32;
var<uniform> parallels: f32;
var<uniform> parallel_reach: f32;
var<uniform> middle: vec2<f32>;
var<uniform> half_height: f32;
var<uniform> pixel: f32;

// A face of the cube: the slice of space it faces along, outward, and its colour.
struct Face {
    slice: Vector,
    colour: vec3<f32>,
}

var<storage, read> faces: array<Face>;

const far = 1e6;

// The direction back along the light arriving from a point of the map, in the body's frame, at unit length.
fn backward(at: vec2<f32>) -> Antivector {
    // The map's point on its chart, the plane through the camera square to its line of sight, and the ray of light
    // arriving from there: the ray from the map's pole behind the camera, reflected in the plane square to the chart's
    // point less the pole, the inverse of the stereographic projection.
    let chart = (axes.dual() * at).sum();
    let arriving = (chart - behind) >> (time - behind);
    // The same ray in the body's frame, where the light runs straight back from the camera to a body at rest, at one
    // unit of the camera's time: less that unit, it runs forward along the light.
    let seen = frame << arriving;
    return time - seen / (mv.t & seen);
}

// How far back along a ray from the camera its line meets the sphere first, far where it passes, by the metric that
// makes the sphere round.
fn sphere_hit(back: Antivector) -> f32 {
    let offset = camera - sphere;
    let b = offset | back;
    let discriminant = b * b - (offset | offset) + radius * radius;
    if (discriminant < 0) {
        return far;
    }
    let distance = -b - sqrt(discriminant);
    return select(far, distance, distance > 0);
}

// Where a ray from the camera enters the cube: how far back along it, far where it passes, and the face it enters by.
struct Entry {
    distance: f32,
    face: u32,
}

// Each face's pairing with the offset from the cube's centre, less half the cube's width, is linear in the distance
// along the ray, b t + c, and the ray is inside where every one is negative: from the last face it enters to the first
// it leaves.
fn cube_hit(back: Antivector) -> Entry {
    let offset = camera - cube;
    var entry = -far;
    var exit = far;
    var entering = 0u;
    for (var index = 0u; index < arrayLength(&faces); index++) {
        let slice = faces[index].slice;
        let b = slice & back;
        let c = (slice & offset) - half_width;
        // Along a face, the ray is inside its plane everywhere or nowhere.
        if (b == 0) {
            if (c > 0) {
                return Entry(far, 0u);
            }
            continue;
        }
        let crossing = -c / b;
        if (b < 0 && crossing > entry) {
            entry = crossing;
            entering = index;
        }
        if (b > 0 && crossing < exit) {
            exit = crossing;
        }
    }
    return Entry(select(far, entry, entry < exit && entry > 0), entering);
}

fn sample(place_on_map: vec2<f32>) -> Sample {
    let at = middle + screen(place_on_map) * half_height;
    let back = backward(at);
    // A pixel's width in space where the ray has run a distance: the angle to the ray a pixel over, times the distance.
    let spread = (backward(at + vec2<f32>(pixel, 0)) - back).squared().square_root();
    let to_sphere = sphere_hit(back);
    let to_cube = cube_hit(back);
    var colour = sky;
    if (to_cube.distance < to_sphere) {
        // The face the ray enters by colours it; an edge where the point lies within half an edge's width of another
        // face's plane.
        let local = camera + back * to_cube.distance - cube;
        colour = faces[to_cube.face].colour;
        var gap = far;
        for (var index = 0u; index < arrayLength(&faces); index++) {
            if (index != to_cube.face) {
                gap = min(gap, half_width - (faces[index].slice & local));
            }
        }
        if (gap < edge_width / 2 * spread * to_cube.distance) {
            colour = edge;
        }
    } else if (to_sphere < far) {
        // The grid: within half a line's width of a meridian or a parallel, along the sphere's surface.
        let unit = (camera + back * to_sphere - sphere) / radius;
        let latitude = asin(clamp(mv.z & unit, -1.0, 1.0));
        let longitude = atan2(mv.y & unit, mv.x & unit);
        let off_meridian = abs(longitude - round(longitude / meridians) * meridians) * cos(latitude);
        let nearest = clamp(round(latitude / parallels) * parallels, -parallel_reach, parallel_reach);
        let off_parallel = abs(latitude - nearest);
        let half_line = grid_width / 2 * spread * to_sphere / radius;
        colour = select(fill, grid, min(off_meridian, off_parallel) < half_line);
    }
    // The outline: where the ray passes within half a line's width of grazing the sphere.
    let offset = camera - sphere;
    let closest = -(offset | back);
    let passing = sqrt(max((offset | offset) - closest * closest, 0.0));
    if (abs(passing - radius) < outline_width / 2 * spread * closest && closest < to_cube.distance) {
        colour = rim;
    }
    return colour;
}
"""


# --- plumbing -------------------------------------------------------------------------
def photograph() -> Shader:
    """The shader of one photograph: the map's frame, its pole behind the camera and one unit of the camera's time,
    the colours and the widths of the lines, in pixels, given as its constants."""
    return Shader(supersampled() + source, core.ga, dict(Rotor=core.Rotor, Antivector=core.Antivector, Vector=Vector),
                  dict(axes=axes, behind=-line_of_sight, time=core.time, sky=to_rgb(sky_colour), fill=to_rgb(sphere_colour),
                       grid=to_rgb(grid_colour), rim=to_rgb(outline_colour), edge=to_rgb(edge_colour),
                       edge_width=edge_width, grid_width=grid_width, outline_width=outline_width))


def animate_photographs(frame: core.Rotor, cameras: Iterable[core.Antivector], sphere: core.Antivector, radius: float,
                        cube: core.Antivector, half_width: float, meridians: float, parallels: float,
                        parallel_reach: float) -> Iterator[np.ndarray]:
    """Each frame's photographs, one lane per case of the body's frame `[cases]`, from where the camera is in each
    `[cases]`: the sphere with its grid of meridians and parallels so many radians apart, the parallels reaching so far
    from its equator, above the cube, its faces coloured by the way they face."""
    shader = photograph()
    faces = dict(slice=face_slices, colour=np.array([to_rgb(colour) for colour in face_colours]))
    (left, right), (bottom, top) = window
    shape = (round(n_pixels * (top - bottom) / (right - left)), n_pixels)
    gap = np.full((lane_gap, n_pixels, 3), 255, dtype=np.uint8)
    for camera in cameras:
        lanes = []
        for case in range(frame.shape[0]):
            image = shader(shape, frame=frame[case], camera=camera[case], sphere=sphere, radius=radius, cube=cube,
                           half_width=half_width, meridians=meridians, parallels=parallels, parallel_reach=parallel_reach,
                           middle=np.array([(left + right) / 2, (bottom + top) / 2]), half_height=(top - bottom) / 2,
                           pixel=(right - left) / n_pixels, faces=faces)["image"]
            lanes += [np.round(np.clip(image, 0, 1) * 255).astype(np.uint8), gap]
        yield np.concatenate(lanes[:-1])
