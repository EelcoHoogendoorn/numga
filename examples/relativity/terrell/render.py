"""Drawing for Terrell rotation, on the GPU: photographs of a passing sphere and cube, one lane per speed, on a
stereographic map of the camera's sky, which draws circles on the sky as circles. Each sample of the map is the ray
of light arriving from there; read in the body's frame, it runs straight back from where the camera is to the pair
at rest, and the first surface it meets colours it."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from matplotlib.colors import to_rgb

from examples.relativity.terrell import core
from examples.util.wgsl import Shader, supersampled, vec3

mv = core.mv
# The colours of the cube's faces: the top, the bottom, the face ahead, the face behind, and the faces towards and
# away from the camera. The sphere's fill, grid and outline, and the edges of the faces.
face_colours = ["#fde68a", "#a8a29e", "#fca5a5", "#93c5fd", "#c4b5fd", "#86efac"]
sphere_colour = "#e0f2fe"
grid_colour = "#0369a1"
outline_colour = "#0c4a6e"
edge_colour = "#334155"
# The window of the map each lane shows, across and up, the pixels across it, and the pixels between the lanes; in
# pixels, the widths of the cube's edges, of the sphere's grid and of its outline.
window = ((-0.61, 0.61), (-0.39, 0.34))
n_pixels = 900
lane_gap = 10
edge_width, grid_width, outline_width = 2.4, 1.2, 2.2


# --- plumbing -------------------------------------------------------------------------
def photograph() -> Shader:
    """The shader of one photograph."""
    # The faces by their outward normals along z, x and y, each the positive way first: away from the camera is +y.
    colours = ", ".join(vec3(to_rgb(face_colours[index])) for index in (0, 1, 2, 3, 5, 4))
    return Shader(supersampled() + f"""
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

// The faces' colours by their outward normals: up, down, ahead, behind, away from the camera and towards it.
const faces = array<vec3<f32>, 6>({colours});
const sky = vec3<f32>(1.0, 1.0, 1.0);
const fill = {vec3(to_rgb(sphere_colour))};
const grid = {vec3(to_rgb(grid_colour))};
const rim = {vec3(to_rgb(outline_colour))};
const edge = {vec3(to_rgb(edge_colour))};
const far = 1e6;
// The widths of the cube's edges, the sphere's grid and its outline, in pixels.
const edge_width = {float(edge_width)};
const grid_width = {float(grid_width)};
const outline_width = {float(outline_width)};

// A place in space, by its pairings with the slices of space.
fn place(at: Antivector) -> vec3<f32> {{
    return vec3<f32>(mv.x & at, mv.y & at, mv.z & at);
}}

// How far back along a ray from an eye its line meets a sphere first, far where it passes.
fn sphere_hit(eye: vec3<f32>, back: vec3<f32>, centre: vec3<f32>) -> f32 {{
    let offset = eye - centre;
    let b = dot(offset, back);
    let discriminant = b * b - dot(offset, offset) + radius * radius;
    if (discriminant < 0) {{
        return far;
    }}
    let distance = -b - sqrt(discriminant);
    return select(far, distance, distance > 0);
}}

// How far back along a ray from an eye it enters an upright cube, far where it passes: between the planes of each
// pair of opposite faces, the latest entry before the earliest exit.
fn cube_hit(eye: vec3<f32>, back: vec3<f32>, centre: vec3<f32>) -> f32 {{
    let inverse = 1 / back;
    let low = (centre - half_width - eye) * inverse;
    let high = (centre + half_width - eye) * inverse;
    let entry = max(max(min(low.x, high.x), min(low.y, high.y)), min(low.z, high.z));
    let exit = min(min(max(low.x, high.x), max(low.y, high.y)), max(low.z, high.z));
    return select(far, entry, entry < exit && entry > 0);
}}

// The unit direction back along the light arriving from a point of the map, in the body's frame.
fn backward(at: vec2<f32>) -> vec3<f32> {{
    let size = dot(at, at);
    // The direction the light travels in the camera's frame, from the map's point by the inverse of the stereographic
    // projection, and the ray of light arriving along it: one unit of the camera's time and one of space.
    let heading = vec3<f32>(-2 * at.x, size - 1, -2 * at.y) / (size + 1);
    let arriving = -mv.t.dual() + mv.x.dual() * heading.x + mv.y.dual() * heading.y + mv.z.dual() * heading.z;
    // The same ray in the body's frame, where the light runs straight back from the camera to a body at rest.
    let seen = frame << arriving;
    return -place(seen) / (mv.t & seen);
}}

fn sample(place_on_map: vec2<f32>) -> Sample {{
    let at = middle + screen(place_on_map) * half_height;
    let back = backward(at);
    let eye = place(camera);
    // A pixel's width in space where the ray has run a distance: the angle to the ray a pixel over, times the distance.
    let spread = length(backward(at + vec2<f32>(pixel, 0)) - back);
    let round_centre = place(sphere);
    let box_centre = place(cube);
    let to_sphere = sphere_hit(eye, back, round_centre);
    let to_cube = cube_hit(eye, back, box_centre);
    var colour = sky;
    if (to_cube < to_sphere) {{
        // The face is the axis the hit lies furthest along from the cube's centre; an edge where a second lies as far.
        let local = (eye + back * to_cube - box_centre) / half_width;
        let reach = abs(local);
        let largest = max(max(reach.x, reach.y), reach.z);
        let axis = select(select(0u, 2u, reach.x == largest), 4u, reach.y == largest && reach.x != largest);
        let along = select(select(local.z, local.x, axis == 2u), local.y, axis == 4u);
        colour = faces[axis + select(1u, 0u, along > 0)];
        let second = reach.x + reach.y + reach.z - largest - min(min(reach.x, reach.y), reach.z);
        if ((1 - second) * half_width < edge_width / 2 * spread * to_cube) {{
            colour = edge;
        }}
    }} else if (to_sphere < far) {{
        // The grid: within half a line's width of a meridian or a parallel, along the sphere's surface.
        let unit = (eye + back * to_sphere - round_centre) / radius;
        let latitude = asin(clamp(unit.z, -1.0, 1.0));
        let longitude = atan2(unit.y, unit.x);
        let off_meridian = abs(longitude - round(longitude / meridians) * meridians) * cos(latitude);
        let nearest = clamp(round(latitude / parallels) * parallels, -parallel_reach, parallel_reach);
        let off_parallel = abs(latitude - nearest);
        let half_line = grid_width / 2 * spread * to_sphere / radius;
        colour = select(fill, grid, min(off_meridian, off_parallel) < half_line);
    }}
    // The outline: where the ray passes within half a line's width of grazing the sphere.
    let offset = eye - round_centre;
    let closest = -dot(offset, back);
    let passing = sqrt(max(dot(offset, offset) - closest * closest, 0.0));
    if (abs(passing - radius) < outline_width / 2 * spread * closest && closest < to_cube) {{
        colour = rim;
    }}
    return colour;
}}
""", core.ga, dict(Rotor=core.ga.gatype.rotor(), Antivector=core.ga.gatype.antivector()), {})


def animate_photographs(frame: core.Rotor, cameras: Iterable[core.Antivector], sphere: core.Antivector, radius: float,
                        cube: core.Antivector, half_width: float, meridians: float, parallels: float,
                        parallel_reach: float) -> Iterator[np.ndarray]:
    """Each frame's photographs, one lane per case of the body's frame `[cases]`, from where the camera is in each
    `[cases]`: the sphere with its grid of meridians and parallels so many radians apart, the parallels reaching so far
    from its equator, above the cube, its faces coloured by the way they face."""
    shader = photograph()
    (left, right), (bottom, top) = window
    shape = (round(n_pixels * (top - bottom) / (right - left)), n_pixels)
    gap = np.full((lane_gap, n_pixels, 3), 255, dtype=np.uint8)
    for camera in cameras:
        lanes = []
        for case in range(frame.shape[0]):
            image = shader(shape, frame=frame[case], camera=camera[case], sphere=sphere, radius=radius, cube=cube,
                           half_width=half_width, meridians=meridians, parallels=parallels, parallel_reach=parallel_reach,
                           middle=np.array([(left + right) / 2, (bottom + top) / 2]), half_height=(top - bottom) / 2,
                           pixel=(right - left) / n_pixels)["image"]
            lanes += [np.round(np.clip(image, 0, 1) * 255).astype(np.uint8), gap]
        yield np.concatenate(lanes[:-1])
