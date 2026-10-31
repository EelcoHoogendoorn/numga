"""Surface-fixed ink feathered along a field's visible projection.

Magnitude controls streak length at fixed seed density and opacity. The longer end fades downstream;
setting `downstream_fraction=0.5` makes an unoriented texture. A zero field leaves stationary specks.
The image-space traces stop at silhouettes and depth jumps; they are not intrinsic surface streamlines.
"""
from __future__ import annotations

from functools import cache

import numpy as np

from matplotlib.axes import Axes

from numga.extensor import Extensor
from numga.gatype import GAType
from examples.util.wgsl import Shader, meshes, solids
from examples.util.wgsl.simplices import view_camera
from examples.util.wgsl.camera import Camera, extrinsic_first, flat_panel, placing, reading, whole
from examples.util.wgsl.library import pinned, samples_per_side

Mesh = meshes.Mesh

n_trace_steps = 64
length_pixels = 102.4
ink_opacity = 0.68

surface_source = """
var<storage, read> points: array<Point>;
var<storage, read> faces: array<vec3<u32>>;
var<storage, read> field: array<Point>;
var<storage, read> colours: array<vec3<f32>>;
struct Normals { a: Tangent, b: Tangent, c: Tangent, }
var<storage, read> normals: array<Normals>;
var<uniform> noise_scale: f32;

fn hash(place: vec3<f32>) -> f32 {
    var p = fract(place * 0.1031);
    p += p.x * (p.y + 33.33) + p.y * (p.z + 33.33) + p.z * (p.x + 33.33);
    return fract((p.x + p.y) * p.z);
}

struct Corner {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: Tangent,
    @location(1) tangent: Point,
    @location(2) place: Point,
    @location(3) depth: f32,
    @location(4) colour: vec3<f32>,
}
struct Shade {
    @location(0) colour: vec4<f32>,
    @location(1) flow: vec4<f32>,
    @location(2) grain: vec4<f32>,
}

@vertex
fn place(@builtin(vertex_index) index: u32) -> Corner {
    let face = faces[index / 3];
    let vertex = face[index % 3];
    let point = points[vertex];
    let corners = normals[index / 3];
    let normal = select(select(corners.c, corners.b, index % 3 == 1), corners.a, index % 3 == 0);
    return Corner(clip(point), normal, field[vertex], point, -paired(screen_toward, point) / weight_of(point),
                  colours[vertex]);
}

@fragment
fn shade(corner: Corner) -> Shade {
    let lighting = lit(corner.colour, corner.normal);
    // The field's change of place in the picture, read by the camera's planes, rows counted down.
    let projected = screen_of(corner.tangent) * vec2<f32>(1, -1);
    return Shade(vec4<f32>(lighting, 1), vec4<f32>(projected, 0, corner.depth), vec4<f32>(seed(corner.place)));
}
"""

surface_grain = """
fn seed(place: Point) -> f32 {
    let coordinates = (axes & place) / noise_scale;
    let cell = floor(coordinates);
    let local = fract(coordinates);
    let blend = local * local * (3.0 - 2.0 * local);
    let low = mix(mix(hash(cell), hash(cell + vec3<f32>(1, 0, 0)), blend.x),
                  mix(hash(cell + vec3<f32>(0, 1, 0)), hash(cell + vec3<f32>(1, 1, 0)), blend.x), blend.y);
    let high = mix(mix(hash(cell + vec3<f32>(0, 0, 1)), hash(cell + vec3<f32>(1, 0, 1)), blend.x),
                   mix(hash(cell + vec3<f32>(0, 1, 1)), hash(cell + vec3<f32>(1, 1, 1)), blend.x), blend.y);
    return 1.0 - smoothstep(0.18, 0.24, mix(low, high, blend.z));
}
"""

# The seeds of a flat field in the points' own algebra, placed by its camera: no lighting, its field, as it is, read
# across and up the picture by the camera's planes, its specks jittered on cells of the picture's own readings.
planar_source = """
var<storage, read> points: array<Point>;
var<storage, read> faces: array<vec3<u32>>;
var<storage, read> field: array<Field>;
var<uniform> noise_scale: f32;

fn hash(place: vec3<f32>) -> f32 {
    var p = fract(place * 0.1031);
    p += p.x * (p.y + 33.33) + p.y * (p.z + 33.33) + p.z * (p.x + 33.33);
    return fract((p.x + p.y) * p.z);
}

struct Corner {
    @builtin(position) position: vec4<f32>,
    @location(0) tangent: Field,
    @location(1) place: Point,
}
struct Shade {
    @location(0) colour: vec4<f32>,
    @location(1) flow: vec4<f32>,
    @location(2) grain: vec4<f32>,
}

// Jittered specks, one to a cell of the picture's readings, `noise_scale` across, an eighth of a cell in radius.
fn seed(place: Point) -> f32 {
    let reading = vec2<f32>(paired(screen_across, place), paired(screen_up, place)) / weight_of(place) / noise_scale;
    let cell = floor(reading);
    var ink = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let neighbour = cell + vec2<f32>(f32(x), f32(y));
            let jitter = vec2<f32>(hash(vec3<f32>(neighbour, 19.1)), hash(vec3<f32>(neighbour, 53.4)));
            let away = reading - neighbour - jitter;
            let distance = sqrt(away.x * away.x + away.y * away.y);
            ink = max(ink, 1.0 - smoothstep(0.09, 0.16, distance));
        }
    }
    return ink;
}

@vertex
fn place(@builtin(vertex_index) index: u32) -> Corner {
    let face = faces[index / 3];
    let vertex = face[index % 3];
    return Corner(clip(points[vertex]), field[vertex], points[vertex]);
}

@fragment
fn shade(corner: Corner) -> Shade {
    // The field's change of place in the picture, in the readings, rows counted down.
    let projected = vec2<f32>(field_paired(screen_across, corner.tangent), field_paired(screen_up, corner.tangent))
                    * vec2<f32>(1, -1);
    return Shade(vec4<f32>(1.0), vec4<f32>(projected, 0, 0), vec4<f32>(seed(corner.place)));
}
"""

trace_source = """
var surface: texture_2d<f32>;
var flow: texture_2d<f32>;
var grains: texture_2d<f32>;
var smooth: sampler;
var<uniform> pixels: vec2<u32>;
var<uniform> background: vec3<f32>;
var<uniform> step_pixels: f32;
var<uniform> reference_speed: f32;
var<uniform> downstream_fraction: f32;
var<uniform> depth_limit: f32;
var<uniform> ink_opacity: f32;
var<storage, read_write> image: array<vec4<f32>>;
var<storage, read_write> coverage: array<f32>;

fn reading(texture: texture_2d<f32>, place: vec2<f32>) -> vec4<f32> {
    return textureSampleLevel(texture, smooth, place / vec2<f32>(pixels), 0.0);
}

fn stroke(start: vec2<f32>, sense: f32) -> f32 {
    var place = start;
    var previous = reading(flow, place);
    var ink = reading(grains, place).a;
    // Gathering upstream finds the seeds whose plumes extend downstream to this pixel.
    let reach = select(downstream_fraction, 1.0 - downstream_fraction, sense > 0.0);
    let step = sense * reach * step_pixels / reference_speed;
    for (var index = 0u; index < n_steps; index++) {
        let midpoint = place + step * previous.xy / 2.0;
        let middle = reading(flow, midpoint);
        if (reading(surface, midpoint).a < 0.98 || abs(middle.w - previous.w) > depth_limit) { break; }
        place += step * middle.xy;
        let sample = reading(flow, place);
        if (any(place < vec2<f32>(0.0)) || any(place >= vec2<f32>(pixels))) { break; }
        if (reading(surface, place).a < 0.98 || abs(sample.w - previous.w) > depth_limit) { break; }
        let time = f32(index + 1u) / f32(n_steps);
        let feather = 0.5 + 0.5 * cos(3.14159265 * time);
        // Coverage grows with length; averaging would wash a seed out as it spreads.
        ink = max(ink, feather * reading(grains, place).a);
        previous = sample;
    }
    return ink;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= pixels.x || id.y >= pixels.y) { return; }
    let place = vec2<f32>(id.xy) + 0.5;
    let colour = reading(surface, place);
    var opacity = 0.0;
    if (colour.a > 0.0) {
        opacity = ink_opacity * max(stroke(place, 1.0), stroke(place, -1.0));
    }
    let result = colour.rgb * (1.0 - opacity) + background * (1.0 - colour.a);
    image[id.y * pixels.x + id.x] = vec4<f32>(result, 1.0);
    coverage[id.y * pixels.x + id.x] = colour.a * opacity;
}
"""


@cache
def seed_shader(grain: str) -> Shader:
    return Shader(solids.lamp_shading() + solids.tangent_seen() + placing(solids.Plane) + surface_source + grain,
                  solids.ga, dict(Point=solids.Point, Plane=solids.Plane, Tangent=solids.Patch), dict(axes=solids.axes, origin=solids.origin),
                  targets=("rgba8unorm", "rgba32float", "rgba32float"))


@cache
def planar_shader(Point: GAType, Plane: GAType, Field: GAType) -> Shader:
    """The seed pass of a flat field of the type `Field` over points of the type `Point`, both read by a camera's
    planes of the type `Plane`, the extrinsic first."""
    pairing = "plane & field" if extrinsic_first(Plane) else "field & plane"
    field_pairing = f"""
// A camera plane's reading of the field, their regressive product, the extrinsic first.
fn field_paired(plane: Plane, field: Field) -> f32 {{
    return {pairing};
}}
"""
    return Shader(placing(Plane) + field_pairing + planar_source, Point.algebra,
                  dict(Point=Point, Plane=Plane, Field=Field), {}, targets=("rgba8unorm", "rgba32float", "rgba32float"))


@cache
def trace_shader() -> Shader:
    return Shader(f"const n_steps = {n_trace_steps}u;\n" + trace_source, solids.ga, {}, {})


def texture(points: solids.Point, mesh: Mesh, field: solids.Point, view: solids.View,
            colours: np.ndarray, reference_speed: float, grain: str, grain_pixels: float,
            downstream_fraction: float, length: float, background: np.ndarray) -> dict:
    """The shaded surface and ink coverage share one projected field and one feathering pass."""
    surface, projected, grains = seed_shader(grain).draw_averaged(
        view.shape, samples_per_side(), 3 * len(mesh.faces), points=points, faces=mesh.faces, field=field,
        normals=meshes.smooth(points, mesh), noise_scale=view.pixel * grain_pixels,
        colours=np.broadcast_to(colours, points.shape + (3,)), **view_camera(view).uniforms(8 * view.half.max()),
        **solids.lighting(0.06), **solids.steps(view))
    return trace_shader()(view.shape, surface=surface, flow=projected, grains=grains,
                          background=background, step_pixels=length / n_trace_steps,
                          reference_speed=max(reference_speed, np.finfo(np.float32).tiny),
                          downstream_fraction=downstream_fraction, depth_limit=6 * view.pixel,
                          ink_opacity=ink_opacity)


def draw(points: solids.Point, mesh: Mesh, field: solids.Point, view: solids.View,
         colours: np.ndarray, reference_speed: float, *, downstream_fraction: float = 0.9,
         length: float = length_pixels) -> np.ndarray:
    """An opaque shaded surface with fixed seeds and magnitude-proportional feathered streaks."""
    result = texture(points, mesh, field, view, colours, reference_speed, surface_grain, 2.2,
                     downstream_fraction, length, np.ones(3))
    return np.round(np.clip(result["image"][..., :3], 0, 1) * 255).astype(np.uint8)


def layer(points: Extensor, mesh: Mesh, field: Extensor, camera: Camera,
          reference_speed: float, colour: np.ndarray, *, downstream_fraction: float = 0.9,
          length: float = length_pixels) -> np.ndarray:
    """A transparent ink layer, at the camera's own pixels, for a flat field over a panel, the points and the field in
    their own algebra, read by its camera."""
    surface, projected, grains = planar_shader(**reading(points, camera), Field=whole(field, points.algebra)).draw_averaged(
        camera.shape, samples_per_side(), 3 * len(mesh.faces), points=points, faces=mesh.faces, field=field,
        noise_scale=camera.pixel * 6.0, **camera.uniforms(1.0))
    result = trace_shader()(camera.shape, surface=surface, flow=projected, grains=grains, background=np.ones(3),
                            step_pixels=length / n_trace_steps,
                            reference_speed=max(reference_speed, np.finfo(np.float32).tiny),
                            downstream_fraction=downstream_fraction, depth_limit=6 * camera.pixel,
                            ink_opacity=ink_opacity)
    rgba = np.empty(camera.shape + (4,))
    rgba[..., :3] = colour
    rgba[..., 3] = result["coverage"][..., 0]
    return rgba


def overlay(ax: Axes, points: Extensor, mesh: Mesh, field: Extensor,
            reference_speed: float, colour: np.ndarray):
    """Draw a flat field as a transparent layer over a panel, keeping the panel's existing geometric annotations, the
    panel looking down on the plane of the points' generators x and y."""
    camera = flat_panel(ax, points.algebra)
    rgba = layer(points, mesh, field, camera, reference_speed, colour)
    shown = pinned(ax, rgba, (*ax.get_xlim(), *ax.get_ylim()))
    shown.set_zorder(3.5)
    return shown

