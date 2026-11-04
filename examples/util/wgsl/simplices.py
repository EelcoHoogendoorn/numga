"""Strokes and markers: the edges and vertices of a mesh drawn on the GPU, in the mesh's own algebra.

A stroke is a 1-simplex and a marker a 0-simplex, drawn as a triangle is: the camera of `examples.util.wgsl.camera`
reads each vertex, and the vertex stage covers a box on the picture padded around the projected edge or point by its
half width; the fragment stage then measures, in pixels, how far each sample lies from the edge's segment or the
point, and keeps the samples within the half width. Round caps and round joins follow from that distance alone, and
the picture's samples, sixteen to a pixel on the GPU, smooth its edges. A chain of edges may be dashed, by the
distance travelled along it from where the chain starts, and an edge may end in a chevron, two short strokes of its
own width back from its end, laid out on the picture. Strokes and dots are the whole vocabulary: an arrow is a stroke
with a chevron or a dot at its tip, a square four strokes. In a lit 3D scene the strokes are tubes among the solids and meshes: each sample's distance
from the segment gives the tube's normal, from the camera's own planes, and how far its surface comes toward the
viewer, so tubes cross the other surfaces where they truly do. Nothing of it depends on the scene's dimension or
signature: the same program draws the edges of a polyline in the plane, of a graph in space or of a world line in a
spacetime diagram. The topology is a `Topology`: its edges are the strokes, whatever its dimension, so a polyline is a
chain of edges and a surface's wireframe its own edges.
"""
from __future__ import annotations

from enum import IntEnum
from typing import Any
from functools import cache

import numpy as np
from matplotlib.axes import Axes

from numga.extensor import Extensor
from numga.gatype import GAType
from examples.util.wgsl import Shader, meshes, pinned
from examples.util.wgsl.runner import device
from examples.util.wgsl.camera import Camera, placing, read, reading
from examples.util.wgsl.library import samples_per_side

Topology = meshes.Topology


class Cap(IntEnum):
    """How a stroke's dashes and its open ends end: square at their ends, square half the stroke's width past them, or
    round. Joins are round."""
    BUTT = 0
    PROJECTING = 1
    ROUND = 2



stroke_source = """
var<storage, read> points: array<Point>;
var<storage, read> edges: array<vec2<u32>>;
var<storage, read> colours: array<vec3<f32>>;
// Each edge's distance from the start of its chain, in samples; whether it ends in a chevron, one or zero; whether
// its start and its end are open, ends of the strokes rather than joins, one or zero; whether it starts where the edge
// before it ends, one or zero, what that edge covers about the join that edge's to draw, so that a translucent chain
// covers every sample once.
var<storage, read> travelled: array<f32>;
var<storage, read> chevrons: array<f32>;
var<storage, read> open_ends: array<vec2<f32>>;
var<storage, read> follows: array<f32>;
// The picture's samples across and down, the strokes' half width, a chevron's stroke's reach back along its edge and
// out to its side, and the dashes' length and the gaps', all in samples; a gap of zero draws the strokes whole.
var<uniform> samples: vec2<f32>;
var<uniform> radius: f32;
var<uniform> chevron: vec2<f32>;
var<uniform> dashes: vec2<f32>;
// How the dashes and open ends end, a `Cap`; and how much the strokes cover.
var<uniform> cap: f32;
var<uniform> opacity: f32;

struct Corner {
    @builtin(position) position: vec4<f32>,
    // The edge's ends on the picture, in samples from its top left; its colour; its distance along its chain and
    // whether it ends in a chevron; its ends' depths along the view.
    @interpolate(flat) ends: vec4<f32>,
    @interpolate(flat) colour: vec3<f32>,
    @interpolate(flat) chain: vec2<f32>,
    @interpolate(flat) depths: vec2<f32>,
    // The edge's run in the camera's frame, across, up and toward the viewer, in the scene's units; and which edge.
    @interpolate(flat) run: vec3<f32>,
    @interpolate(flat) edge: u32,
}

// A place in clip space as samples from the picture's top left.
fn sampled(place: vec4<f32>) -> vec2<f32> {
    return vec2<f32>(place.x + 1.0, 1.0 - place.y) * 0.5 * samples;
}

// Whether a sample, in samples from the picture's top left, lies within the half width of an edge's segment.
fn within(edge: vec2<u32>, position: vec2<f32>) -> bool {
    let from = sampled(clip(points[edge.x]));
    let span = sampled(clip(points[edge.y])) - from;
    let offset = position - from;
    let share = clamp((offset.x * span.x + offset.y * span.y) / max(span.x * span.x + span.y * span.y, 1e-12), 0.0, 1.0);
    let away = offset - span * share;
    return away.x * away.x + away.y * away.y <= radius * radius;
}

@vertex
fn place(@builtin(vertex_index) index: u32) -> Corner {
    let edge = edges[index / 6];
    let start = clip(points[edge.x]);
    let end = clip(points[edge.y]);
    let from = sampled(start);
    let to = sampled(end);
    // The box around the edge, a half width and a sample beyond it every way: along the edge and across it.
    let span = to - from;
    let length = sqrt(span.x * span.x + span.y * span.y);
    let along = select(vec2<f32>(1.0, 0.0), span / length, length > 0.0);
    let across = vec2<f32>(-along.y, along.x);
    let reach = radius + chevron.y * chevrons[index / 6] + 1.0;
    let corner = array<vec2<f32>, 6>(vec2<f32>(0, -1), vec2<f32>(1, -1), vec2<f32>(1, 1),
                                     vec2<f32>(0, -1), vec2<f32>(1, 1), vec2<f32>(0, 1))[index % 6];
    let at = mix(from - along * reach, to + along * reach, corner.x) + across * reach * corner.y;
    let depth = mix(start.z, end.z, corner.x);
    let position = vec4<f32>(at.x / samples.x * 2.0 - 1.0, 1.0 - at.y / samples.y * 2.0, depth, 1.0);
    // Each end's depth along the view: how far it lies away from the viewer, in the scene's units.
    let depths = vec2<f32>(-paired(screen_toward, points[edge.x]) / weight_of(points[edge.x]),
                           -paired(screen_toward, points[edge.y]) / weight_of(points[edge.y]));
    let run = vec3<f32>(paired(screen_across, points[edge.y]), paired(screen_up, points[edge.y]),
                        paired(screen_toward, points[edge.y])) / weight_of(points[edge.y])
            - vec3<f32>(paired(screen_across, points[edge.x]), paired(screen_up, points[edge.x]),
                        paired(screen_toward, points[edge.x])) / weight_of(points[edge.x]);
    return Corner(position, vec4<f32>(from, to), colours[index / 6], vec2<f32>(travelled[index / 6], chevrons[index / 6]),
                  depths, run, index / 6);
}

// Where a sample lies against the edge: how far along it, a share of its length; how far across its line and on
// which side, and how far from the segment's nearest point, in samples; past which end it lies, start or end, or
// neither; whether it is covered, by the stroke, a dash or the chevron; and whether by the chevron alone, and how far
// from the nearer of its strokes.
struct Against {
    share: f32,
    across: vec2<f32>,
    away: vec2<f32>,
    past: vec2<f32>,
    covered: bool,
    barbed: bool,
    barb: vec2<f32>,
}

// A sample's offset from the nearest point of a segment, in samples.
fn off_segment(offset: vec2<f32>, span: vec2<f32>) -> vec2<f32> {
    let share = clamp((offset.x * span.x + offset.y * span.y) / max(span.x * span.x + span.y * span.y, 1e-12), 0.0, 1.0);
    return offset - span * share;
}

fn against(corner: Corner) -> Against {
    let from = corner.ends.xy;
    let span = corner.ends.zw - from;
    let length = sqrt(span.x * span.x + span.y * span.y);
    let along_unit = select(vec2<f32>(1.0, 0.0), span / length, length > 0.0);
    let offset = corner.position.xy - from;
    // The sample in the edge's own frame: how far along it, and how far across.
    let along = offset.x * along_unit.x + offset.y * along_unit.y;
    // The sample's distance from the edge's segment, in samples: from its nearest point, the ends included, which
    // rounds the joins.
    let share = clamp(along, 0.0, length);
    let away = offset - along_unit * share;
    let joined = sqrt(away.x * away.x + away.y * away.y);
    let sideways_distance = abs(offset.y * along_unit.x - offset.x * along_unit.y);
    // How far along the chain the sample lies outside what is drawn: past an open end, or between dashes.
    let ends_open = open_ends[corner.edge];
    let past_start = select(0.0, -along, along < 0.0 && ends_open.x > 0.0);
    let past_end = select(0.0, along - length, along > length && ends_open.y > 0.0);
    let travel = corner.chain.x + clamp(along, 0.0, length);
    let period = dashes.x + dashes.y;
    let phase = travel - period * floor(travel / period);
    let between = select(0.0, min(phase - dashes.x, period - phase), dashes.y > 0.0 && phase > dashes.x);
    let outside = max(max(past_start, past_end), between);
    // Within a dash, the round joins; outside it, the cap: round, square past the end, or none.
    let capped = select(select(false, outside <= radius && sideways_distance <= radius, cap == 1.0),
                        outside * outside + sideways_distance * sideways_distance <= radius * radius, cap == 2.0);
    let in_shaft = select(capped, joined <= radius, outside <= 0.0);
    let sideways = offset - along_unit * along;
    let past = vec2<f32>(select(0.0, 1.0, along < 0.0), select(0.0, 1.0, along > length));
    let owned = !(follows[corner.edge] > 0.0 && within(edges[corner.edge - 1u], corner.position.xy));
    // The chevron: two strokes from the edge's end, back along it and out to either side, each as wide as the edge,
    // shrunk to fit an edge shorter than they are long.
    let to_end = corner.position.xy - corner.ends.zw;
    let side = vec2<f32>(-along_unit.y, along_unit.x);
    let fit = min(1.0, length / max(sqrt(chevron.x * chevron.x + chevron.y * chevron.y), 1e-6));
    let left = off_segment(to_end, (-along_unit * chevron.x + side * chevron.y) * fit);
    let right = off_segment(to_end, (-along_unit * chevron.x - side * chevron.y) * fit);
    let barb = select(right, left, dot(left, left) < dot(right, right));
    let in_barb = corner.chain.y > 0.0 && dot(barb, barb) <= radius * radius;
    return Against(select(0.0, share / length, length > 0.0), sideways, away, past, (in_shaft || in_barb) && owned,
                   in_barb && !in_shaft, barb);
}
"""

flat_stroke = """
// The colour weighted by the strokes' opacity, as the picture's samples average it.
@fragment
fn shade(corner: Corner) -> @location(0) vec4<f32> {
    if (!against(corner).covered) { discard; }
    return vec4<f32>(corner.colour * opacity, opacity);
}
"""

# A stroke in a lit 3D scene: a tube, its colour lit through its normal, its depth along the view where it shows, as
# the meshes write theirs for the tracer to compose.
tube_stroke = """
// A pixel's width in the scene's units, at the picture's own pixels; the samples a pixel; whether each edge's start and
// end are open, ends of the strokes rather than joins, one or zero.
var<uniform> scene_pixel: f32;
var<uniform> oversampling: f32;

struct Shade {
    @location(0) colour: vec4<f32>,
    @location(1) depth: vec4<f32>,
    @builtin(frag_depth) nearness: f32,
}

@fragment
fn shade(corner: Corner) -> Shade {
    let sample = against(corner);
    if (!sample.covered) { discard; }
    // Past an open end, a ball about it: the normal from the sample's place against the end point. Elsewhere the
    // tube's normal from the sample's place across its line, not from the nearest point of its segment, so that a
    // chain of edges shades as one tube through its joints. Either way across and up the picture toward the sample,
    // rows counted down, and toward the viewer by what the radius leaves.
    let ends_open = open_ends[corner.edge];
    // A chevron's stroke shades as a ball does, about its nearest point.
    let open = sample.past.x * ends_open.x + sample.past.y * ends_open.y > 0.0 || sample.barbed;
    let offset = select(select(sample.across, sample.away, open), sample.barb, sample.barbed);
    let distance = sqrt(offset.x * offset.x + offset.y * offset.y);
    let across = min(distance / radius, 1.0);
    let toward = sqrt(1.0 - across * across);
    let side = select(vec2<f32>(0.0), offset / max(distance, 1e-6), distance > 0.0) * across;
    // On the tube, made square to the edge's run in space, so that an edge running into depth turns its face as a
    // tube does; on a ball, left as it is.
    let seen = vec3<f32>(side.x, -side.y, toward);
    let run = corner.run;
    let run_squared = max(run.x * run.x + run.y * run.y + run.z * run.z, 1e-12);
    let along_run = (seen.x * run.x + seen.y * run.y + seen.z * run.z) / run_squared;
    let square = seen - run * along_run;
    let size = sqrt(square.x * square.x + square.y * square.y + square.z * square.z);
    let normal = select(select(seen, square / size, size > 1e-6), seen, open);
    // The depth along the view, the tube's surface nearer the viewer by its bulge; a chevron, laid out on the picture,
    // nearer by its reach too, so that it shows whole over a surface its edge lies on.
    let bulge = toward * radius + select(0.0, chevron.x, sample.barbed);
    let along_view = mix(corner.depths.x, corner.depths.y, sample.share) - bulge / oversampling * scene_pixel;
    return Shade(vec4<f32>(lit_seen(corner.colour, normal), 1.0), vec4<f32>(along_view, 0.0, 0.0, 1.0),
                 0.5 + along_view / screen_depth);
}
"""

marker_source = """
var<storage, read> points: array<Point>;
var<storage, read> colours: array<vec3<f32>>;
// Each dot's radius, in samples; the picture's samples across and down; and how much the dots cover.
var<storage, read> radii: array<f32>;
var<uniform> samples: vec2<f32>;
var<uniform> opacity: f32;

struct Corner {
    @builtin(position) position: vec4<f32>,
    @interpolate(flat) middle: vec2<f32>,
    @interpolate(flat) colour: vec3<f32>,
    @interpolate(flat) radius: f32,
}

@vertex
fn place(@builtin(vertex_index) index: u32) -> Corner {
    let point = clip(points[index / 6]);
    let middle = vec2<f32>(point.x + 1.0, 1.0 - point.y) * 0.5 * samples;
    let corner = array<vec2<f32>, 6>(vec2<f32>(-1, -1), vec2<f32>(1, -1), vec2<f32>(1, 1),
                                     vec2<f32>(-1, -1), vec2<f32>(1, 1), vec2<f32>(-1, 1))[index % 6];
    // A box about the point, a sample beyond its radius every way.
    let radius = radii[index / 6];
    let at = middle + corner * (radius + 1.0);
    let position = vec4<f32>(at.x / samples.x * 2.0 - 1.0, 1.0 - at.y / samples.y * 2.0, point.z, 1.0);
    return Corner(position, middle, colours[index / 6], radius);
}

@fragment
fn shade(corner: Corner) -> @location(0) vec4<f32> {
    let away = corner.position.xy - corner.middle;
    if (away.x * away.x + away.y * away.y > corner.radius * corner.radius) { discard; }
    return vec4<f32>(corner.colour * opacity, opacity);
}
"""


@cache
def stroke_shader(Point: GAType, Plane: GAType) -> Shader:
    """The strokes of edges whose points are of the type `Point`, placed by a camera's planes of the type `Plane`."""
    return Shader(placing(Plane) + stroke_source + flat_stroke, Point.algebra, dict(Point=Point, Plane=Plane), {},
                  targets=("rgba8unorm",), over=True)


@cache
def tube_shader() -> Shader:
    """The strokes of a lit 3D scene as tubes, in the solids' algebra, whose lamp lights them: the colour a byte a
    channel and the depth alone at full precision, as the meshes write them."""
    from examples.util.wgsl import solids
    return Shader(solids.lamp_shading() + placing(solids.Plane) + stroke_source + tube_stroke, solids.ga,
                  dict(Point=solids.Point, Plane=solids.Plane), {}, targets=("rgba8unorm", "r32float"))


@cache
def marker_shader(Point: GAType, Plane: GAType) -> Shader:
    """The dots of points of the type `Point`, placed by a camera's planes of the type `Plane`."""
    return Shader(placing(Plane) + marker_source, Point.algebra, dict(Point=Point, Plane=Plane), {},
                  targets=("rgba8unorm",), over=True)


def chevron_reach(length: float) -> np.ndarray:
    """A chevron's stroke `length` long, a third of a right angle off its edge: how far it reaches back along the edge
    and out to its side."""
    return length * np.array([np.sqrt(3) / 2, 1 / 2])


def following(edges: np.ndarray) -> np.ndarray:
    """Whether each edge `[edges]` starts where the edge before it ends, one or zero: a join of a chain."""
    follows = np.zeros(len(edges))
    follows[1:] = edges[1:, 0] == edges[:-1, 1]
    return follows


def travelled(points: Extensor, edges: np.ndarray, camera: Camera) -> np.ndarray:
    """Each edge's distance along its chain from where the chain starts, in pixels `[edges]`: the edges in their order,
    a chain starting again wherever an edge does not start where the one before it ends. Read at the hand-off to the
    picture, by the camera's planes, as the shader reads them."""
    readings = (np.stack([read(camera.across, points).to_array(), read(camera.up, points).to_array()], axis=-1)
                / (read(camera.weight, points).to_array() + camera.unit)[..., None]).reshape(-1, 2) / camera.pixel
    lengths = np.linalg.norm(readings[edges[:, 1]] - readings[edges[:, 0]], axis=-1)
    starts = np.ones(len(edges), dtype=bool)
    starts[1:] = edges[1:, 0] != edges[:-1, 1]
    totals = np.cumsum(lengths)
    return totals - lengths - np.maximum.accumulate(np.where(starts, totals - lengths, 0))


def strokes(points: Extensor, topology: Topology, colours: np.ndarray, width: float, camera: Camera,
            chevrons: np.ndarray = np.zeros(1), chevron: float = 0.0, dashes: tuple[float, float] = (1.0, 0.0),
            cap: Cap = Cap.ROUND, opacity: float = 1.0) -> Layer:
    """The mesh's edges as strokes `width` pixels wide over its points, each in its colour `[edges, 3]`, read by the
    camera: a layer of colour and coverage, clear where no stroke lies. The edges where `chevrons` `[edges]`
    is one end in a chevron, two strokes `chevron` pixels long back from the end, a third of a right angle to either
    side; `dashes`, a dash's length and a gap's in pixels, breaks each chain of edges into dashes where the gap is not
    zero, each dash and each open end ended by the cap; `opacity` how much the strokes cover."""
    side = samples_per_side()
    rows, columns = camera.shape
    edges = np.asarray(topology.elements[1], dtype=np.uint32)
    # No edges, such as a trail not yet begun, draw nothing: a clear picture.
    if len(edges) == 0:
        return np.zeros(camera.shape + (4,))
    return Layer(stroke_shader(**reading(points, camera)), camera.shape, 6 * len(edges), dict(
        points=points, edges=edges,
        colours=np.broadcast_to(colours, (len(edges), 3)),
        travelled=travelled(points, edges, camera) * side, chevrons=np.broadcast_to(chevrons, len(edges)).astype(float),
        open_ends=np.isin(edges, topology.exterior[0]).astype(float),
        follows=following(edges),
        samples=np.array([columns * side, rows * side], dtype=float), radius=width * side / 2,
        chevron=chevron_reach(chevron) * side, dashes=np.array(dashes) * side, cap=float(cap), opacity=opacity,
        **camera.uniforms(1.0)))


def view_camera(view) -> Camera:
    """The camera of a solids view, its planes the view's camera turning the solids' planes x, y and z, its middle the
    view's own middle."""
    from examples.util.wgsl import solids
    mv = solids.mv
    return Camera(view.camera >> mv.x, view.camera >> mv.y, view.camera >> mv.z, mv.w, 0.0, np.zeros(2), view.half,
                  view.pixels)


def tubes(points: Extensor, topology: Topology, colours: np.ndarray, width: float, view, gloss: float = 0.0,
          chevrons: np.ndarray = np.zeros(1), chevron: float = 0.0, dashes: tuple[float, float] = (1.0, 0.0)):
    """The mesh's edges as tubes `width` pixels across among the solids and meshes of a lit 3D scene, its points of the
    solids' algebra, each edge in its colour `[edges, 3]`, at every sample of the view's pixels, for `solids.draw` and
    `Scene` to compose as they compose meshes; chevrons and dashes as `strokes` draws them, a chevron on the
    picture, facing the viewer."""
    from examples.util.wgsl import solids
    from examples.util.wgsl.meshes import Samples
    camera = view_camera(view)
    side = samples_per_side()
    rows, columns = camera.shape
    edges = np.asarray(topology.elements[1], dtype=np.uint32)
    return Samples(tube_shader(), 6 * len(edges), dict(
        points=points, edges=edges, colours=np.broadcast_to(colours, (len(edges), 3)),
        travelled=travelled(points, edges, camera) * side, chevrons=np.broadcast_to(chevrons, len(edges)).astype(float),
        samples=np.array([columns * side, rows * side], dtype=float), radius=width * side / 2,
        chevron=chevron_reach(chevron) * side, dashes=np.array(dashes) * side, scene_pixel=view.pixel,
        oversampling=float(side), cap=float(Cap.ROUND), opacity=1.0,
        open_ends=np.isin(edges, topology.exterior[0]).astype(float),
        follows=following(edges),
        **camera.uniforms(8 * view.half.max()), **solids.lighting(gloss)))


def markers(points: Extensor, colours: np.ndarray, radii: np.ndarray, camera: Camera, opacity: float = 1.0) -> Layer:
    """Dots about the points `[points]`, each in its colour `[points, 3]` and its radius in pixels, one for all or one
    each `[points]`, read by the camera: a layer of colour and coverage, clear where no dot lies."""
    side = samples_per_side()
    rows, columns = camera.shape
    # No points draw nothing: a clear picture.
    if len(points) == 0:
        return np.zeros(camera.shape + (4,))
    return Layer(marker_shader(**reading(points, camera)), camera.shape, 6 * len(points), dict(
        points=points, colours=np.broadcast_to(colours, (len(points), 3)),
        radii=np.broadcast_to(radii, len(points)).astype(float) * side,
        samples=np.array([columns * side, rows * side], dtype=float), opacity=opacity, **camera.uniforms(1.0)))


# --- compositing ---------------------------------------------------------------------
# A picture made on the host, its colour weighted by its coverage, laid over the panel's samples: each sample takes its
# pixel's colour.
image_source = """
@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var<uniform> side: vec4<u32>;

struct Corner {
    @builtin(position) position: vec4<f32>,
}

// One triangle over the whole picture.
@vertex
fn place(@builtin(vertex_index) index: u32) -> Corner {
    let corner = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0))[index];
    return Corner(vec4<f32>(corner, 0.5, 1.0));
}

@fragment
fn shade(corner: Corner) -> @location(0) vec4<f32> {
    return textureLoad(image, vec2<u32>(corner.position.xy) / side.x, 0);
}
"""

# Each pixel's samples averaged, colour weighted by coverage, and the colour divided back by the coverage where anything
# covers, both within the unit their rounding may overshoot.
average_source = """
@group(0) @binding(0) var samples: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> pixels: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> size: vec4<u32>;

@compute @workgroup_size(8, 8)
fn average(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    var sum = vec4<f32>(0.0);
    for (var down = 0u; down < size.z; down++) {
        for (var across = 0u; across < size.z; across++) {
            sum += textureLoad(samples, vec2<u32>(id.x * size.z + across, id.y * size.z + down), 0);
        }
    }
    let weighted = sum / f32(size.z * size.z);
    let colour = select(weighted.xyz, weighted.xyz / weighted.w, weighted.w > 0.0);
    pixels[id.y * size.x + id.x] = min(vec4<f32>(colour, weighted.w), vec4<f32>(1.0));
}
"""


class Compositor:
    """The pictures of a panel drawn on the GPU into one picture of its samples, each laid over the ones before in
    order, colour weighted by coverage, sample by sample; then each pixel's samples averaged once and the panel read
    back. A layer's draw is recorded when the panel is composited, so a layer laid in every frame of an animation is
    drawn in every frame."""

    def __init__(self) -> None:
        import wgpu
        self.device = device()
        laid_over = {"color": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"},
                     "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"}}
        module = self.device.create_shader_module(code=image_source)
        self.images = self.device.create_render_pipeline(
            layout="auto", vertex={"module": module, "entry_point": "place", "buffers": []},
            fragment={"module": module, "entry_point": "shade", "targets": [{"format": "rgba8unorm", "blend": laid_over}]},
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            depth_stencil={"format": "depth32float", "depth_write_enabled": False, "depth_compare": "always"})
        module = self.device.create_shader_module(code=average_source)
        self.averaging = self.device.create_compute_pipeline(layout="auto", compute={"module": module, "entry_point": "average"})
        self.kept = {}                                                         # dict[tuple, Any]

    def kept_buffer(self, role: tuple, n_bytes: int, usage: int) -> Any:
        """The buffer kept for a role, made again only when a call needs it larger."""
        buffer = self.kept.get(role)
        if buffer is None or buffer.size < n_bytes:
            buffer = self.device.create_buffer(size=n_bytes, usage=usage)
            self.kept[role] = buffer
        return buffer

    def surface(self, role: tuple, shape: tuple[int, int], format: str, usage: int) -> Any:
        """The texture kept for a role and a picture's shape, rows and columns, made on the first call that needs it."""
        key = role + (shape,)
        if key not in self.kept:
            self.kept[key] = self.device.create_texture(size=(shape[1], shape[0], 1), format=format, usage=usage)
        return self.kept[key]

    def image_group(self, image: np.ndarray, side: int, index: int) -> Any:
        """A picture made on the host, its colour weighted by its coverage, bound for the image pipeline."""
        import wgpu
        coverage = image[..., 3:]
        weighted = np.ascontiguousarray(np.concatenate((image[..., :3] * coverage, coverage), axis=-1), dtype=np.float32)
        rows, columns = weighted.shape[:2]
        texture = self.surface(("image", index), (rows, columns), "rgba32float",
                               wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self.device.queue.write_texture({"texture": texture}, weighted.tobytes(),
                                        {"bytes_per_row": columns * 16, "rows_per_image": rows}, (columns, rows, 1))
        sides = self.kept_buffer(("side", index), 16, wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        self.device.queue.write_buffer(sides, 0, np.array([side, 0, 0, 0], dtype=np.uint32).tobytes())
        return self.device.create_bind_group(layout=self.images.get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": texture.create_view()},
            {"binding": 1, "resource": {"buffer": sides, "offset": 0, "size": 16}}])

    def composite(self, pictures: list, size: tuple[int, int]) -> np.ndarray:
        """Pictures of a panel `rows, columns`, layers and pictures made on the host, laid over each other sample by
        sample, averaged once and read back: colour and coverage `[rows, columns, 4]`."""
        import wgpu
        rows, columns = size
        side = samples_per_side()
        shape = (rows * side, columns * side)
        # Each picture's pipeline and values, bound before the drawing is recorded, each under a key of its own.
        draws = [(picture.shader.raster, picture.shader.bind(shape, picture.values, ("layer", index))[0],
                  picture.n_vertices) if isinstance(picture, Layer) else
                 (self.images, self.image_group(np.asarray(picture), side, index), 3)
                 for index, picture in enumerate(pictures)]
        canvas = self.surface(("canvas",), shape, "rgba8unorm",
                              wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING)
        depth = self.surface(("depth",), shape, "depth32float", wgpu.TextureUsage.RENDER_ATTACHMENT)
        encoder = self.device.create_command_encoder()
        drawing = encoder.begin_render_pass(
            color_attachments=[{"view": canvas.create_view(), "clear_value": (0, 0, 0, 0), "load_op": "clear",
                                "store_op": "store"}],
            depth_stencil_attachment={"view": depth.create_view(), "depth_clear_value": 1.0,
                                      "depth_load_op": "clear", "depth_store_op": "discard"})
        for pipeline, group, n_vertices in draws:
            drawing.set_pipeline(pipeline)
            drawing.set_bind_group(0, group)
            drawing.draw(n_vertices)
        drawing.end()
        n_bytes = rows * columns * 16
        pixels = self.kept_buffer(("pixels",), n_bytes, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        counts = self.kept_buffer(("size",), 16, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
        self.device.queue.write_buffer(counts, 0, np.array([columns, rows, side, 0], dtype=np.uint32).tobytes())
        group = self.device.create_bind_group(layout=self.averaging.get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": canvas.create_view()},
            {"binding": 1, "resource": {"buffer": pixels, "offset": 0, "size": n_bytes}},
            {"binding": 2, "resource": {"buffer": counts, "offset": 0, "size": 16}}])
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(self.averaging)
        compute.set_bind_group(0, group)
        compute.dispatch_workgroups(-(-columns // 8), -(-rows // 8), 1)
        compute.end()
        readback = self.kept_buffer(("readback",), n_bytes, wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ)
        encoder.copy_buffer_to_buffer(pixels, 0, readback, 0, n_bytes)
        self.device.queue.submit([encoder.finish()])
        readback.map_sync(wgpu.MapMode.READ, 0, n_bytes)
        image = np.frombuffer(readback.read_mapped(0, n_bytes), dtype=np.float32).reshape(rows, columns, 4).copy()
        readback.unmap()
        return image


@cache
def compositor() -> Compositor:
    return Compositor()


class Layer:
    """A picture of a panel to be drawn on the GPU: a shader's triangles over the camera's pixels, `rows, columns`,
    drawn when the panel's pictures are composited, sample by sample over the ones before it. Its own colour and
    coverage `[rows, columns, 4]` are its picture composited alone, `composited([layer])`."""

    def __init__(self, shader: Shader, size: tuple[int, int], n_vertices: int, values: dict) -> None:
        self.shader, self.size, self.n_vertices, self.values = shader, size, n_vertices, values

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.size + (4,)


def composited(pictures: list) -> np.ndarray:
    """Pictures of a panel, layers drawn on the GPU or colour and coverage `[rows, columns, 4]` made on the host, each
    laid over the ones before, sample by sample: colour and coverage `[rows, columns, 4]`."""
    return compositor().composite(pictures, tuple(pictures[0].shape[:2]))


def overlay(ax: Axes, pictures: list[np.ndarray]):
    """Pictures of a panel, as its camera draws them, laid over each other in order and pinned into the panel at its
    own pixels, clear where none covers; the image, for an animation to set its data."""
    return pinned(ax, composited(pictures), (*ax.get_xlim(), *ax.get_ylim()))
