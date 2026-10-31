"""Strokes and markers: the edges and vertices of a mesh drawn on the GPU, in the mesh's own algebra.

A stroke is a 1-simplex and a marker a 0-simplex, drawn as a triangle is: the camera of `examples.util.wgsl.camera`
reads each vertex, and the vertex stage covers a box on the picture padded around the projected edge or point by its
half width; the fragment stage then measures, in pixels, how far each sample lies from the edge's segment or the
point, and keeps the samples within the half width. Round caps and round joins follow from that distance alone, and
the picture's samples, sixteen to a pixel on the GPU, smooth its edges. An edge may end in an arrowhead, a triangle
in its own frame along and across it, and a chain of edges may be dashed, by the distance travelled along it from
where the chain starts. In a lit 3D scene the strokes are tubes among the solids and meshes: each sample's distance
from the segment gives the tube's normal, from the camera's own planes, and how far its surface comes toward the
viewer, so tubes cross the other surfaces where they truly do. Nothing of it depends on the scene's dimension or
signature: the same program draws the edges of a polyline in the plane, of a graph in space or of a world line in a
spacetime diagram. The topology is a `Mesh`: its edges are the strokes, whatever its dimension, so a polyline is a
chain of edges and a surface's wireframe its own edges.
"""
from __future__ import annotations

from enum import IntEnum
from functools import cache

import numpy as np
from matplotlib.axes import Axes

from numga.extensor import Extensor
from examples.util.wgsl import Shader, meshes, pinned
from examples.util.wgsl.camera import Camera, flat_panel, placing
from examples.util.wgsl.library import samples_per_side

Mesh = meshes.Mesh


class Cap(IntEnum):
    """How a stroke's dashes and its open ends end: square at their ends, square half the stroke's width past them, or
    round. Joins are round."""
    BUTT = 0
    PROJECTING = 1
    ROUND = 2


class Marker(IntEnum):
    """The outline of a marker about its point: a circle, a square along the picture's rows and columns, or that square
    turned an eighth of a turn."""
    DISC = 0
    SQUARE = 1
    DIAMOND = 2

stroke_source = """
var<storage, read> points: array<Point>;
var<storage, read> edges: array<vec2<u32>>;
var<storage, read> colours: array<vec3<f32>>;
// Each edge's distance from the start of its chain, in samples; whether it ends in a head, one or zero; whether its
// start and its end are open, ends of the strokes rather than joins, one or zero.
var<storage, read> travelled: array<f32>;
var<storage, read> heads: array<f32>;
var<storage, read> open_ends: array<vec2<f32>>;
// The picture's samples across and down, the strokes' half width, the heads' length and half width, and the dashes'
// length and the gaps', all in samples; a gap of zero draws the strokes whole.
var<uniform> samples: vec2<f32>;
var<uniform> radius: f32;
var<uniform> head: vec2<f32>;
var<uniform> dashes: vec2<f32>;
// How the dashes and open ends end, a `Cap`; and how much the strokes cover.
var<uniform> cap: f32;
var<uniform> opacity: f32;

struct Corner {
    @builtin(position) position: vec4<f32>,
    // The edge's ends on the picture, in samples from its top left; its colour; its distance along its chain and
    // whether it ends in a head; its ends' depths along the view.
    @location(0) @interpolate(flat) ends: vec4<f32>,
    @location(1) @interpolate(flat) colour: vec3<f32>,
    @location(2) @interpolate(flat) chain: vec2<f32>,
    @location(3) @interpolate(flat) depths: vec2<f32>,
    // The edge's run in the camera's frame, across, up and toward the viewer, in the scene's units; and which edge.
    @location(4) @interpolate(flat) run: vec3<f32>,
    @location(5) @interpolate(flat) edge: u32,
}

// A place in clip space as samples from the picture's top left.
fn sampled(place: vec4<f32>) -> vec2<f32> {
    return vec2<f32>(place.x + 1.0, 1.0 - place.y) * 0.5 * samples;
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
    let reach = max(radius, head.y) + 1.0;
    let corner = array<vec2<f32>, 6>(vec2<f32>(0, -1), vec2<f32>(1, -1), vec2<f32>(1, 1),
                                     vec2<f32>(0, -1), vec2<f32>(1, 1), vec2<f32>(0, 1))[index % 6];
    let at = mix(from - along * reach, to + along * reach, corner.x) + across * reach * corner.y;
    let depth = mix(start.z, end.z, corner.x);
    let position = vec4<f32>(at.x / samples.x * 2.0 - 1.0, 1.0 - at.y / samples.y * 2.0, depth, 1.0);
    // Each end's depth along the view: how far it lies away from the viewer, in the scene's units.
    let depths = vec2<f32>(-(screen_toward & points[edge.x]) / weight_of(points[edge.x]),
                           -(screen_toward & points[edge.y]) / weight_of(points[edge.y]));
    let run = vec3<f32>(screen_across & points[edge.y], screen_up & points[edge.y], screen_toward & points[edge.y])
              / weight_of(points[edge.y])
            - vec3<f32>(screen_across & points[edge.x], screen_up & points[edge.x], screen_toward & points[edge.x])
              / weight_of(points[edge.x]);
    return Corner(position, vec4<f32>(from, to), colours[index / 6], vec2<f32>(travelled[index / 6], heads[index / 6]),
                  depths, run, index / 6);
}

// Where a sample lies against the edge: how far along it, a share of its length; how far across its line and on
// which side, and how far from the segment's nearest point, in samples; past which end it lies, start or end, or
// neither; and whether it is covered, by the shaft, the head or a dash.
struct Against {
    share: f32,
    across: vec2<f32>,
    away: vec2<f32>,
    past: vec2<f32>,
    covered: bool,
}

fn against(corner: Corner) -> Against {
    let from = corner.ends.xy;
    let span = corner.ends.zw - from;
    let length = sqrt(span.x * span.x + span.y * span.y);
    let along_unit = select(vec2<f32>(1.0, 0.0), span / length, length > 0.0);
    let offset = corner.position.xy - from;
    // The sample in the edge's own frame: how far along it, and how far across.
    let along = offset.x * along_unit.x + offset.y * along_unit.y;
    let across = abs(offset.y * along_unit.x - offset.x * along_unit.y);
    // A head is a triangle back from the end, its base a head's length from it, or the edge's own length where that
    // is shorter, the head shrunk to fit; the shaft stops at its base.
    let headed = corner.chain.y > 0.0;
    let head_length = min(head.x, length);
    let shaft_end = select(length, length - head_length, headed);
    let back = length - along;
    let in_head = headed && back >= 0.0 && back <= head_length && across <= head.y * back / head.x;
    // The sample's distance from the shaft's segment, in samples: from its nearest point, the ends included, which
    // rounds the joins.
    let share = clamp(along, 0.0, max(shaft_end, 0.0));
    let away = offset - along_unit * share;
    let joined = sqrt(away.x * away.x + away.y * away.y);
    let sideways_distance = abs(offset.y * along_unit.x - offset.x * along_unit.y);
    // How far along the chain the sample lies outside what is drawn: past an open end, or between dashes.
    let ends_open = open_ends[corner.edge];
    let past_start = select(0.0, -along, along < 0.0 && ends_open.x > 0.0);
    let past_end = select(0.0, along - shaft_end, along > shaft_end && ends_open.y > 0.0);
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
    let past = vec2<f32>(select(0.0, 1.0, along < 0.0), select(0.0, 1.0, along > shaft_end));
    return Against(select(0.0, share / length, length > 0.0), sideways, away, past, in_head || in_shaft);
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
    let open = sample.past.x * ends_open.x + sample.past.y * ends_open.y > 0.0;
    let offset = select(sample.across, sample.away, open);
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
    let unit = select(select(seen, square / size, size > 1e-6), seen, open);
    let normal = screen_across * unit.x + screen_up * unit.y + screen_toward * unit.z;
    let direction = -screen_toward.dual();
    // The depth along the view, the tube's surface nearer the viewer by its bulge.
    let along_view = mix(corner.depths.x, corner.depths.y, sample.share) - toward * radius / oversampling * scene_pixel;
    return Shade(vec4<f32>(lit(corner.colour, normal, direction), 1.0), vec4<f32>(along_view, 0.0, 0.0, 1.0),
                 0.5 + along_view / screen_depth);
}
"""

marker_source = """
var<storage, read> points: array<Point>;
var<storage, read> colours: array<vec3<f32>>;
// How far from its point each marker reaches, out to its outer radius from its inner one, in samples, filled from
// zero; the picture's samples across and down; the markers' outline, a `Marker`; and how much they cover.
var<storage, read> reaches: array<vec2<f32>>;
var<uniform> samples: vec2<f32>;
var<uniform> marker: f32;
var<uniform> opacity: f32;

struct Corner {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) middle: vec2<f32>,
    @location(1) @interpolate(flat) colour: vec3<f32>,
    @location(2) @interpolate(flat) reach: vec2<f32>,
}

@vertex
fn place(@builtin(vertex_index) index: u32) -> Corner {
    let point = clip(points[index / 6]);
    let middle = vec2<f32>(point.x + 1.0, 1.0 - point.y) * 0.5 * samples;
    let corner = array<vec2<f32>, 6>(vec2<f32>(-1, -1), vec2<f32>(1, -1), vec2<f32>(1, 1),
                                     vec2<f32>(-1, -1), vec2<f32>(1, 1), vec2<f32>(-1, 1))[index % 6];
    // A box about the point reaching the diamond's corners, the farthest any outline reaches.
    let reach = reaches[index / 6];
    let at = middle + corner * (reach.x * 1.4142136 + 1.0);
    let position = vec4<f32>(at.x / samples.x * 2.0 - 1.0, 1.0 - at.y / samples.y * 2.0, point.z, 1.0);
    return Corner(position, middle, colours[index / 6], reach);
}

@fragment
fn shade(corner: Corner) -> @location(0) vec4<f32> {
    let away = abs(corner.position.xy - corner.middle);
    // The distance from the point by the marker's outline: round, square, or the square turned an eighth.
    let distance = select(select(sqrt(away.x * away.x + away.y * away.y), max(away.x, away.y), marker == 1.0),
                          (away.x + away.y) * 0.70710678, marker == 2.0);
    if (distance > corner.reach.x || distance < corner.reach.y) { discard; }
    return vec4<f32>(corner.colour * opacity, opacity);
}
"""


@cache
def stroke_shader(algebra) -> Shader:
    """The strokes of edges whose points belong to an algebra, placed by that algebra's camera."""
    types = dict(Point=algebra.gatype.antivector(), Plane=algebra.gatype.vector())
    return Shader(placing() + stroke_source + flat_stroke, algebra, types, {}, targets=("rgba8unorm",))


@cache
def tube_shader() -> Shader:
    """The strokes of a lit 3D scene as tubes, in the solids' algebra, whose lamp lights them: the colour a byte a
    channel and the depth alone at full precision, as the meshes write them."""
    from examples.util.wgsl import solids
    return Shader(solids.lamp_shading() + placing() + stroke_source + tube_stroke, solids.ga,
                  dict(Point=solids.Point, Plane=solids.Plane), {}, targets=("rgba8unorm", "r32float"))


@cache
def marker_shader(algebra) -> Shader:
    """The markers of points of an algebra, placed by that algebra's camera."""
    types = dict(Point=algebra.gatype.antivector(), Plane=algebra.gatype.vector())
    return Shader(placing() + marker_source, algebra, types, {}, targets=("rgba8unorm",))


def straight(image: np.ndarray) -> np.ndarray:
    """Colour and coverage from samples averaged as colour weighted by coverage: the colour divided back, both within the
    unit its rounding may overshoot."""
    coverage = image[..., 3:]
    return np.minimum(np.concatenate((image[..., :3] / np.where(coverage > 0, coverage, 1), coverage), axis=-1), 1)


def travelled(points: Extensor, edges: np.ndarray, camera: Camera) -> np.ndarray:
    """Each edge's distance along its chain from where the chain starts, in pixels `[edges]`: the edges in their order,
    a chain starting again wherever an edge does not start where the one before it ends. Read at the hand-off to the
    picture, by the camera's planes, as the shader reads them."""
    readings = (np.stack([(camera.across & points).to_array(), (camera.up & points).to_array()], axis=-1)
                / ((camera.weight & points).to_array() + camera.unit)[..., None]).reshape(-1, 2) / camera.pixel
    lengths = np.linalg.norm(readings[edges[:, 1]] - readings[edges[:, 0]], axis=-1)
    starts = np.ones(len(edges), dtype=bool)
    starts[1:] = edges[1:, 0] != edges[:-1, 1]
    totals = np.cumsum(lengths)
    return totals - lengths - np.maximum.accumulate(np.where(starts, totals - lengths, 0))


def strokes(points: Extensor, mesh: Mesh, colours: np.ndarray, width: float, camera: Camera,
            heads: np.ndarray = np.zeros(1), head: float = 0.0, dashes: tuple[float, float] = (1.0, 0.0),
            cap: Cap = Cap.ROUND, opacity: float = 1.0) -> np.ndarray:
    """The mesh's edges as strokes `width` pixels wide over its points, each in its colour `[edges, 3]`, read by the
    camera: `[rows, columns, 4]`, colour and coverage, clear where no stroke lies. The edges where `heads` `[edges]` is
    one end in a head `head` pixels long and as wide as its length; `dashes`, a dash's length and a gap's in pixels,
    breaks each chain of edges into dashes where the gap is not zero, each dash and each open end ended by the cap;
    `opacity` how much the strokes cover."""
    side = samples_per_side()
    rows, columns = camera.shape
    edges = np.asarray(mesh.edges, dtype=np.uint32)
    # No edges, such as a trail not yet begun, draw nothing: a clear picture.
    if len(edges) == 0:
        return np.zeros(camera.shape + (4,))
    image, = stroke_shader(points.algebra).draw_averaged(
        camera.shape, side, 6 * len(edges), points=points, edges=edges,
        colours=np.broadcast_to(colours, (len(edges), 3)),
        travelled=travelled(points, edges, camera) * side, heads=np.broadcast_to(heads, len(edges)).astype(float),
        open_ends=np.isin(edges, mesh.exterior(0)).astype(float),
        samples=np.array([columns * side, rows * side], dtype=float), radius=width * side / 2,
        head=np.array([head, head / 2]) * side, dashes=np.array(dashes) * side, cap=float(cap), opacity=opacity,
        **camera.uniforms(1.0))
    return straight(image)


def view_camera(view) -> Camera:
    """The camera of a solids view, its planes the view's camera turning the solids' planes x, y and z, its middle the
    view's own middle."""
    from examples.util.wgsl import solids
    mv = solids.mv
    return Camera(view.camera >> mv.x, view.camera >> mv.y, view.camera >> mv.z, mv.w, 0.0, np.zeros(2), view.half,
                  view.pixels)


def tubes(points: Extensor, mesh: Mesh, colours: np.ndarray, width: float, view, gloss: float = 0.0,
          heads: np.ndarray = np.zeros(1), head: float = 0.0, dashes: tuple[float, float] = (1.0, 0.0)):
    """The mesh's edges as tubes `width` pixels across among the solids and meshes of a lit 3D scene, its points of the
    solids' algebra, each edge in its colour `[edges, 3]`, at every sample of the view's pixels, for `solids.draw` and
    `Scene` to compose as they compose meshes; heads and dashes as `strokes` draws them."""
    from examples.util.wgsl import solids
    from examples.util.wgsl.meshes import Samples
    camera = view_camera(view)
    side = samples_per_side()
    rows, columns = camera.shape
    edges = np.asarray(mesh.edges, dtype=np.uint32)
    return Samples(tube_shader(), 6 * len(edges), dict(
        points=points, edges=edges, colours=np.broadcast_to(colours, (len(edges), 3)),
        travelled=travelled(points, edges, camera) * side, heads=np.broadcast_to(heads, len(edges)).astype(float),
        samples=np.array([columns * side, rows * side], dtype=float), radius=width * side / 2,
        head=np.array([head, head / 2]) * side, dashes=np.array(dashes) * side, scene_pixel=view.pixel,
        oversampling=float(side), cap=float(Cap.ROUND), opacity=1.0,
        open_ends=np.isin(edges, mesh.exterior(0)).astype(float),
        **camera.uniforms(8 * view.half.max()), **solids.lighting(view.camera, gloss)))


def markers(points: Extensor, colours: np.ndarray, outer: np.ndarray, camera: Camera, inner: np.ndarray = np.zeros(1),
            marker: Marker = Marker.DISC, opacity: float = 1.0) -> np.ndarray:
    """Markers about the points `[points]`, each in its colour `[points, 3]`, covering from its `inner` to its `outer`
    pixels `[points]` from its point by the marker's outline, filled from zero, a ring further out, read by the camera:
    `[rows, columns, 4]`, colour and coverage, clear where no marker lies."""
    side = samples_per_side()
    rows, columns = camera.shape
    # No points draw nothing: a clear picture.
    if len(points) == 0:
        return np.zeros(camera.shape + (4,))
    image, = marker_shader(points.algebra).draw_averaged(
        camera.shape, side, 6 * len(points), points=points, colours=np.broadcast_to(colours, (len(points), 3)),
        reaches=np.stack([np.broadcast_to(outer, len(points)), np.broadcast_to(inner, len(points))], axis=-1) * side,
        samples=np.array([columns * side, rows * side], dtype=float), marker=float(marker), opacity=opacity, **camera.uniforms(1.0))
    return straight(image)


def composited(pictures: list[np.ndarray]) -> np.ndarray:
    """Pictures of colour and coverage `[rows, columns, 4]`, each laid over the ones before: one such picture."""
    colour, coverage = np.zeros(pictures[0].shape[:2] + (3,)), np.zeros(pictures[0].shape[:2] + (1,))
    for picture in pictures:
        colour = colour * (1 - picture[..., 3:]) + picture[..., :3] * picture[..., 3:]
        coverage = coverage * (1 - picture[..., 3:]) + picture[..., 3:]
    return np.minimum(np.concatenate((colour / np.where(coverage > 0, coverage, 1), coverage), axis=-1), 1)


def overlay(ax: Axes, pictures: list[np.ndarray]):
    """Pictures of a panel, as its camera draws them, laid over each other in order and pinned into the panel at its
    own pixels, clear where none covers; the image, for an animation to set its data."""
    return pinned(ax, composited(pictures), (*ax.get_xlim(), *ax.get_ylim()))


def stroke_overlay(ax: Axes, points: Extensor, mesh: Mesh, colours: np.ndarray, width: float):
    """The mesh's edges drawn as strokes into a panel at its own pixels, the panel looking down on the plane of the
    points' generators x and y; the image, for an animation to set its data."""
    camera = flat_panel(ax, points.algebra)
    return pinned(ax, strokes(points, mesh, colours, width, camera), (*ax.get_xlim(), *ax.get_ylim()))
