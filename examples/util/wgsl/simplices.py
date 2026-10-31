"""Strokes and markers: the edges and vertices of a mesh drawn on the GPU, in the mesh's own algebra.

A stroke is a 1-simplex and a marker a 0-simplex, drawn as a triangle is: the camera of `examples.util.wgsl.camera`
reads each vertex, and the vertex stage covers a box on the picture padded around the projected edge or point by its
half width; the fragment stage then measures, in pixels, how far each sample lies from the edge's segment or the point,
and keeps the samples within the half width. Round caps and round joins follow from that distance alone, and the
picture's samples, sixteen to a pixel on the GPU, smooth its edges. An edge may end in an arrowhead, a triangle in its
own frame along and across it, and a chain of edges may be dashed, by the distance travelled along it from where the
chain starts. Nothing of it depends on the scene's dimension or signature: the same program draws the edges of a
polyline in the plane, of a graph in space or of a world line in a spacetime diagram. The topology is a `Mesh`: its
edges are the strokes, whatever its dimension, so a polyline is a chain of edges and a surface's wireframe its own
edges.
"""
from __future__ import annotations

from functools import cache

import numpy as np
from matplotlib.axes import Axes

from numga.extensor import Extensor
from examples.util.wgsl import Shader, meshes, pinned
from examples.util.wgsl.camera import Camera, flat_panel, placing
from examples.util.wgsl.library import samples_per_side

Mesh = meshes.Mesh

stroke_source = """
var<storage, read> points: array<Point>;
var<storage, read> edges: array<vec2<u32>>;
var<storage, read> colours: array<vec3<f32>>;
// Each edge's distance from the start of its chain, in samples, and whether it ends in a head, one or zero.
var<storage, read> travelled: array<f32>;
var<storage, read> heads: array<f32>;
// The picture's samples across and down, the strokes' half width, the heads' length and half width, and the dashes'
// length and the gaps', all in samples; a gap of zero draws the strokes whole.
var<uniform> samples: vec2<f32>;
var<uniform> radius: f32;
var<uniform> head: vec2<f32>;
var<uniform> dashes: vec2<f32>;

struct Corner {
    @builtin(position) position: vec4<f32>,
    // The edge's ends on the picture, in samples from its top left; its colour; its distance along its chain and
    // whether it ends in a head.
    @location(0) @interpolate(flat) ends: vec4<f32>,
    @location(1) @interpolate(flat) colour: vec3<f32>,
    @location(2) @interpolate(flat) chain: vec2<f32>,
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
    return Corner(position, vec4<f32>(from, to), colours[index / 6], vec2<f32>(travelled[index / 6], heads[index / 6]));
}

@fragment
fn shade(corner: Corner) -> @location(0) vec4<f32> {
    let from = corner.ends.xy;
    let span = corner.ends.zw - from;
    let length = sqrt(span.x * span.x + span.y * span.y);
    let along_unit = select(vec2<f32>(1.0, 0.0), span / length, length > 0.0);
    let offset = corner.position.xy - from;
    // The sample in the edge's own frame: how far along it, and how far across.
    let along = offset.x * along_unit.x + offset.y * along_unit.y;
    let across = abs(offset.y * along_unit.x - offset.x * along_unit.y);
    // A head is a triangle back from the end, its base a head's length from it; the shaft stops at its base.
    let headed = corner.chain.y > 0.0;
    let shaft_end = select(length, length - head.x, headed);
    let back = length - along;
    let in_head = headed && back >= 0.0 && back <= head.x && across <= head.y * back / head.x;
    // The sample's distance from the shaft's segment, in samples: from its nearest point, the ends included.
    let share = clamp(along, 0.0, max(shaft_end, 0.0));
    let away = offset - along_unit * share;
    let in_shaft = away.x * away.x + away.y * away.y <= radius * radius;
    // Dashes by the distance along the chain, where there are gaps.
    let travel = corner.chain.x + clamp(along, 0.0, length);
    let period = dashes.x + dashes.y;
    let dashed = dashes.y <= 0.0 || travel - period * floor(travel / period) <= dashes.x;
    if (!(in_head || (in_shaft && dashed))) { discard; }
    return vec4<f32>(corner.colour, 1.0);
}
"""

marker_source = """
var<storage, read> points: array<Point>;
var<storage, read> colours: array<vec3<f32>>;
var<uniform> samples: vec2<f32>;
var<uniform> radius: f32;

struct Corner {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) middle: vec2<f32>,
    @location(1) @interpolate(flat) colour: vec3<f32>,
}

@vertex
fn place(@builtin(vertex_index) index: u32) -> Corner {
    let point = clip(points[index / 6]);
    let middle = vec2<f32>(point.x + 1.0, 1.0 - point.y) * 0.5 * samples;
    let corner = array<vec2<f32>, 6>(vec2<f32>(-1, -1), vec2<f32>(1, -1), vec2<f32>(1, 1),
                                     vec2<f32>(-1, -1), vec2<f32>(1, 1), vec2<f32>(-1, 1))[index % 6];
    let at = middle + corner * (radius + 1.0);
    let position = vec4<f32>(at.x / samples.x * 2.0 - 1.0, 1.0 - at.y / samples.y * 2.0, point.z, 1.0);
    return Corner(position, middle, colours[index / 6]);
}

@fragment
fn shade(corner: Corner) -> @location(0) vec4<f32> {
    let away = corner.position.xy - corner.middle;
    if (away.x * away.x + away.y * away.y > radius * radius) { discard; }
    return vec4<f32>(corner.colour, 1.0);
}
"""


@cache
def stroke_shader(algebra) -> Shader:
    """The strokes of edges whose points belong to an algebra, placed by that algebra's camera."""
    types = dict(Point=algebra.gatype.antivector(), Plane=algebra.gatype.vector())
    return Shader(placing() + stroke_source, algebra, types, {}, targets=("rgba8unorm",))


@cache
def marker_shader(algebra) -> Shader:
    """The markers of points of an algebra, placed by that algebra's camera."""
    types = dict(Point=algebra.gatype.antivector(), Plane=algebra.gatype.vector())
    return Shader(placing() + marker_source, algebra, types, {}, targets=("rgba8unorm",))


def straight(image: np.ndarray) -> np.ndarray:
    """Colour and coverage from samples averaged as colour weighted by coverage: the colour divided back."""
    coverage = image[..., 3:]
    return np.concatenate((image[..., :3] / np.where(coverage > 0, coverage, 1), coverage), axis=-1)


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
            heads: np.ndarray = np.zeros(1), head: float = 0.0, dashes: tuple[float, float] = (1.0, 0.0)) -> np.ndarray:
    """The mesh's edges as strokes `width` pixels wide over its points, each in its colour `[edges, 3]`, read by the
    camera: `[rows, columns, 4]`, colour and coverage, clear where no stroke lies. The edges where `heads` `[edges]` is
    one end in a head `head` pixels long and as wide as its length; `dashes`, a dash's length and a gap's in pixels,
    breaks each chain of edges into dashes where the gap is not zero."""
    side = samples_per_side()
    rows, columns = camera.shape
    edges = np.asarray(mesh.edges, dtype=np.uint32)
    image, = stroke_shader(points.algebra).draw_averaged(
        camera.shape, side, 6 * len(edges), points=points, edges=edges,
        colours=np.broadcast_to(colours, (len(edges), 3)),
        travelled=travelled(points, edges, camera) * side, heads=np.broadcast_to(heads, len(edges)).astype(float),
        samples=np.array([columns * side, rows * side], dtype=float), radius=width * side / 2,
        head=np.array([head, head / 2]) * side, dashes=np.array(dashes) * side, **camera.uniforms(1.0))
    return straight(image)


def markers(points: Extensor, colours: np.ndarray, width: float, camera: Camera) -> np.ndarray:
    """Discs `width` pixels across at the points `[points]`, each in its colour `[points, 3]`, read by the camera:
    `[rows, columns, 4]`, colour and coverage, clear where no marker lies."""
    side = samples_per_side()
    rows, columns = camera.shape
    image, = marker_shader(points.algebra).draw_averaged(
        camera.shape, side, 6 * len(points), points=points, colours=np.broadcast_to(colours, (len(points), 3)),
        samples=np.array([columns * side, rows * side], dtype=float), radius=width * side / 2, **camera.uniforms(1.0))
    return straight(image)


def stroke_overlay(ax: Axes, points: Extensor, mesh: Mesh, colours: np.ndarray, width: float):
    """The mesh's edges drawn as strokes into a panel at its own pixels, the panel looking down on the plane of the
    points' generators x and y; the image, for an animation to set its data."""
    camera = flat_panel(ax, points.algebra)
    return pinned(ax, strokes(points, mesh, colours, width, camera), (*ax.get_xlim(), *ax.get_ylim()))
