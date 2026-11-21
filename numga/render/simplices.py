"""Edges and vertices of a simplicial `Topology` are rendered on the GPU directly in the scene's own
algebra as screen-space strokes and markers or as lit 3D tubes.

A stroke is a 1-simplex and a marker is a 0-simplex. `Camera` projects each vertex by regressive
pairing against the camera's planes without metric assumptions, so the same shader source draws
polylines in the plane, graphs in space, or world lines in spacetime. The vertex stage expands a
screen-space quad around each projected segment or point padded by its half-width, and the fragment
stage keeps supersampled subpixels within that Euclidean pixel distance of the segment or centre. A
segment's half-width runs linearly between its two points' widths, so a chain narrows smoothly.
Clamping along the segment rounds joins automatically, while open chain endpoints and dash breaks
apply `Cap` geometry parameterized by cumulative screen-space arc length along contiguous edge
chains. Translucent chains suppress duplicate coverage across shared vertices by yielding samples
inside the preceding edge's segment. An edge may also carry a tip chevron formed by two angled
screen-space barbs scaled down when the host segment is shorter than the barb reach.

In lit 3D scenes (`tubes`), strokes share the `PGA3D` frame and directional lighting of `solids`
and emit per-sample colour and view-ray depth into sample slots for `solids` and `Scene` to
composite against quadrics and triangulated surfaces. Tube normals are orthogonalized against each
edge's 3D camera-frame run along interior spans so foreshortened segments shade as true cylinders
across joints, while open caps and chevrons shade spherically and advance their depth by their
radius bulge so arrows sitting on a surface remain legible.
"""
from __future__ import annotations

from enum import IntEnum
from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from numga import GATypeError, dim
from numga.extensor import Extensor
from numga.gatype import GAType
from numga.topology import Topology
from numga.wgsl import Shader, samples_per_side

from . import solids
from .camera import Camera, check_projection, placing, read, reading
from .drawing import Draw

if TYPE_CHECKING:
    from .solids import View


class Cap(IntEnum):
    """Open chain endpoints and dash breaks terminate square at the endpoint, square half a
    stroke width past it, or round, while interior joins remain round."""
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
// The strokes' half width at each point, in samples, across each edge from one end's to the other's.
var<storage, read> radii: array<f32>;
// The picture's samples across and down, a chevron's stroke's reach back along its edge and out to its side, and the
// dashes' length and the gaps', all in samples; a gap of zero draws the strokes whole.
var<uniform> samples: vec2<f32>;
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
    // The half widths at the edge's ends, in samples.
    @interpolate(flat) radii: vec2<f32>,
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
    let radius = mix(radii[edge.x], radii[edge.y], share);
    return away.x * away.x + away.y * away.y <= radius * radius;
}

@vertex
fn place(@builtin(vertex_index) index: u32) -> Corner {
    let edge = edges[index / 6];
    let start = clip(points[edge.x]);
    let end = clip(points[edge.y]);
    let from = sampled(start);
    let to = sampled(end);
    // The box around the edge, its wider end's half width and a sample beyond it every way: along it and across it.
    let span = to - from;
    let length = sqrt(span.x * span.x + span.y * span.y);
    let along = select(vec2<f32>(1.0, 0.0), span / length, length > 0.0);
    let across = vec2<f32>(-along.y, along.x);
    let ends_radii = vec2<f32>(radii[edge.x], radii[edge.y]);
    let reach = max(ends_radii.x, ends_radii.y) + chevron.y * chevrons[index / 6] + 1.0;
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
                  depths, run, index / 6, ends_radii);
}

// Where a sample lies against the edge: how far along it, a share of its length; how far across its line and on
// which side, and how far from the segment's nearest point, in samples; past which end it lies, start or end, or
// neither; whether it is covered, by the stroke, a dash or the chevron; and whether by the chevron alone, and how far
// from the nearer of its strokes; and the half width there, in samples.
struct Against {
    share: f32,
    across: vec2<f32>,
    away: vec2<f32>,
    past: vec2<f32>,
    covered: bool,
    barbed: bool,
    barb: vec2<f32>,
    radius: f32,
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
    // The half width at the segment's nearest point, from one end's to the other's: past an end, that end's.
    let radius = mix(corner.radii.x, corner.radii.y, select(0.0, share / length, length > 0.0));
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
    // The chevron: two strokes from the edge's end, back along it and out to either side, each as wide as the edge's
    // end, shrunk to fit an edge shorter than they are long.
    let to_end = corner.position.xy - corner.ends.zw;
    let side = vec2<f32>(-along_unit.y, along_unit.x);
    let fit = min(1.0, length / max(sqrt(chevron.x * chevron.x + chevron.y * chevron.y), 1e-6));
    let left = off_segment(to_end, (-along_unit * chevron.x + side * chevron.y) * fit);
    let right = off_segment(to_end, (-along_unit * chevron.x - side * chevron.y) * fit);
    let barb = select(right, left, dot(left, left) < dot(right, right));
    let in_barb = corner.chain.y > 0.0 && dot(barb, barb) <= corner.radii.y * corner.radii.y;
    return Against(select(0.0, share / length, length > 0.0), sideways, away, past, (in_shaft || in_barb) && owned,
                   in_barb && !in_shaft, barb, select(radius, corner.radii.y, in_barb && !in_shaft));
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

# In a lit 3D scene each stroke shades as a cylinder through its normal and writes view-ray depth
# alongside colour for the solids ray tracer to composite.
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
    // A tube may thin to nothing at a point, a half width of zero that is not divided by.
    let across = min(distance / max(sample.radius, 1e-6), 1.0);
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
    let bulge = toward * sample.radius + select(0.0, chevron.x, sample.barbed);
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
    """The screen-space stroke shader binds `Point` vertices against `Plane` camera planes in
    `Point.algebra`."""
    return Shader(placing(Plane) + stroke_source + flat_stroke, Point.algebra, dict(Point=Point, Plane=Plane), {})


@cache
def tube_shader() -> Shader:
    """The lit `PGA3D` tube shader writes premultiplied RGBA colour and full-precision view depth
    under `solids` lamp shading."""
    return Shader(solids.lamp_shading() + placing(solids.Plane) + stroke_source + tube_stroke, solids.ga,
                  dict(Point=solids.Point, Plane=solids.Plane), {})


@cache
def marker_shader(Point: GAType, Plane: GAType) -> Shader:
    """The screen-space circular marker shader binds `Point` vertices against `Plane` camera planes
    in `Point.algebra`."""
    return Shader(placing(Plane) + marker_source, Point.algebra, dict(Point=Point, Plane=Plane), {})


def _pixels(value: Extensor | np.ndarray | float, name: str) -> np.ndarray | float:
    """Read pixel widths, radii, or chevron lengths, requiring `dim.pixel` on checked `Extensor` inputs."""
    if isinstance(value, Extensor):
        if value.has_units and not value.units.is_zero and value.unit != dim.pixel:
            raise GATypeError(f"{name} must carry pixel units; got {value.gatype.signature}")
        return value.to_array()
    return value


def chevron_reach(length: Extensor | float) -> np.ndarray:
    """A chevron barb of pixel `length` angled thirty degrees off its edge projects onto axial and
    transverse components `(sqrt(3) / 2, 1 / 2) * length`."""
    return float(_pixels(length, "chevron")) * np.array([np.sqrt(3) / 2, 1 / 2])


def following(edges: np.ndarray) -> np.ndarray:
    """Each edge whose start vertex matches the end vertex of the immediately preceding edge carries
    `1.0` so translucent chains cover shared joints only once."""
    follows = np.zeros(len(edges))
    follows[1:] = edges[1:, 0] == edges[:-1, 1]
    return follows


def travelled(points: Extensor, edges: np.ndarray, camera: Camera) -> np.ndarray:
    """Cumulative projected pixel distance precedes each edge along contiguous chains and resets
    wherever consecutive edges do not share an endpoint."""
    readings = (np.stack([read(camera.across, points).to_array(), read(camera.up, points).to_array()], axis=-1)
                / (read(camera.weight, points).to_array() + camera.unit)[..., None]).reshape(-1, 2) / camera.pixel
    lengths = np.linalg.norm(readings[edges[:, 1]] - readings[edges[:, 0]], axis=-1)
    starts = np.ones(len(edges), dtype=bool)
    starts[1:] = edges[1:, 0] != edges[:-1, 1]
    totals = np.cumsum(lengths)
    return totals - lengths - np.maximum.accumulate(np.where(starts, totals - lengths, 0))


def strokes(points: Extensor, topology: Topology, colours: np.ndarray, width: Extensor | np.ndarray | float,
            camera: Camera, chevrons: np.ndarray = np.zeros(1), chevron: Extensor | float = 0.0,
            dashes: tuple[float, float] = (1.0, 0.0), cap: Cap = Cap.ROUND, opacity: float = 1.0) -> Draw | np.ndarray:
    """Project the 1-simplices of `topology` over `points` (`[V]`) through `camera` as a 2D stroke
    `Draw` pass (or a transparent `[rows, columns, 4]` array when `topology` has no edges) `width`
    pixels wide (`[V]` or one number, linear along each edge between its points' widths) in
    `colours` (`[E, 3]` or `[3]`), with optional tip chevrons and periodic dash-gap patterns along
    contiguous edge chains."""
    check_projection(points, camera)
    width_px = _pixels(width, "width")
    chevron_px = chevron_reach(chevron)
    side = samples_per_side()
    rows, columns = camera.shape
    edges = np.asarray(topology.elements[1], dtype=np.uint32)
    # An empty edge set (such as a trajectory not yet begun) produces a transparent RGBA layer.
    if len(edges) == 0:
        return np.zeros(camera.shape + (4,))
    return Draw(stroke_shader(**reading(points, camera)), camera.shape, 6 * len(edges), dict(
        points=points, edges=edges,
        colours=np.broadcast_to(colours, (len(edges), 3)),
        travelled=travelled(points, edges, camera) * side, chevrons=np.broadcast_to(chevrons, len(edges)).astype(float),
        open_ends=np.isin(edges, topology.exterior[0]).astype(float),
        follows=following(edges),
        radii=np.broadcast_to(width_px, points.shape).astype(float) * side / 2,
        samples=np.array([columns * side, rows * side], dtype=float),
        chevron=chevron_px * side, dashes=np.array(dashes) * side, cap=float(cap), opacity=opacity,
        **camera.uniforms(1.0)))


def view_camera(view: View) -> Camera:
    """`view.to_camera()` converts an orthographic `PGA3D` view into its `Camera`."""
    return view.to_camera()


def tubes(points: Extensor, topology: Topology, colours: np.ndarray, width: Extensor | np.ndarray | float, view: View,
          gloss: float = 0.0, chevrons: np.ndarray = np.zeros(1), chevron: Extensor | float = 0.0,
          dashes: tuple[float, float] = (1.0, 0.0)) -> Draw:
    """Render the 1-simplices of `topology` over `points` (`[V]`) in `PGA3D` as supersampled 3D
    tubes `width` pixels wide (`[V]` or one number, a truncated cone along each edge between its
    points' widths) in `colours` (`[E, 3]` or `[3]`) for depth-ordered compositing alongside quadric
    solids and triangulated surfaces."""
    camera = view.to_camera()
    side = samples_per_side()
    rows, columns = camera.shape
    edges = np.asarray(topology.elements[1], dtype=np.uint32)
    return Draw(tube_shader(), camera.shape, 6 * len(edges), dict(
        points=points, edges=edges, colours=np.broadcast_to(colours, (len(edges), 3)),
        travelled=travelled(points, edges, camera) * side, chevrons=np.broadcast_to(chevrons, len(edges)).astype(float),
        radii=np.broadcast_to(_pixels(width, "width"), points.shape).astype(float) * side / 2,
        samples=np.array([columns * side, rows * side], dtype=float),
        chevron=chevron_reach(chevron) * side, dashes=np.array(dashes) * side, scene_pixel=view.pixel,
        oversampling=float(side), cap=float(Cap.ROUND), opacity=1.0,
        open_ends=np.isin(edges, topology.exterior[0]).astype(float),
        follows=following(edges),
        **camera.uniforms(8 * view.half.max()), **solids.lighting(gloss)))


def markers(points: Extensor, colours: np.ndarray, radii: Extensor | np.ndarray | float, camera: Camera,
            opacity: float = 1.0) -> Draw | np.ndarray:
    """Project circular dots of pixel `radii` around `points` (`[N]`) in `colours` (`[N, 3]` or
    `[3]`) through `camera` into a 2D `Draw` pass (or a transparent `[rows, columns, 4]` array when
    `points` is empty)."""
    check_projection(points, camera)
    radii_px = _pixels(radii, "radii")
    side = samples_per_side()
    rows, columns = camera.shape
    # An empty point batch produces a transparent RGBA layer.
    if len(points) == 0:
        return np.zeros(camera.shape + (4,))
    return Draw(marker_shader(**reading(points, camera)), camera.shape, 6 * len(points), dict(
        points=points, colours=np.broadcast_to(colours, (len(points), 3)),
        radii=np.broadcast_to(radii_px, len(points)).astype(float) * side,
        samples=np.array([columns * side, rows * side], dtype=float), opacity=opacity, **camera.uniforms(1.0)))
