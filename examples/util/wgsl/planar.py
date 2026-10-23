"""Pictures of the plane, drawn on the GPU: PGA2D geometry in, the RGB bytes of a window onto the plane out.

A window is a rectangle of the plane and its rows and columns of square pixels. Each pixel's centre is a point of the
plane, and every primitive covers a share of the pixel by a pairing with that point, worked out analytically, so one
point a pixel draws soft edges:

- a conic, a map `Plane <- Point`, drawn about the radius it is drawn at: the distance between that radius, a line
  paired with the point and held above a floor, and the square root of `conic(p) & p` less an offset, eased over three
  quarters of a pixel; only on the positive side of any of its clip lines;
- a region, where `conic(p) & p` is negative: its value over its polar's slope, `conic(p)`, is the distance to its edge;
- a triangle, on the inner side of each edge's line, the line at unit weight paired with the point its distance;
- a stroke along the segments of a polyline: each segment's line at unit weight paired with the point is the distance
  across it, and the line square to it through its start paired with the point the distance along it, on dashes of a
  pattern along the polyline, each dash ended by its cap.

Across an edge, a primitive covers the share of the pixel's width on its side. The primitives are composited in the
order given, each in its colour, by its opacity times its share of the pixel.
"""

from __future__ import annotations

from enum import IntEnum
from functools import cache
from typing import NamedTuple

import numpy as np

from numga import NumpyContext, concatenate
from numga.algebras import PGA2D as ga
from examples.util.wgsl import Shader

mv = NumpyContext(ga).multivector
Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
Quadric = ga.gatype((Plane, Point))                                          # Plane <- Point
origin = mv.w.dual()                                                         # [] Point
# The whole plane, as a clip line: the line at infinity pairs every point of unit weight to one.
everywhere = mv.w                                                            # [] Plane
# A conic that covers no point of unit weight.
nothing = (mv.w * (mv.w & Point)).cast(Quadric)                              # [] Plane <- Point


class Cap(IntEnum):
    """How a stroke's dashes end: square at their ends, square half the stroke's width past them, or round."""
    BUTT = 0
    PROJECTING = 1
    ROUND = 2


source = """
// A primitive, composited in order. A conic, `conics[first]` drawn about the radius `radii[first]` and above its floor,
// less its offset, on the positive side of any of `clips[clip_first]` up to `clips[clip_end]`; a region, where
// `conics[first]` is negative; a triangle, `corners[3 first]` to `corners[3 first + 2]`; or a stroke along the segments
// `first` up to `end`, its width, the lengths of its dashes and of the gaps between them, in the plane's units, and its
// cap. Its colour, and how much of what lies under it it hides.
struct Primitive {
    kind: u32,
    first: u32,
    end: u32,
    clip_first: u32,
    clip_end: u32,
    cap: u32,
    colour: vec3<f32>,
    opacity: f32,
    offset: f32,
    floor: f32,
    width: f32,
    dash: f32,
    gap: f32,
}

var<storage, read> primitives: array<Primitive>;
var<storage, read> conics: array<Quadric>;
var<storage, read> radii: array<Plane>;
var<storage, read> clips: array<Plane>;
var<storage, read> corners: array<Point>;
var<storage, read> starts: array<Point>;
var<storage, read> ends: array<Point>;
// How far along its polyline each segment starts.
var<storage, read> phases: array<f32>;
// The window's top left corner, a pixel's width in the plane, and what lies under every primitive.
var<uniform> corner: Point;
var<uniform> pixel: f32;
var<uniform> background: vec3<f32>;
// The image's columns and rows, and each pixel's colour.
var<uniform> pixels: vec2<u32>;
var<storage, read_write> image: array<vec3<f32>>;

const conic = 0u;
const region = 1u;
const triangle = 2u;
const projecting = 1u;
const round_cap = 2u;

// The share of a pixel's width about a place that the stretch from `low` to `high` covers.
fn overlap(low: f32, high: f32, place: f32) -> f32 {
    return max(min(high, place + pixel / 2) - max(low, place - pixel / 2), 0.0) / pixel;
}

// The share of a pixel's width on the inner side of an edge, the pixel's centre a distance inside it.
fn inner(distance: f32) -> f32 {
    return clamp(0.5 + distance / pixel, 0.0, 1.0);
}

fn conic_share(primitive: Primitive, point: Point) -> f32 {
    var shown = false;
    for (var index = primitive.clip_first; index < primitive.clip_end; index++) {
        shown = shown || (clips[index] & point) > 0;
    }
    let value = max((conics[primitive.first](point) & point) - primitive.offset, 0.0);
    let distance = max(radii[primitive.first] & point, primitive.floor) - sqrt(value);
    return select(0.0, 1 / (1 + exp(clamp(-distance / (0.75 * pixel), -30.0, 30.0))), shown);
}

// Inside a region by its value over twice the slope of its polar: the distance to its edge, to first order.
fn region_share(index: u32, point: Point) -> f32 {
    let polar = conics[index](point);
    return inner(-(polar & point) / (2 * polar.norm()));
}

// Inside where the point, joined with each edge, spans the triangle's area with its sign; each edge's line at unit
// weight pairs with the point to its distance.
fn triangle_share(index: u32, point: Point) -> f32 {
    let a = corners[3 * index];
    let b = corners[3 * index + 1];
    let c = corners[3 * index + 2];
    let side = sign(a & b & c);
    return inner(((a & b).normalized() & point) * side) * inner(((b & c).normalized() & point) * side)
        * inner(((c & a).normalized() & point) * side);
}

// The share of a pixel a dash from `low` to `high` along a segment covers, the pixel's centre `along` it and `across`
// it: a band of the stroke's width, cut square at the dash's ends, or past them by its cap.
fn dash_share(primitive: Primitive, low: f32, high: f32, along: f32, across: f32) -> f32 {
    let half = primitive.width / 2;
    if (low > high) {
        return 0.0;
    }
    if (primitive.cap == round_cap) {
        let distance = sqrt((along - clamp(along, low, high)) * (along - clamp(along, low, high)) + across * across);
        return overlap(-half, half, distance);
    }
    let reach = select(0.0, half, primitive.cap == projecting);
    return overlap(-half, half, across) * overlap(low - reach, high + reach, along);
}

fn stroke_share(primitive: Primitive, point: Point) -> f32 {
    var share = 0.0;
    let period = primitive.dash + primitive.gap;
    for (var index = primitive.first; index < primitive.end; index++) {
        let start = starts[index];
        let end = ends[index];
        // Across the segment: its line at unit weight paired with the point; along it: the line square to it through
        // its start, pointing to its end. Along the polyline, from its start.
        let line = (start & end).normalized();
        let square = -(line | start);
        let across = line & point;
        let low = phases[index];
        let high = low + (square & end);
        let along = low + (square & point);
        if (primitive.gap == 0) {
            share = max(share, dash_share(primitive, low, high, along, across));
            continue;
        }
        // The dash the pixel's centre lies along, and those either side of it, within the segment.
        let dash = floor(along / period);
        for (var step = -1.0; step <= 1.0; step += 1.0) {
            let first = (dash + step) * period;
            share = max(share, dash_share(primitive, max(first, low), min(first + primitive.dash, high), along, across));
        }
    }
    return share;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= pixels.x || id.y >= pixels.y) {
        return;
    }
    // The pixel's centre: right and down from the window's top left corner.
    let point = corner + (mv.x.dual() * (f32(id.x) + 0.5) - mv.y.dual() * (f32(id.y) + 0.5)) * pixel;
    var colour = background;
    for (var index = 0u; index < arrayLength(&primitives); index++) {
        let primitive = primitives[index];
        var share = 0.0;
        if (primitive.kind == conic) {
            share = conic_share(primitive, point);
        } else if (primitive.kind == region) {
            share = region_share(primitive.first, point);
        } else if (primitive.kind == triangle) {
            share = triangle_share(primitive.first, point);
        } else {
            share = stroke_share(primitive, point);
        }
        colour = mix(colour, primitive.colour, primitive.opacity * share);
    }
    image[id.y * pixels.x + id.x] = colour;
}
"""


@cache
def shader() -> Shader:
    return Shader(source, ga, dict(Point=Point, Plane=Plane, Quadric=Quadric))


class Window(NamedTuple):
    """A rectangle of the plane: its top left corner, a pixel's width, and its rows and columns of square pixels."""
    left: float
    top: float
    pixel: float
    shape: tuple[int, int]


def window(x_range: tuple[float, float], y_range: tuple[float, float], rows: int) -> Window:
    """The window `rows` pixels down `y_range`, as many square pixels across `x_range`, about its middle, as fit."""
    pixel = (y_range[1] - y_range[0]) / rows
    columns = round((x_range[1] - x_range[0]) / pixel)
    return Window(sum(x_range) / 2 - columns * pixel / 2, y_range[1], pixel, (rows, columns))


class Conics(NamedTuple):
    """Conics `[n]` drawn about a radius: covering a pixel by its centre's distance inside the radius, a line `[n]`
    paired with the point, at least the floor `[n]`, beyond the square root of `conic(p) & p` less the offset `[n]`,
    eased over three quarters of a pixel; and only on the positive side of any of the clip lines `[n, clips]`,
    `everywhere` for no clip. One colour `[3]` or one each `[n, 3]`, and the opacity of all."""
    conics: Quadric
    offsets: np.ndarray
    radii: Plane
    floors: np.ndarray
    clips: Plane
    colour: np.ndarray
    opacity: float


class Regions(NamedTuple):
    """The regions where conics `[n]` are negative; one colour `[3]` or one each `[n, 3]`, and the opacity of all."""
    conics: Quadric
    colour: np.ndarray
    opacity: float


class Triangles(NamedTuple):
    """Filled triangles, their corners `[n, 3]`; one colour `[3]` or one each `[n, 3]`, and the opacity of all."""
    corners: Point
    colour: np.ndarray
    opacity: float


class Strokes(NamedTuple):
    """Polylines through points of unit weight `[n, k]`, each stroked `width` pixels across along its `k - 1` segments;
    dashed `dashes` pixels on and off along each, starting on, `(1, 0)` for solid; every dash, and every segment of a
    solid stroke, ended by the cap. One colour `[3]` or one each `[n, 3]`, and the opacity of all."""
    points: Point
    width: float
    colour: np.ndarray
    opacity: float
    dashes: tuple[float, float] = (1.0, 0.0)
    cap: Cap = Cap.BUTT


def draw(window: Window, layers: list, background: np.ndarray) -> np.ndarray:
    """The layers, `Conics`, `Regions`, `Triangles` and `Strokes`, composited in order over the background `[3]`, each
    primitive of a layer after the one before: the window's RGB bytes `[rows, columns, 3]`."""
    found = {name: [] for name in ("conics", "radii", "clips", "corners", "starts", "ends", "phases")}
    primitives = []
    for layer in layers:
        size = layer[0].shape[0]
        # Each primitive's kind, where its parts and clip lines start and how many it has of each; a conic's offset
        # and floor; a stroke's width and dashes, in pixels, and its cap.
        n_clips, offsets, floors, width, (dash, gap), cap = 0, np.zeros(size), np.zeros(size), 0.0, (0.0, 0.0), Cap.BUTT
        clip_first = held(found["clips"])
        if isinstance(layer, (Conics, Regions)):
            kind, first, n_parts = (0 if isinstance(layer, Conics) else 1), held(found["conics"]), 1
            found["conics"].append(layer.conics.cast(Quadric))
        if isinstance(layer, Conics):
            n_clips, offsets, floors = layer.clips.shape[-1], layer.offsets, layer.floors
            found["radii"].append(layer.radii.cast(Plane).broadcast_to(layer.conics.shape))
            found["clips"].append(layer.clips.cast(Plane).broadcast_to(layer.conics.shape + (n_clips,)).reshape(size * n_clips))
        elif isinstance(layer, Regions):
            found["radii"].append(everywhere.cast(Plane).broadcast_to(layer.conics.shape))
        elif isinstance(layer, Triangles):
            kind, first, n_parts = 2, held(found["corners"]) // 3, 1
            found["corners"].append(layer.corners.cast(Point).reshape(3 * size))
        else:
            kind, first, n_parts = 3, held(found["starts"]), layer.points.shape[1] - 1
            width, (dash, gap), cap = layer.width, layer.dashes, layer.cap
            points = layer.points.cast(Point)
            found["starts"].append(points[:, :-1].reshape(size * n_parts))
            found["ends"].append(points[:, 1:].reshape(size * n_parts))
            # Where each segment starts along its polyline: the lengths of the segments before it.
            lengths = (points[:, 1:] - points[:, :-1]).dual().norm().to_array()   # [n, k - 1]
            found["phases"].append((np.cumsum(lengths, axis=1) - lengths).reshape(-1))
        firsts = first + n_parts * np.arange(size)
        clip_firsts = clip_first + n_clips * np.arange(size)
        primitives.append(dict(kind=np.full(size, kind), first=firsts, end=firsts + n_parts, clip_first=clip_firsts,
                               clip_end=clip_firsts + n_clips, cap=np.full(size, int(cap)),
                               colour=np.broadcast_to(np.asarray(layer.colour, dtype=float), (size, 3)),
                               opacity=np.full(size, layer.opacity), offset=np.broadcast_to(offsets, size),
                               floor=np.broadcast_to(floors, size), width=np.full(size, width * window.pixel),
                               dash=np.full(size, dash * window.pixel), gap=np.full(size, gap * window.pixel)))
    # A storage array binds at least one element: an empty one holds a filler that no primitive reads.
    fillers = dict(conics=nothing, radii=everywhere, clips=everywhere, corners=origin, starts=origin, ends=origin)
    arrays = {name: concatenate(parts) if parts else fillers[name].broadcast_to((1,))
              for name, parts in found.items() if name != "phases"}
    image = shader()(window.shape, **arrays, phases=np.concatenate(found["phases"] or [np.zeros(1)]),
                     primitives={name: np.concatenate([part[name] for part in primitives]) for name in primitives[0]},
                     corner=mv.point([window.left, window.top]), pixel=window.pixel,
                     background=np.asarray(background, dtype=float))["image"]
    return np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)


def held(parts: list) -> int:
    """How many elements batches hold together."""
    return sum(part.shape[0] for part in parts)
