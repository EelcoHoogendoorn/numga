"""Pictures of the plane, drawn on the GPU: PGA2D geometry in, the RGB bytes of a window onto the plane out.

A window is a rectangle of the plane and its rows and columns of square pixels. Each pixel's centre is a point of the
plane, and every primitive covers a share of the pixel by a pairing with that point, worked out analytically, so one
point a pixel draws soft edges:

- a conic, a map `Plane <- Point`, drawn about the radius it is drawn at: the distance between that radius, a line
  paired with the point and held above a floor, and the square root of `conic(p) & p` less an offset, eased over three
  quarters of a pixel; only on the positive side of any of its clip lines;
- a region, where `conic(p) & p` is negative: its value over its polar's slope, `conic(p)`, is the distance to its edge;
- a curve, where `conic(p) & p` vanishes, stroked about that distance; an envelope, a map from lines to the points
  they touch, is drawn as the curve of its inverse, the conic of its points;
- a line, whole across the window: the line at unit weight paired with the point is the distance across it;
- a polygon, where its edges wind about the point: each triangle of the fan from its first corner counts with its
  sign, on the inner side of each edge's line, the line at unit weight paired with the point its distance;
- a stroke along the segments of a polyline: each segment's line at unit weight paired with the point is the distance
  across it, and the line square to it through its start paired with the point the distance along it, on dashes of a
  pattern along the polyline, each dash ended by its cap and its segments joined round;
- a mark about a point: a disc, by the join of the two points, whose norm is their distance; a square or a diamond, by
  the pairings of two square lines through it.

Across an edge, a primitive covers the share of the pixel's width on its side. The primitives are composited in the
order given, each in its colour, by its opacity times its share of the pixel. Points and lines of either orientation
are drawn alike.
"""

from __future__ import annotations

from enum import IntEnum
from functools import cache
from typing import NamedTuple

import numpy as np

from numga import Extensor, NumpyContext, concatenate
from numga.algebras import PGA2D as ga
from numga.gatype import GAType
from examples.util.wgsl import Shader
from examples.util.wgsl.library import intrinsic_one

mv = NumpyContext(ga).multivector
Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
Quadric = ga.gatype((Plane, Point))                                          # Plane <- Point
Envelope = ga.gatype((Point, Plane))                                         # Point <- Plane
origin = mv.w.dual()                                                         # [] Point
# The whole plane, as a clip line: the line at infinity pairs every point of unit weight to one.
everywhere = mv.w                                                            # [] Plane
# A conic that covers no point of unit weight.
nothing = (mv.w * (mv.w & Point)).cast(Quadric)                              # [] Plane <- Point
# The intrinsic scalar one: a product with it keeps every coefficient and flips only the orientation.
flip = intrinsic_one(mv)                                                     # [] Scalar
# A solid stroke: one dash, no gap.
SOLID = (1.0, 0.0)


class Cap(IntEnum):
    """How a stroke's dashes end: square at their ends, square half the stroke's width past them, or round."""
    BUTT = 0
    PROJECTING = 1
    ROUND = 2


class Marker(IntEnum):
    """The outline of a mark about its point: a circle, a square along the window's rows and columns, or that square
    turned an eighth of a turn."""
    DISC = 0
    SQUARE = 1
    DIAMOND = 2


class Kind(IntEnum):
    """What a primitive is, as the shader reads it."""
    CONIC = 0
    REGION = 1
    CURVE = 2
    LINE = 3
    POLYGON = 4
    STROKE = 5
    MARK = 6


source = """
// A primitive, composited in order, of its kind, and where its parts lie: `first` among the conics for a conic, a
// region or a curve, among the planes for a line, among the points for a mark, and its points from `first` up to `end`
// for a polygon or a stroke. A conic, drawn about the radius `planes[plane_first]` and above its floor, less its
// offset, on the positive side of any of the clip lines after it up to `planes[plane_end]`; a curve or a line, `width`
// across; a stroke, its width, the lengths of its dashes and of the gaps between them, in the plane's units, and its
// cap, its `shape`; a mark, its marker, its `shape`, covering from `inner` to `outer` from its point. Its colour, and
// how much of what lies under it it hides.
struct Primitive {
    kind: u32,
    first: u32,
    end: u32,
    plane_first: u32,
    plane_end: u32,
    shape: u32,
    colour: vec3<f32>,
    opacity: f32,
    offset: f32,
    floor: f32,
    width: f32,
    dash: f32,
    gap: f32,
    inner: f32,
    outer: f32,
}

var<storage, read> primitives: array<Primitive>;
var<storage, read> conics: array<Quadric>;
var<storage, read> planes: array<Plane>;
var<storage, read> points: array<Point>;
// How far along its polyline each point of a stroke lies.
var<storage, read> phases: array<f32>;
// The window's top left corner, a pixel's width in the plane, and what lies under every primitive.
var<uniform> corner: Point;
var<uniform> pixel: f32;
var<uniform> background: vec3<f32>;
// The image's columns and rows, and each pixel's colour.
var<uniform> pixels: vec2<u32>;
var<storage, read_write> image: array<vec3<f32>>;

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
    for (var index = primitive.plane_first + 1; index < primitive.plane_end; index++) {
        shown = shown || (planes[index] & point) > 0;
    }
    let value = max((conics[primitive.first](point) & point) - primitive.offset, 0.0);
    let distance = max(planes[primitive.plane_first] & point, primitive.floor) - sqrt(value);
    return select(0.0, 1 / (1 + exp(clamp(-distance / (0.75 * pixel), -30.0, 30.0))), shown);
}

// Inside a region by its value over twice the slope of its polar: the distance to its edge, to first order.
fn region_share(index: u32, point: Point) -> f32 {
    let polar = conics[index](point);
    return inner(-(polar & point) / (2 * polar.norm()));
}

// Within half the width of a curve by the same distance; nowhere where the polar has no slope.
fn curve_share(primitive: Primitive, point: Point) -> f32 {
    let polar = conics[primitive.first](point);
    let slope = 2 * polar.norm();
    let distance = (polar & point) / slope;
    return select(overlap(-primitive.width / 2, primitive.width / 2, distance), 0.0, slope == 0);
}

fn line_share(primitive: Primitive, point: Point) -> f32 {
    return overlap(-primitive.width / 2, primitive.width / 2, planes[primitive.first].normalized() & point);
}

// Where the polygon winds about the point, from the fan of triangles from its first corner, each counted with the
// sign of its area: inside where the point, joined with each edge, spans the triangle's area with its sign, each
// edge's line at unit weight pairing with the point to its distance. Along an edge two triangles share, their shares
// make up the whole pixel. Each triangle lies across its longest edge, both angles at that edge being acute, so its
// share is also cut to that edge's length along it: past a thin triangle's sharp corners, where its edges' lines lie
// within a pixel of each other, it covers nothing.
fn polygon_share(primitive: Primitive, point: Point) -> f32 {
    let a = points[primitive.first];
    var winding = 0.0;
    for (var index = primitive.first + 1; index + 1 < primitive.end; index++) {
        let b = points[index];
        let c = points[index + 1];
        let area = a & b & c;
        let side = sign(area);
        let share = inner(((a & b).normalized() & point) * side) * inner(((b & c).normalized() & point) * side)
            * inner(((c & a).normalized() & point) * side);
        winding += select(side * share * spanned(a, b, c, point), 0.0, area == 0);
    }
    return min(abs(winding), 1.0);
}

// The share of a pixel's width that lies along the longest edge of a triangle of points at unit weight, between its
// two ends: the line square to the edge through its first end, pointing to its other end, paired with the point.
fn spanned(a: Point, b: Point, c: Point, point: Point) -> f32 {
    let ab = (a & b).norm();
    let bc = (b & c).norm();
    let ca = (c & a).norm();
    let start = select(select(c, b, bc >= ca), a, ab >= bc && ab >= ca);
    let end = select(select(a, c, bc >= ca), b, ab >= bc && ab >= ca);
    let length = max(ab, max(bc, ca));
    let square = -((start & end).normalized() | start);
    return overlap(0.0, length, square & point);
}

// The share of a pixel a dash from `low` to `high` along a segment covers, the pixel's centre `along` it and `across`
// it: a band of the stroke's width, cut square at the dash's ends, and at an end the dash ends at, rather than carrying
// on into the segment next to it, past it by its cap. Where it carries on, the band reaches half a pixel past the end,
// so that the pixel on the joint, half on each segment, is covered whole by either.
fn dash_share(primitive: Primitive, low: f32, high: f32, low_capped: bool, high_capped: bool, along: f32,
              across: f32) -> f32 {
    let half = primitive.width / 2;
    if (low > high) {
        return 0.0;
    }
    let reach = select(0.0, half, primitive.shape == cap_projecting);
    let band = overlap(-half, half, across)
        * overlap(low - select(pixel / 2, reach, low_capped), high + select(pixel / 2, reach, high_capped), along);
    let round = primitive.shape == cap_round;
    let from_low = sqrt((along - low) * (along - low) + across * across);
    let from_high = sqrt((along - high) * (along - high) + across * across);
    return max(band, max(select(0.0, overlap(-half, half, from_low), round && low_capped),
                         select(0.0, overlap(-half, half, from_high), round && high_capped)));
}

// The round join at a segment's start, where the stroke carries on from the segment before: within half its width of
// the corner, behind the segment's start and past the end of the one before, on the outer side of the turn that the
// two bands leave open.
fn join_share(primitive: Primitive, before: Point, corner: Point, square: Plane, point: Point) -> f32 {
    let line = (before & corner).normalized();
    let past = -(line | corner) & point;
    let behind = -(square & point);
    let distance = (corner & point).norm();
    return overlap(-primitive.width / 2, primitive.width / 2, distance) * inner(past) * inner(behind);
}

fn stroke_share(primitive: Primitive, point: Point) -> f32 {
    var share = 0.0;
    let period = primitive.dash + primitive.gap;
    for (var index = primitive.first; index + 1 < primitive.end; index++) {
        let start = points[index];
        let end = points[index + 1];
        // A segment of no length draws nothing, and has no line to draw it by; its neighbours join across it.
        if ((start & end).norm() == 0) {
            continue;
        }
        // Across the segment: its line at unit weight paired with the point; along it: the line square to it through
        // its start, pointing to its end. Along the polyline, from its start.
        let line = (start & end).normalized();
        let square = -(line | start);
        let across = line & point;
        let low = phases[index];
        let high = phases[index + 1];
        let along = low + (square & point);
        // A segment after the first joins the one before, unless that one has no length and no direction to join.
        let after_first = index > primitive.first && (points[index - 1] & start).norm() > 0;
        let before_last = index + 2 < primitive.end;
        // A segment's start on a dash, after the first segment, joins it to the one before.
        let into = low - floor(low / period) * period;
        if (after_first && (primitive.gap == 0 || (into > 0 && into < primitive.dash))) {
            share = max(share, join_share(primitive, points[index - 1], start, square, point));
        }
        if (primitive.gap == 0) {
            share = max(share, dash_share(primitive, low, high, !after_first, !before_last, along, across));
            continue;
        }
        // The dash the pixel's centre lies along, and those either side of it, within the segment.
        let dash = floor(along / period);
        for (var step = -1.0; step <= 1.0; step += 1.0) {
            let first = (dash + step) * period;
            let last = first + primitive.dash;
            share = max(share, dash_share(primitive, max(first, low), min(last, high), !after_first || first >= low,
                                          !before_last || last <= high, along, across));
        }
    }
    return share;
}

// About a disc's point by its distance, the norm of their join; about a square's or a diamond's by the pairings of the
// lines x = 0 and y = 0 with the two, their offsets across the window's columns and rows, turned an eighth of a turn
// for a diamond.
fn mark_share(primitive: Primitive, point: Point) -> f32 {
    let centre = points[primitive.first];
    let low = primitive.inner;
    let high = primitive.outer;
    if (primitive.shape == marker_disc) {
        let distance = (centre & point).norm();
        return overlap(-high, high, distance) - overlap(-low, low, distance);
    }
    let x = (mv.x & point) - (mv.x & centre);
    let y = (mv.y & point) - (mv.y & centre);
    let turned = primitive.shape == marker_diamond;
    let u = select(x, (x + y) * sqrt(0.5), turned);
    let v = select(y, (x - y) * sqrt(0.5), turned);
    return overlap(-high, high, u) * overlap(-high, high, v) - overlap(-low, low, u) * overlap(-low, low, v);
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
        if (primitive.kind == kind_conic) {
            share = conic_share(primitive, point);
        } else if (primitive.kind == kind_region) {
            share = region_share(primitive.first, point);
        } else if (primitive.kind == kind_curve) {
            share = curve_share(primitive, point);
        } else if (primitive.kind == kind_line) {
            share = line_share(primitive, point);
        } else if (primitive.kind == kind_polygon) {
            share = polygon_share(primitive, point);
        } else if (primitive.kind == kind_stroke) {
            share = stroke_share(primitive, point);
        } else {
            share = mark_share(primitive, point);
        }
        colour = mix(colour, primitive.colour, primitive.opacity * share);
    }
    image[id.y * pixels.x + id.x] = colour;
}
"""


@cache
def shader() -> Shader:
    # Each kind, cap and marker as a constant of the shader, `kind_conic`, `cap_round`.
    constants = {f"{kind.__name__.lower()}_{member.name.lower()}": member.value
                 for kind in (Kind, Cap, Marker) for member in kind}
    return Shader(source, ga, dict(Point=Point, Plane=Plane, Quadric=Quadric), constants)


class Window(NamedTuple):
    """A rectangle of the plane: its top left corner, a pixel's width, and its rows and columns of square pixels."""
    left: float
    top: float
    pixel: float
    shape: tuple[int, int]

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Its left, right, bottom and top, as matplotlib places an image."""
        rows, columns = self.shape
        return self.left, self.left + columns * self.pixel, self.top - rows * self.pixel, self.top


def window(x_range: tuple[float, float], y_range: tuple[float, float], rows: int) -> Window:
    """The window `rows` pixels down `y_range`, as many square pixels across `x_range`, about its middle, as fit."""
    pixel = (y_range[1] - y_range[0]) / rows
    columns = round((x_range[1] - x_range[0]) / pixel)
    return Window(sum(x_range) / 2 - columns * pixel / 2, y_range[1], pixel, (rows, columns))


def filling(x_range: tuple[float, float], y_range: tuple[float, float], room: tuple[int, int]) -> Window:
    """The window of square pixels about the middle of `x_range` and `y_range` that they fill one way of a room of
    columns and rows, the other way as many pixels as they span."""
    pixel = max((x_range[1] - x_range[0]) / room[0], (y_range[1] - y_range[0]) / room[1])
    rows, columns = round((y_range[1] - y_range[0]) / pixel), round((x_range[1] - x_range[0]) / pixel)
    return Window(sum(x_range) / 2 - columns * pixel / 2, sum(y_range) / 2 + rows * pixel / 2, pixel, (rows, columns))


def oriented(value: Extensor, kind: GAType) -> Extensor:
    """A value as `kind`, through the intrinsic scalar one where their orientations differ."""
    return (value if value.gatype.orientation is kind.orientation else value * flip).cast(kind)


def point_conics(maps: Extensor) -> Quadric:
    """Maps between points and lines of either orientation as maps `Plane <- Point`: each side through the intrinsic
    scalar one where its orientation differs."""
    if maps.gatype.input_orientations[0] is not Point.orientation:
        maps = maps(Point * flip)
    return oriented(maps, Quadric)


def unit(points: Extensor) -> Point:
    """Points `[...]` of either orientation, at unit weight, as `Point`."""
    return oriented(points / (mv.w & points), Point)


class Parts:
    """The conics, planes, points and the points' phases that primitives draw from, gathered as layers add them."""

    def __init__(self) -> None:
        self.conics, self.planes, self.points, self.phases = [], [], [], []

    @staticmethod
    def add(held: list, values) -> int:
        """Where values `[m]` start among those held with them."""
        first = sum(part.shape[0] for part in held)
        held.append(values)
        return first

    def add_points(self, points: Point, phases: np.ndarray) -> int:
        """Where points `[m]` at unit weight, with their phases `[m]`, start among the points."""
        self.phases.append(phases)
        return self.add(self.points, points)


def fields(kind: Kind, first: np.ndarray, colour: np.ndarray, opacity: float, **rest) -> dict:
    """The fields of a layer's primitives `[n]`, from where each starts among its parts: its colour `[3]` or
    `[n, 3]` for each, its opacity, and those of its kind."""
    size = len(first)
    return dict(kind=np.full(size, int(kind)), first=first,
                colour=np.broadcast_to(np.asarray(colour, dtype=float), (size, 3)), opacity=np.full(size, opacity),
                **{name: np.broadcast_to(value, size) for name, value in rest.items()})


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

    def primitives(self, parts: Parts, pixel: float) -> dict:
        # Each conic's radius, then its clip lines.
        size, n_clips = self.conics.shape[0], self.clips.shape[-1]
        radii = oriented(self.radii, Plane).broadcast_to(self.conics.shape)[:, None]
        clips = oriented(self.clips, Plane).broadcast_to(self.conics.shape + (n_clips,))
        plane_first = parts.add(parts.planes, concatenate([radii, clips], axis=1).reshape(size * (n_clips + 1)))
        plane_firsts = plane_first + (n_clips + 1) * np.arange(size)
        first = parts.add(parts.conics, point_conics(self.conics)) + np.arange(size)
        return fields(Kind.CONIC, first, self.colour, self.opacity, plane_first=plane_firsts,
                      plane_end=plane_firsts + n_clips + 1, offset=self.offsets, floor=self.floors)


class Regions(NamedTuple):
    """The regions where conics `[n]` are negative; one colour `[3]` or one each `[n, 3]`, and the opacity of all."""
    conics: Quadric
    colour: np.ndarray
    opacity: float

    def primitives(self, parts: Parts, pixel: float) -> dict:
        first = parts.add(parts.conics, point_conics(self.conics)) + np.arange(self.conics.shape[0])
        return fields(Kind.REGION, first, self.colour, self.opacity)


class Curves(NamedTuple):
    """The curves where conics `[n]`, maps from points to lines, vanish, stroked `width` pixels across; one colour
    `[3]` or one each `[n, 3]`, and the opacity of all."""
    conics: Quadric
    width: float
    colour: np.ndarray
    opacity: float

    def primitives(self, parts: Parts, pixel: float) -> dict:
        # Each conic at its largest coefficient one, which draws the same curve.
        conics = point_conics(self.conics)
        first = parts.add(parts.conics, conics / np.abs(conics.kernel).max(axis=(-1, -2))) + np.arange(conics.shape[0])
        return fields(Kind.CURVE, first, self.colour, self.opacity, width=self.width * pixel)


class Envelopes(NamedTuple):
    """The curves that envelopes `[n]`, maps from lines to points, touch, stroked `width` pixels across: the curves of
    their inverses, the conics of their points. One colour `[3]` or one each `[n, 3]`, and the opacity of all."""
    envelopes: Envelope
    width: float
    colour: np.ndarray
    opacity: float

    def primitives(self, parts: Parts, pixel: float) -> dict:
        return Curves(self.envelopes.inverse(), self.width, self.colour, self.opacity).primitives(parts, pixel)


class Lines(NamedTuple):
    """Lines `[n]`, whole across the window, stroked `width` pixels across; one colour `[3]` or one each `[n, 3]`, and
    the opacity of all."""
    lines: Plane
    width: float
    colour: np.ndarray
    opacity: float

    def primitives(self, parts: Parts, pixel: float) -> dict:
        first = parts.add(parts.planes, oriented(self.lines, Plane)) + np.arange(self.lines.shape[0])
        return fields(Kind.LINE, first, self.colour, self.opacity, width=self.width * pixel)


class Polygons(NamedTuple):
    """Filled polygons, their corners `[n, k]`, covering the points their edges wind about; one colour `[3]` or one
    each `[n, 3]`, and the opacity of all."""
    corners: Point
    colour: np.ndarray
    opacity: float

    def primitives(self, parts: Parts, pixel: float) -> dict:
        size, n_corners = self.corners.shape
        first = parts.add_points(unit(self.corners).reshape(size * n_corners), np.zeros(size * n_corners))
        firsts = first + n_corners * np.arange(size)
        return fields(Kind.POLYGON, firsts, self.colour, self.opacity, end=firsts + n_corners)


class Strokes(NamedTuple):
    """Polylines through points `[n, k]`, each stroked `width` pixels across along its `k - 1` segments; dashed
    `dashes` pixels on and off along each, starting on, `SOLID` for solid; every dash, and every solid polyline, ended
    by the cap, and its segments joined round. One colour `[3]` or one each `[n, 3]`, and the opacity of all."""
    points: Point
    width: float
    colour: np.ndarray
    opacity: float
    dashes: tuple[float, float]
    cap: Cap

    def primitives(self, parts: Parts, pixel: float) -> dict:
        # Where each point lies along its polyline: the lengths of the segments before it.
        size, n_points = self.points.shape
        points = unit(self.points)
        lengths = (points[:, 1:] - points[:, :-1]).dual().norm().to_array()    # [n, k - 1]
        phases = np.concatenate([np.zeros((size, 1)), np.cumsum(lengths, axis=1)], axis=1)   # [n, k]
        first = parts.add_points(points.reshape(size * n_points), phases.reshape(-1))
        firsts = first + n_points * np.arange(size)
        return fields(Kind.STROKE, firsts, self.colour, self.opacity, end=firsts + n_points, shape=int(self.cap),
                      width=self.width * pixel, dash=self.dashes[0] * pixel, gap=self.dashes[1] * pixel)


class Marks(NamedTuple):
    """Marks about points `[n]`, each covering from `inner` to `outer` pixels from its point, across the marker's
    outline: filled from 0, a ring from further out. One colour `[3]` or one each `[n, 3]`, and the opacity of all."""
    points: Point
    marker: Marker
    inner: float
    outer: float
    colour: np.ndarray
    opacity: float

    def primitives(self, parts: Parts, pixel: float) -> dict:
        size = self.points.shape[0]
        first = parts.add_points(unit(self.points), np.zeros(size)) + np.arange(size)
        return fields(Kind.MARK, first, self.colour, self.opacity, shape=int(self.marker), inner=self.inner * pixel,
                      outer=self.outer * pixel)


def draw(window: Window, layers: list, background: np.ndarray) -> np.ndarray:
    """The layers, `Conics`, `Regions`, `Curves`, `Envelopes`, `Lines`, `Polygons`, `Strokes` and `Marks`, composited
    in order over the background `[3]`, each primitive of a layer after the one before: the window's RGB bytes
    `[rows, columns, 3]`."""
    parts = Parts()
    found = [layer.primitives(parts, window.pixel) for layer in layers]
    # Every field of every primitive; those a kind does not read are zero.
    names = ("kind", "first", "end", "plane_first", "plane_end", "shape", "colour", "opacity", "offset", "floor",
             "width", "dash", "gap", "inner", "outer")
    primitives = {name: np.concatenate([layer.get(name, np.zeros(len(layer["first"]), dtype=int)) for layer in found])
                  for name in names}
    # A storage array binds at least one element: an empty one holds a filler that no primitive reads.
    fillers = dict(conics=nothing, planes=everywhere, points=origin)
    arrays = {name: concatenate(getattr(parts, name)) if getattr(parts, name) else fillers[name].broadcast_to((1,))
              for name in fillers}
    image = shader()(window.shape, **arrays, phases=np.concatenate(parts.phases or [np.zeros(1)]),
                     primitives=primitives, corner=mv.point([window.left, window.top]), pixel=window.pixel,
                     background=np.asarray(background, dtype=float))["image"]
    return np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
