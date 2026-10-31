"""Pictures of space, drawn on the GPU: PGA2D geometry in, the RGB bytes of a window onto space out.

A window is a rectangle of space and its rows and columns of square pixels. Each pixel's centre is a point of
space, and every primitive covers a share of the pixel by a pairing with that point, worked out analytically, so one
point a pixel draws soft edges:

- a quadric, a map `Plane <- Point`, drawn about the radius it is drawn at: the distance between that radius, a plane
  paired with the point and held above a floor, and the square root of `quadric(p) & p` less an offset, eased over three
  quarters of a pixel; only on the positive side of any of its clip planes;
- a region, where `quadric(p) & p` is negative: its value over its polar's slope, `quadric(p)`, is the distance to its
  edge;
- a curve, where `quadric(p) & p` vanishes, stroked about that distance; an envelope, a map from planes to the points
  they touch, is drawn as the curve of its inverse, the quadric of its points;
- a flat, a plane or a line, whole across the window: the flat at unit weight paired with the point is the distance
  across it;
- a polygon, where its edges wind about the point: each triangle of the fan from its first corner counts with its
  sign, on the inner side of each edge's line, the line at unit weight paired with the point its distance;
- a stroke along the segments of a polyline: each segment's line at unit weight paired with the point is the distance
  across it, and the plane square to it through its start paired with the point the distance along it, on dashes of a
  pattern along the polyline, each dash ended by its cap and its segments joined round;
- a mark about a point: a disc, by the join of the two points, whose norm is their distance; a square or a diamond, by
  the pairings of two square planes through it.

Across an edge, a primitive covers the share of the pixel's width on its side. The primitives are composited in the
order given, each in its colour, by its opacity times its share of the pixel. Points of either orientation are drawn
alike, and so are planes and the lines of PGA2D, which share their grade.
"""

from __future__ import annotations

from enum import IntEnum
from functools import cache
from itertools import groupby
from typing import NamedTuple

import numpy as np

from numga import Extensor, NumpyContext, concatenate, stack
from numga.algebras import PGA2D as ga
from numga.gatype import GAType
from examples.util.wgsl import Shader, simplices
from examples.util.wgsl.camera import Camera
from examples.util.wgsl.simplices import Cap, Marker

context = NumpyContext(ga)
mv = context.multivector
Point = ga.gatype.antivector()                                               # Antivector-
Plane = ga.gatype.vector()                                                   # Vector+
Scalar = ga.gatype.scalar()                                                  # Scalar+
Quadric = ga.gatype((Plane, Point))                                          # Plane+ <- Point-
Envelope = ga.gatype((Point, Plane))                                         # Point- <- Plane+
origin = mv.w.dual()                                                         # [] Point-
# Everywhere, as a clip plane: the plane at infinity pairs every point of unit weight to one.
everywhere = mv.w                                                            # [] Plane+
# A quadric that covers no point of unit weight.
nothing = (mv.w * (mv.w & Point)).cast(Quadric)                              # [] Plane+ <- Point-
# The intrinsic scalar one: a product with it keeps every coefficient and flips only the orientation.
flip = mv.antipseudoscalar([1.0])                                            # [] Antipseudoscalar-
# A solid stroke: one dash, no gap.
SOLID = (1.0, 0.0)


class Kind(IntEnum):
    """What a primitive is, as the shader reads it."""
    QUADRIC = 0
    REGION = 1
    CURVE = 2
    FLAT = 3
    POLYGON = 4


source = """
// A primitive, composited in order, of its kind, and where its parts lie: `first` among the quadrics for a quadric, a
// region or a curve, among the planes for a flat, among the points for a mark, and its points from `first` up to
// `end` for a polygon or a stroke. A quadric, drawn about the radius `planes[plane_first]` and above its floor, less its
// offset, on the positive side of any of the clip planes after it up to `planes[plane_end]`; a curve or a flat,
// `width` across; a stroke, its width, the lengths of its dashes and of the gaps between them, in units of length,
// and its cap, its `shape`; a mark, its marker, its `shape`, covering from `inner` to `outer` from its point. Its
// colour, and how much of what lies under it it hides.
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
var<storage, read> quadrics: array<Quadric>;
var<storage, read> planes: array<Plane>;
var<storage, read> points: array<Point>;
// How far along its polyline each point of a stroke lies.
var<storage, read> phases: array<f32>;
// The window's top left corner, a pixel's width, and what lies under every primitive at each pixel.
var<uniform> corner: Point;
var<uniform> pixel: f32;
var<storage, read> background: array<vec3<f32>>;
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

fn quadric_share(primitive: Primitive, point: Point) -> f32 {
    var shown = false;
    for (var index = primitive.plane_first + 1; index < primitive.plane_end; index++) {
        shown = shown || (planes[index] & point) > 0;
    }
    let value = max((quadrics[primitive.first](point) & point) - primitive.offset, 0.0);
    let distance = max(planes[primitive.plane_first] & point, primitive.floor) - sqrt(value);
    return select(0.0, 1 / (1 + exp(clamp(-distance / (0.75 * pixel), -30.0, 30.0))), shown);
}

// Inside a region by its value over twice the slope of its polar: the distance to its edge, to first order.
fn region_share(index: u32, point: Point) -> f32 {
    let polar = quadrics[index](point);
    return inner(-(polar & point) / (2 * polar.norm()));
}

// Within half the width of a curve by the same distance; nowhere where the polar has no slope.
fn curve_share(primitive: Primitive, point: Point) -> f32 {
    let polar = quadrics[primitive.first](point);
    let slope = 2 * polar.norm();
    let distance = (polar & point) / slope;
    return select(overlap(-primitive.width / 2, primitive.width / 2, distance), 0.0, slope == 0);
}

fn flat_share(primitive: Primitive, point: Point) -> f32 {
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
// two ends: the plane square to the edge through its first end, pointing to its other end, paired with the point.
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

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= pixels.x || id.y >= pixels.y) {
        return;
    }
    // The pixel's centre: right and down from the window's top left corner.
    let point = corner + (mv.x.dual() * (f32(id.x) + 0.5) - mv.y.dual() * (f32(id.y) + 0.5)) * pixel;
    var colour = background[id.y * pixels.x + id.x];
    for (var index = 0u; index < arrayLength(&primitives); index++) {
        let primitive = primitives[index];
        var share = 0.0;
        if (primitive.kind == kind_quadric) {
            share = quadric_share(primitive, point);
        } else if (primitive.kind == kind_region) {
            share = region_share(primitive.first, point);
        } else if (primitive.kind == kind_curve) {
            share = curve_share(primitive, point);
        } else if (primitive.kind == kind_flat) {
            share = flat_share(primitive, point);
        } else {
            share = polygon_share(primitive, point);
        }
        colour = mix(colour, primitive.colour, primitive.opacity * share);
    }
    image[id.y * pixels.x + id.x] = colour;
}
"""


@cache
def shader() -> Shader:
    # Each kind as a constant of the shader, `kind_quadric`.
    constants = {f"kind_{member.name.lower()}": member.value for member in Kind}
    return Shader(source, ga, dict(Point=Point, Plane=Plane, Quadric=Quadric), constants)


class Window(NamedTuple):
    """A rectangle of space: its top left corner, a pixel's width, and its rows and columns of square pixels."""
    corner: Point
    pixel: float
    shape: tuple[int, int]

    @property
    def left(self) -> float:
        return float((mv.x & self.corner).to_array())

    @property
    def top(self) -> float:
        return float((mv.y & self.corner).to_array())

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Its left, right, bottom and top, as matplotlib places an image."""
        rows, columns = self.shape
        return self.left, self.left + columns * self.pixel, self.top - rows * self.pixel, self.top


def window(x_range: tuple[float, float], y_range: tuple[float, float], rows: int) -> Window:
    """The window `rows` pixels down `y_range`, as many square pixels across `x_range`, about its middle, as fit."""
    pixel = (y_range[1] - y_range[0]) / rows
    columns = round((x_range[1] - x_range[0]) / pixel)
    corner = origin + mv.x.dual() * (sum(x_range) / 2 - columns * pixel / 2) + mv.y.dual() * y_range[1]
    return Window(corner, pixel, (rows, columns))


def filling(x_range: tuple[float, float], y_range: tuple[float, float], room: tuple[int, int]) -> Window:
    """The window of square pixels about the middle of `x_range` and `y_range` that they fill one way of a room of
    columns and rows, the other way as many pixels as they span."""
    pixel = max((x_range[1] - x_range[0]) / room[0], (y_range[1] - y_range[0]) / room[1])
    rows, columns = round((y_range[1] - y_range[0]) / pixel), round((x_range[1] - x_range[0]) / pixel)
    centre = origin + mv.x.dual() * (sum(x_range) / 2) + mv.y.dual() * (sum(y_range) / 2)
    corner = centre - mv.x.dual() * (columns * pixel / 2) + mv.y.dual() * (rows * pixel / 2)
    return Window(corner, pixel, (rows, columns))


def view(points: list[Extensor], margin: float, room: tuple[int, int]) -> Window:
    """The window, as `filling` makes it for a room of columns and rows, that holds batches of points of either
    orientation and any weight, with a margin as a share of their larger extent: framed by the support lines square
    to x and y, the lines along the axes that touch the points on either side."""
    seen = concatenate([unit(batch).reshape(-1) for batch in points])        # [points] Point
    across = stack((mv.x, mv.y))                                              # [2] Plane
    heights = across[:, None] & seen                                          # [2, points] Scalar
    low, high = heights.min(axis=-1), heights.max(axis=-1)                    # [2] Scalar+, Scalar+
    spans = high - low                                                       # [2] Scalar+
    spread = margin * spans.max()
    middles = across - mv.w * ((low + high) / 2)                              # [2] Plane+
    centre = (middles[0] ^ middles[1]) * flip                                 # [] Point-
    spans = spans + 2 * spread                                               # [2] Scalar+
    pixels = spans / np.asarray(room)                                        # [2] Scalar+
    pixel = pixels.max()                                         # [] Scalar+
    columns, rows = np.rint((spans / pixel).to_array()).astype(int)
    corner = centre - mv.x.dual() * (columns * pixel / 2) + mv.y.dual() * (rows * pixel / 2)
    return Window(corner, float(pixel.to_array()), (rows, columns))


def oriented(value: Extensor, kind: GAType) -> Extensor:
    """A value as `kind`, through the intrinsic scalar one where their orientations differ."""
    return (value if value.gatype.orientation is kind.orientation else value * flip).cast(kind)


def point_quadrics(maps: Extensor) -> Quadric:
    """Maps between points and planes, or lines, of either orientation as maps `Plane <- Point`: each side through
    the intrinsic scalar one where its orientation differs."""
    if maps.gatype.input_orientations[0] is not Point.orientation:
        maps = maps(Point * flip)
    return oriented(maps, Quadric)


def unit(points: Extensor) -> Point:
    """Points `[...]` of either orientation, at unit weight, as `Point`."""
    return oriented(points / (mv.w & points), Point)


class Parts:
    """The quadrics, planes, points and the points' phases that primitives draw from, gathered as layers add them."""

    def __init__(self) -> None:
        self.quadrics, self.planes, self.points, self.phases = [], [], [], []

    @staticmethod
    def add(held: list, values) -> int:
        """Where values `[m]` start among those held with them."""
        first = sum(part.shape[0] for part in held)
        held.append(values)
        return first

    def add_points(self, points: Point, phases: Scalar) -> int:
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


class Quadrics(NamedTuple):
    """Quadrics `[n]` drawn about a radius: covering a pixel by its centre's distance inside the radius, a plane `[n]`
    paired with the point, at least the floor `[n]`, beyond the square root of `quadric(p) & p` less the offset `[n]`,
    eased over three quarters of a pixel; and only on the positive side of any of the clip planes `[n, clips]`,
    `everywhere` for no clip. One colour `[3]` or one each `[n, 3]`, and the opacity of all."""
    quadrics: Quadric
    offsets: np.ndarray
    radii: Plane
    floors: np.ndarray
    clips: Plane
    colour: np.ndarray
    opacity: float

    def primitives(self, parts: Parts, pixel: float) -> dict:
        # Each quadric's radius, then its clip planes.
        size, n_clips = self.quadrics.shape[0], self.clips.shape[-1]
        radii = oriented(self.radii, Plane).broadcast_to(self.quadrics.shape)[:, None]
        clips = oriented(self.clips, Plane).broadcast_to(self.quadrics.shape + (n_clips,))
        plane_first = parts.add(parts.planes, concatenate([radii, clips], axis=1).reshape(size * (n_clips + 1)))
        plane_firsts = plane_first + (n_clips + 1) * np.arange(size)
        first = parts.add(parts.quadrics, point_quadrics(self.quadrics)) + np.arange(size)
        return fields(Kind.QUADRIC, first, self.colour, self.opacity, plane_first=plane_firsts,
                      plane_end=plane_firsts + n_clips + 1, offset=self.offsets, floor=self.floors)


class Regions(NamedTuple):
    """The regions where quadrics `[n]` are negative; one colour `[3]` or one each `[n, 3]`, and the opacity of all."""
    quadrics: Quadric
    colour: np.ndarray
    opacity: float

    def primitives(self, parts: Parts, pixel: float) -> dict:
        first = parts.add(parts.quadrics, point_quadrics(self.quadrics)) + np.arange(self.quadrics.shape[0])
        return fields(Kind.REGION, first, self.colour, self.opacity)


class Curves(NamedTuple):
    """The curves where quadrics `[n]`, maps from points to planes, vanish, stroked `width` pixels across; one colour
    `[3]` or one each `[n, 3]`, and the opacity of all."""
    quadrics: Quadric
    width: float
    colour: np.ndarray
    opacity: float

    def primitives(self, parts: Parts, pixel: float) -> dict:
        # A unit largest gain keeps the same curve while giving its form a common scale.
        quadrics = point_quadrics(self.quadrics)
        quadrics = quadrics / quadrics.svdvals()[..., 0]
        first = parts.add(parts.quadrics, quadrics) + np.arange(quadrics.shape[0])
        return fields(Kind.CURVE, first, self.colour, self.opacity, width=self.width * pixel)


class Envelopes(NamedTuple):
    """The curves that envelopes `[n]`, maps from planes to points, touch, stroked `width` pixels across: the curves of
    their inverses, the quadrics of their points. One colour `[3]` or one each `[n, 3]`, and the opacity of all."""
    envelopes: Envelope
    width: float
    colour: np.ndarray
    opacity: float

    def primitives(self, parts: Parts, pixel: float) -> dict:
        return Curves(self.envelopes.inverse(), self.width, self.colour, self.opacity).primitives(parts, pixel)


class Flats(NamedTuple):
    """Flats `[n]`, planes or lines, whole across the window, stroked `width` pixels across; one colour `[3]` or one
    each `[n, 3]`, and the opacity of all."""
    flats: Plane
    width: float
    colour: np.ndarray
    opacity: float

    def primitives(self, parts: Parts, pixel: float) -> dict:
        first = parts.add(parts.planes, oriented(self.flats, Plane)) + np.arange(self.flats.shape[0])
        return fields(Kind.FLAT, first, self.colour, self.opacity, width=self.width * pixel)


class Polygons(NamedTuple):
    """Filled polygons, their corners `[n, k]`, covering the points their edges wind about; one colour `[3]` or one
    each `[n, 3]`, and the opacity of all."""
    corners: Point
    colour: np.ndarray
    opacity: float

    def primitives(self, parts: Parts, pixel: float) -> dict:
        size, n_corners = self.corners.shape
        first = parts.add_points(unit(self.corners).reshape(size * n_corners), mv.scalar(np.zeros((size * n_corners, 1))))
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

    def picture(self, camera: Camera) -> np.ndarray:
        """The polylines drawn by the simplex rasteriser, each a chain of edges: colour and coverage."""
        size, n_points = self.points.shape
        colours = np.repeat(np.broadcast_to(self.colour, (size, 3)), n_points - 1, axis=0)
        return simplices.strokes(unit(self.points).reshape(-1), simplices.Mesh.chains([n_points] * size), colours,
                                 self.width, camera, dashes=self.dashes, cap=self.cap, opacity=self.opacity)


class Marks(NamedTuple):
    """Marks about points `[n]`, each covering from `inner` to `outer` pixels from its point, across the marker's
    outline: filled from 0, a ring from further out. One colour `[3]` or one each `[n, 3]`, and the opacity of all."""
    points: Point
    marker: Marker
    inner: float
    outer: float
    colour: np.ndarray
    opacity: float

    def picture(self, camera: Camera) -> np.ndarray:
        """The marks drawn by the simplex rasteriser: colour and coverage."""
        return simplices.markers(unit(self.points), self.colour, self.outer, camera, inner=self.inner,
                                 marker=self.marker, opacity=self.opacity)


def window_camera(window: Window) -> Camera:
    """The window as a camera of the plane: read by the planes x and y, its middle and half its extent in them."""
    rows, columns = window.shape
    half = np.array([columns, rows]) * window.pixel / 2
    middle = np.array([window.left, window.top]) + half * np.array([1, -1])
    return Camera(mv.x, mv.y, 0 * mv.x, mv.w, 0.0, middle, half, columns)


def draw(window: Window, layers: list, background: np.ndarray) -> np.ndarray:
    """The layers, `Quadrics`, `Regions`, `Curves`, `Envelopes`, `Flats`, `Polygons`, `Strokes` and `Marks`, composited
    in order over the background, a colour `[3]` or a picture of the window `[rows, columns, 3]`, each primitive of a
    layer after the one before: the window's RGB bytes `[rows, columns, 3]`. Strokes and marks are drawn by the simplex
    rasteriser, the rest pixel by pixel here; each run of the rest is drawn over the picture so far."""
    image = np.broadcast_to(np.asarray(background, dtype=float), window.shape + (3,))
    for rasterised, run in groupby(layers, key=lambda layer: hasattr(layer, "picture")):
        if rasterised:
            for layer in run:
                picture = layer.picture(window_camera(window))
                image = image * (1 - picture[..., 3:]) + picture[..., :3] * picture[..., 3:]
        else:
            image = traced(window, list(run), image)
    return np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)


def traced(window: Window, layers: list, background: np.ndarray) -> np.ndarray:
    """The layers drawn pixel by pixel over a picture of the window `[rows, columns, 3]`: the new picture."""
    parts = Parts()
    found = [layer.primitives(parts, window.pixel) for layer in layers]
    # Every field of every primitive; those a kind does not read are zero.
    names = ("kind", "first", "end", "plane_first", "plane_end", "shape", "colour", "opacity", "offset", "floor",
             "width", "dash", "gap", "inner", "outer")
    primitives = {name: np.concatenate([layer.get(name, np.zeros(len(layer["first"]), dtype=int)) for layer in found])
                  for name in names}
    # A storage array binds at least one element: an empty one holds a filler that no primitive reads.
    fillers = dict(quadrics=nothing, planes=everywhere, points=origin)
    arrays = {name: concatenate(getattr(parts, name)) if getattr(parts, name) else fillers[name].broadcast_to((1,))
              for name in fillers}
    image = shader()(window.shape, **arrays, phases=concatenate(parts.phases or [mv.scalar([[0]])]),
                     primitives=primitives, corner=window.corner, pixel=window.pixel,
                     background=np.broadcast_to(np.asarray(background, dtype=float), window.shape + (3,)))["image"]
    return image
