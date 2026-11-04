"""Quadrics drawn under a camera, in the scene's own algebra: regions where they are negative, curves where they vanish.

Every sample of the picture is read back into the scene by the camera's preimage, the point the camera reads there, and
each quadric, a map from points to planes, is evaluated there by its form `quadric(point) & point`, plus a linear part
`plane & point`, so that a flat is the curve of a linear form. A region, like a solid of `solids`, is the intersection of
a few quadrics, and covers the samples where all their forms are negative; a curve covers those within half its width
of where its form vanishes, the form's distance to zero over its change
across one sample, as `contours` draws a level line. No metric enters: the camera's readings measure the picture, so a
conic of a projective plane, of a vector plane or of a spacetime diagram is drawn alike, and the picture's samples, sixteen
to a pixel, smooth its edges.
"""
from __future__ import annotations

from enum import IntEnum
from functools import cache

import numpy as np

from numga import NumpyContext
from numga.extensor import Extensor
from numga.gatype import GAType
from examples.util.wgsl import Shader
from examples.util.wgsl.camera import Camera, extrinsic_first, placing, preimage, whole
from examples.util.wgsl.library import level_lines, samples_per_side
from examples.util.wgsl.simplices import Layer


class Kind(IntEnum):
    """What a layer of quadrics covers."""
    REGION = 0
    CURVE = 1


source = """
// The quadrics, maps from points to the planes of their forms, and their linear parts, such planes, `bounds` to a region
// or one to a curve; each region's or curve's colour, and how much it covers.
var<storage, read> quadrics: array<Quadric>;
var<storage, read> flats: array<Form>;
var<storage, read> colours: array<vec3<f32>>;
var<uniform> opacity: f32;
var<uniform> bounds: u32;
// What the quadrics cover, a `Kind`, and a curve's width, in samples.
var<uniform> kind: f32;
var<uniform> width: f32;
// The picture's samples across and down; the point the camera reads at the origin of its readings, and the steps it
// reads as one across and one up.
var<uniform> samples: vec2<f32>;
var<uniform> origin: Point;
var<uniform> step_across: Point;
var<uniform> step_up: Point;

struct Corner {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) index: u32,
}

// Each quadric a pair of triangles over the whole picture.
@vertex
fn place(@builtin(vertex_index) index: u32) -> Corner {
    let corner = array<vec2<f32>, 6>(vec2<f32>(-1, -1), vec2<f32>(1, -1), vec2<f32>(1, 1),
                                     vec2<f32>(-1, -1), vec2<f32>(1, 1), vec2<f32>(-1, 1))[index % 6];
    return Corner(vec4<f32>(corner, 0.5, 1.0), index / 6);
}

@fragment
fn shade(corner: Corner) -> @location(0) vec4<f32> {
    // The sample's readings across and up, and the point the camera reads there.
    let framed = vec2<f32>(corner.position.x / samples.x * 2.0 - 1.0, 1.0 - corner.position.y / samples.y * 2.0);
    let reading = screen_middle + screen_half * framed;
    let point = origin + step_across * reading.x + step_up * reading.y;
    // The forms at the point, with their linear parts: a curve's own and its change across one sample; a region's
    // largest, negative where all of them are.
    var value = -1e30;
    for (var bound = 0u; bound < bounds; bound++) {
        let index = corner.index * bounds + bound;
        value = max(value, form_paired(quadrics[index](point), point) + form_paired(flats[index], point));
    }
    let rate = fwidth(value);
    let covered = select(level_mark(value, rate, width), select(0.0, 1.0, value < 0.0), kind == 0.0);
    if (covered <= 0.0) { discard; }
    return vec4<f32>(colours[corner.index] * opacity * covered, opacity * covered);
}
"""


@cache
def shader(Point: GAType, Plane: GAType, Form: GAType) -> Shader:
    """The quadrics that map points of the type `Point`, read by a camera of planes of the type `Plane`, to planes of
    the type `Form`, of either orientation: their form a plane's pairing with a point, the extrinsic first."""
    pairing = "plane & point" if extrinsic_first(Form) else "point & plane"
    form_pairing = f"""
// A plane of the forms paired with a point: a form's value there, the extrinsic first.
fn form_paired(plane: Form, point: Point) -> f32 {{
    return {pairing};
}}
"""
    return Shader(level_lines() + placing(Plane) + form_pairing + source, Point.algebra,
                  dict(Point=Point, Plane=Plane, Form=Form, Quadric=Point.algebra.gatype((Form, Point))), {},
                  targets=("rgba8unorm",), over=True)


def layer(quadrics: Extensor, flats: Extensor, kind: Kind, width: float, colours: np.ndarray, camera: Camera,
          opacity: float) -> Layer:
    """Quadrics `[n, bounds]`, maps from points to planes, with their linear parts, planes `[n, bounds]`, covering what
    `kind` says, read by the camera: a layer of colour and coverage."""
    algebra = quadrics.algebra
    Plane = whole(camera.across, algebra)
    Point = points_read_by(Plane)
    origin, step_across, step_up, _ = preimage(camera, Point)
    side = samples_per_side()
    rows, columns = camera.shape
    return Layer(shader(Point, Plane, whole(flats, algebra)), camera.shape, 6 * len(quadrics), dict(
        quadrics=quadrics.reshape(-1), flats=flats.reshape(-1),
        bounds=quadrics.shape[1], colours=np.broadcast_to(colours, (len(quadrics), 3)), opacity=opacity, kind=float(kind),
        width=width * side, samples=np.array([columns * side, rows * side], dtype=float), origin=origin,
        step_across=step_across, step_up=step_up, **camera.uniforms(1.0)))


def points_read_by(Plane: GAType) -> GAType:
    """The points that planes of the type `Plane` read: the complementary grade, of the other orientation."""
    algebra = Plane.algebra
    grade = bin(int(Plane.output_subspace.masks[0])).count("1")
    return algebra.gatype.from_grades([len(algebra.signature) - grade]).derive.with_orientations(
        Plane.derive.flipped_orientation.orientations)


def nothing(kind: GAType, shape: tuple[int, ...]) -> Extensor:
    """Zero of a type, `shape` of them: the part a layer's quadrics or flats lack."""
    return NumpyContext(kind.algebra).extensor(kind, np.zeros(shape + kind.structural_shape))


def form_of(quadrics: Extensor) -> GAType:
    """The planes quadrics map points to: the whole of their grade, at their own orientation."""
    output = quadrics.gatype.derive.output
    grade = bin(int(output.output_subspace.masks[0])).count("1")
    return quadrics.algebra.gatype.from_grades([grade]).derive.with_orientations(output.orientations)


def regions(quadrics: Extensor, colours: np.ndarray, camera: Camera, opacity: float = 1.0) -> Layer:
    """Regions `[n]`, each where its quadrics `[n, bounds]`, maps from points to planes of either orientation, are all
    negative, each in its colour `[n, 3]` or all in one `[3]`: their picture."""
    return layer(quadrics, nothing(form_of(quadrics), quadrics.shape), Kind.REGION, 0.0, colours, camera, opacity)


def curves(quadrics: Extensor, width: float, colours: np.ndarray, camera: Camera, opacity: float = 1.0) -> Layer:
    """The curves where quadrics `[n]`, maps from points to planes of either orientation, vanish, `width` pixels across,
    each in its colour `[n, 3]` or all in one `[3]`: their picture. A curve is a zero set, the same whatever the form's
    sign, so a quadric of a join draws its curve as it is."""
    return layer(quadrics[:, None], nothing(form_of(quadrics), quadrics.shape + (1,)), Kind.CURVE, width, colours,
                 camera, opacity)


def flats(planes: Extensor, width: float, colours: np.ndarray, camera: Camera, opacity: float = 1.0) -> Layer:
    """Flats `[n]`, planes or lines of either orientation, where a plane's pairing with a point vanishes, across the
    whole picture, `width` pixels across, each in its colour `[n, 3]` or all in one `[3]`: their picture."""
    Form = whole(planes, planes.algebra)
    Point = points_read_by(whole(camera.across, planes.algebra))
    return layer(nothing(planes.algebra.gatype((Form, Point)), planes.shape + (1,)), planes[:, None], Kind.CURVE, width,
                 colours, camera, opacity)
