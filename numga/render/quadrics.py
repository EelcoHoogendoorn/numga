"""Implicit quadric regions, conics, and linear flats are evaluated directly on the GPU in the
scene's own algebra without introducing a metric.

Each subpixel sample maps back into the scene through `camera`'s least-squares `preimage`, yielding
`point = origin + x * step_across + y * step_up` in the complementary grade and opposite orientation
of `camera`'s planes. Every quadric (`Form <- Point`, of either output orientation) and optional
linear flat (`Form`) evaluates `(quadric(point) & point) + (flat & point)` with the extrinsic factor
placed first, so conics and lines render identically across projective planes, vector planes, and
spacetime diagrams.

A `REGION` layer covers samples where all `bounds` forms of a region are negative, whereas a `CURVE`
layer (`curves` and `flats`) anti-aliases the zero set to a constant thickness of `width` pixels by
dividing the form's value by its screen-space variation `fwidth`, making curve width independent of
overall form sign or scale.
"""
from __future__ import annotations

from enum import IntEnum
from functools import cache

import numpy as np

from numga import NumpyContext
from numga.extensor import Extensor
from numga.gatype import GAType
from numga.wgsl import Shader, level_lines, samples_per_side

from .camera import Camera, extrinsic_first, placing, preimage, whole
from .drawing import Draw


class Kind(IntEnum):
    """Coverage mode selecting negative-interior filling (`REGION`) or zero-set stroking (`CURVE`)."""
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
    @interpolate(flat) index: u32,
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
    """Compile and cache the implicit-quadric shader for scene points `Point`, camera planes `Plane`,
    and quadric output planes `Form` (of either orientation, paired extrinsic first)."""
    pairing = "plane & point" if extrinsic_first(Form) else "point & plane"
    form_pairing = f"""
// A plane of the forms paired with a point: a form's value there, the extrinsic first.
fn form_paired(plane: Form, point: Point) -> f32 {{
    return {pairing};
}}
"""
    return Shader(level_lines() + placing(Plane) + form_pairing + source, Point.algebra,
                  dict(Point=Point, Plane=Plane, Form=Form, Quadric=Point.algebra.gatype((Form, Point))), {})


def layer(quadrics: Extensor, flats: Extensor, kind: Kind, width: float, colours: np.ndarray, camera: Camera,
          opacity: float) -> Draw:
    """Build a full-screen `Draw` pass evaluating `quadrics` (`[n, bounds]`) plus linear `flats`
    (`[n, bounds]`) over `camera` according to `kind`, with RGB `colours` (`[n, 3]` or `[3]`) and
    stroke `width` in pixels."""
    algebra = quadrics.algebra
    Plane = whole(camera.across, algebra)
    Point = points_read_by(Plane)
    origin, step_across, step_up, _ = preimage(camera, Point)
    side = samples_per_side()
    rows, columns = camera.shape
    return Draw(shader(Point, Plane, whole(flats, algebra)), camera.shape, 6 * len(quadrics), dict(
        quadrics=quadrics.reshape(-1), flats=flats.reshape(-1),
        bounds=quadrics.shape[1], colours=np.broadcast_to(colours, (len(quadrics), 3)), opacity=opacity, kind=float(kind),
        width=width * side, samples=np.array([columns * side, rows * side], dtype=float), origin=origin,
        step_across=step_across, step_up=step_up, **camera.uniforms(1.0)))


def points_read_by(Plane: GAType) -> GAType:
    """Return the full-grade `GAType` of complementary grade and opposite orientation paired by
    `Plane`."""
    algebra = Plane.algebra
    grade = bin(int(Plane.output_subspace.masks[0])).count("1")
    return algebra.gatype.from_grades([len(algebra.signature) - grade]).derive.with_orientations(
        Plane.derive.flipped_orientation.orientations)


def nothing(kind: GAType, shape: tuple[int, ...]) -> Extensor:
    """Return a zero-filled `Extensor` of `kind` with batch `shape`."""
    return NumpyContext(kind.algebra).extensor(kind, np.zeros(shape + kind.structural_shape))


def form_of(quadrics: Extensor) -> GAType:
    """Return the full-grade output `GAType` of `quadrics` with its orientation preserved."""
    output = quadrics.gatype.derive.output
    grade = bin(int(output.output_subspace.masks[0])).count("1")
    return quadrics.algebra.gatype.from_grades([grade]).derive.with_orientations(output.orientations)


def regions(quadrics: Extensor, colours: np.ndarray, camera: Camera, opacity: float = 1.0) -> Draw:
    """Draw `n` filled regions where all `bounds` forms in `quadrics` (`[n, bounds]`, mapping points
    to planes of either orientation) are negative, coloured by `colours` (`[n, 3]` or `[3]`)."""
    return layer(quadrics, nothing(form_of(quadrics), quadrics.shape), Kind.REGION, 0.0, colours, camera, opacity)


def curves(quadrics: Extensor, width: float, colours: np.ndarray, camera: Camera, opacity: float = 1.0) -> Draw:
    """Draw the zero-set curves of `quadrics` (`[n]`, mapping points to planes of either orientation)
    at `width` pixels wide in `colours` (`[n, 3]` or `[3]`), independent of overall form sign."""
    return layer(quadrics[:, None], nothing(form_of(quadrics), quadrics.shape + (1,)), Kind.CURVE, width, colours,
                 camera, opacity)


def flats(planes: Extensor, width: float, colours: np.ndarray, camera: Camera, opacity: float = 1.0) -> Draw:
    """Draw the zero-set lines of linear `planes` (`[n]`, of either orientation) across `camera` at
    `width` pixels wide in `colours` (`[n, 3]` or `[3]`)."""
    Form = whole(planes, planes.algebra)
    Point = points_read_by(whole(camera.across, planes.algebra))
    return layer(nothing(planes.algebra.gatype((Form, Point)), planes.shape + (1,)), planes[:, None], Kind.CURVE, width,
                 colours, camera, opacity)
