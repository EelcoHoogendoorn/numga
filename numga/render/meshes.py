"""Triangle surfaces in `PGA3D` rasterize on the GPU under `solids`'s directional lamp along parallel
orthographic rays, writing premultiplied RGBA colour and view-ray depth at every subpixel sample so
`solids` and `Scene` composite triangulated surfaces sample-by-sample with analytic quadrics and 3D
tubes.

Each face corner pairs its `Point` directly with the camera's planes and carries an intrinsic tangent
plane (`solids.Patch`) whose slopes against the camera's preimage steps determine surface lighting.
`flat` assigns each triangle's three-point join `a & b & c` to all three corners, whereas `smooth`
sums those area-weighted face joins onto shared vertices through topological incidence and its
adjugate. Surface appearance is parameterized by `Colour`, which bundles a WGSL `colour(...)`
function (`plain` for interpolated corner colours, `by_face` for per-face colours, or custom shaders
using barycentric coordinates `corner` and `oversampling`) with its Python constants.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from numga.topology import Topology
from numga.wgsl import Shader, samples_per_side

from . import solids
from .camera import placing
from .drawing import Draw
from .solids import (Line, Plane, Point, View, ga, lamp_shading, lighting, steps,
                     tangent_seen)


class Colour(NamedTuple):
    """Surface colour specification pairing WGSL `source` defining
    `fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32>`
    (returning straight RGBA from interpolated corner `values`, flat `face_values`, barycentric
    weights `corner`, and front-facing flag `front`) with compile-time Python `constants` bound
    alongside `oversampling`."""
    source: str
    constants: dict


def source(colour: str) -> str:
    """Return the WGSL vertex and fragment stages shading a triangle surface with `colour`,
    discarding zero-alpha samples and writing premultiplied RGBA colour alongside view depth."""
    return colour + """
var<storage, read> points: array<Point>;
var<storage, read> faces: array<vec3<u32>>;
// Values at each corner of each face, three to a face, and each face's own.
var<storage, read> values: array<vec4<f32>>;
var<storage, read> face_values: array<vec4<f32>>;
// Each face's tangent plane at its three corners, read by the picture for the face's slopes.
struct Corners {
    a: Tangent,
    b: Tangent,
    c: Tangent,
}
var<storage, read> normals: array<Corners>;
// `oversampling`, a constant given from Python: how many drawn pixels make one of the image's, across and up; a
// colour function inking a line some pixels wide by `fwidth` scales it by this.

// Interpolated at every sample of a pixel, so its colour is supersampled along with its edges.
struct Corner {
    @builtin(position) position: vec4<f32>,
    @interpolate(perspective, sample) values: vec4<f32>,
    @interpolate(flat) face_values: vec4<f32>,
    @interpolate(perspective, sample) normal: Tangent,
    @interpolate(perspective, sample) corner: vec3<f32>,
    @interpolate(perspective, sample) depth: f32,
}

// A pixel's colour, and its depth along the view in the depth target's first channel.
struct Shade {
    @location(0) colour: vec4<f32>,
    @location(1) depth: vec4<f32>,
}

// A corner on the image, placed by the camera's planes; its depth along the view, how far it lies from the viewer.
@vertex
fn place(@builtin(vertex_index) index: u32) -> Corner {
    let face = faces[index / 3];
    let vertex = face[index % 3];
    let point = points[vertex];
    let corners = normals[index / 3];
    let normal = select(select(corners.c, corners.b, index % 3 == 1), corners.a, index % 3 == 0);
    let corner = vec3<f32>(f32(index % 3 == 0), f32(index % 3 == 1), f32(index % 3 == 2));
    return Corner(clip(point), values[index], face_values[index / 3], normal, corner,
                  -paired(screen_toward, point) / weight_of(point));
}

// A pixel of a face: its colour, lit by the lamp through its normal blended across the face, with a glint.
@fragment
fn shade(corner: Corner) -> Shade {
    // Whether the camera sees the side the normal points to.
    let front = slopes(corner.normal).z >= 0;
    let surface = colour(corner.values, corner.face_values, corner.corner, front);
    if (surface.a <= 0) {
        discard;
    }
    let shaded = lit(surface.rgb, corner.normal);
    return Shade(vec4<f32>(shaded * surface.a, surface.a), vec4<f32>(corner.depth, 0.0, 0.0, 1.0));
}
"""


# The vertices' own colours, blended across each face.
plain = Colour("""
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {
    return vec4<f32>(values.rgb, 1.0);
}
""", {})


# Each face's own colour.
by_face = Colour("""
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {
    return vec4<f32>(face_values.rgb, 1.0);
}
""", {})


# --- plumbing -------------------------------------------------------------------------
# The compiled mesh shaders, by the WGSL of their colour and the bytes of its constants.
shaders = {}                                                                    # dict[tuple, Shader]


def shader(colour: Colour) -> Shader:
    key = (colour.source, tuple((name, np.asarray(value).tobytes()) for name, value in colour.constants.items()))
    if key not in shaders:
        # It writes its colour, premultiplied by how much it hides, and its depth, as `drawing.NEAREST` keeps them.
        shaders[key] = Shader(lamp_shading() + tangent_seen() + placing(Plane) + source(colour.source), ga,
                              dict(Point=Point, Plane=Plane, Tangent=solids.Patch), dict(oversampling=float(samples_per_side())) | colour.constants)
    return shaders[key]


def flat(points: Point, faces: np.ndarray) -> dict:
    """Assign each triangle's joined tangent plane (`[faces] Patch-`) identically to all three of its
    corners `a`, `b`, and `c`."""
    corners = points[faces]
    joined = corners[:, 0] & corners[:, 1] & corners[:, 2]                    # [faces] Patch-
    return dict(a=joined, b=joined, c=joined)


def smooth(points: Point, topology: Topology) -> dict:
    """Sum the joined tangent planes of incident triangles around each vertex of `topology` (so
    larger triangles carry proportionally greater weight) and gather the resulting smooth `Patch-`
    planes onto face corners `a`, `b`, and `c`."""
    # Topological incidence holds plain numbers, which take the scene's algebra where they meet its points.
    incidence = topology.incidence[2, 0]
    corners = incidence * points.field()                                    # [3] Point-[faces]
    joined = corners[0] & corners[1] & corners[2]                            # Patch-[faces]
    # Read incidence backwards to sum each vertex's neighbouring tangents, then forwards to its face corners.
    tangents = incidence.sum(axis=0).adjugate() * joined                     # Patch-[vertices]
    corners = (incidence * tangents).batch()                                 # [3, faces] Patch-
    return dict(a=corners[0], b=corners[1], c=corners[2])


def samples(points: Point, faces: np.ndarray, values: np.ndarray, normals: dict, view: View,
            face_values: np.ndarray, colour: Colour, gloss: float) -> Draw:
    """Build a supersampled `Draw` pass for `faces` (`[F, 3]`) over `points` (`[V]`) with corner
    `values` (`[F, 3, C]` or broadcastable, `C <= 4`), per-face `face_values` (`[F, C]` or
    broadcastable), corner `normals`, surface `colour`, and specular `gloss` for depth-ordered
    compositing in `solids.layer` or `Scene`."""
    def padded(entries: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        entries = np.broadcast_to(entries, shape + (np.shape(entries)[-1],))
        return np.concatenate([entries, np.zeros(shape + (4 - entries.shape[-1],))], axis=-1)
    return Draw(shader(colour), view.shape, 3 * len(faces),
                   dict(points=points, faces=faces, values=padded(values, (len(faces), 3)).reshape(-1, 4),
                        face_values=padded(face_values, (len(faces),)), normals=normals,
                        **view.to_camera().uniforms(8 * view.half.max()), **lighting(gloss), **steps(view)))


def layer(points: Point, faces: np.ndarray, values: np.ndarray, normals: dict, view: View,
          face_values: np.ndarray, colour: Colour, gloss: float) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize and supersample a triangle surface in `view`, returning coverage-weighted RGBA
    colour `[rows, columns, 4]` and view depth `[rows, columns]`."""
    return solids.layer([], view, inks=[], translucent=[],
                        meshes=[(samples(points, faces, values, normals, view, face_values, colour, gloss), 1.0)])


def circles(rims: Point, pivots: Line, n_points: int) -> Point:
    """Generate closed circle loops `[..., n_points] Point` by rotating each rim point in `rims`
    (`[...]`) in `n_points` equal steps around its corresponding axis line in `pivots` (`[...]`)."""
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)                       # [n]
    turns = (pivots.normalized()[..., None] * (angles / 2)).exp()                       # [..., n] Motor
    return turns >> rims[..., None]


def draw(points: Point, faces: np.ndarray, values: np.ndarray, normals: dict, view: View, background: np.ndarray,
         face_values: np.ndarray, colour: Colour, gloss: float) -> np.ndarray:
    """Render a lit triangle surface over RGB `background` as `uint8` RGB `[rows, columns, 3]`."""
    return solids.draw([], view, background, inks=[], translucent=[],
                       meshes=[(samples(points, faces, values, normals, view, face_values, colour, gloss), 1.0)])
