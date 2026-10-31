"""Triangle meshes in PGA3D, rasterized on the GPU: lit by the lamp of `solids`, seen by its camera along
parallel rays, on a plain background.

Each corner of a face reads its point, its colour and the face's shading normal there, and places itself on
the image by its pairings with the camera's own planes; the normals are flat, the face's own, or smooth, the
faces around each vertex averaged, as the caller builds them.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from numga.sparse import SparseExtensor
from examples import instantiate
from examples.util.wgsl import Shader
from examples.util.wgsl import solids
from examples.util.wgsl.library import samples_per_side
from examples.util.wgsl.camera import placing
from examples.util.wgsl.solids import (Line, Plane, Point, View, flip, ga, lamp_shading, lighting, mv, origin, steps,
                                       tangent_seen)

Mesh = instantiate("examples.util.mesh", solids.ga).Mesh


class Colour(NamedTuple):
    """How a mesh is coloured: WGSL defining `fn colour(values: vec4<f32>, face_values: vec4<f32>, corner:
    vec3<f32>, front: bool) -> vec4<f32>`, the colour and how much of what lies behind it it hides, from each pixel's
    values, the corners' blended across the face, the face's own, how near the pixel lies to each corner, and whether
    the camera sees the side the face's normal points to; and the constants it names, numbers and vectors of two to
    four numbers given from Python by name, beside the mesh shader's own `oversampling`."""
    source: str
    constants: dict


def source(colour: str) -> str:
    """WGSL drawing a mesh, coloured by `colour`, the WGSL of a `Colour`. It writes the colour and the depth, each
    weighted by how much of a pixel the mesh covers and by how much it hides; a pixel that hides nothing is not drawn,
    and leaves what lies behind it to be seen."""
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
    @location(0) @interpolate(perspective, sample) values: vec4<f32>,
    @location(1) @interpolate(flat) face_values: vec4<f32>,
    @location(2) @interpolate(perspective, sample) normal: Tangent,
    @location(3) @interpolate(perspective, sample) corner: vec3<f32>,
    @location(4) @interpolate(perspective, sample) depth: f32,
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
        # The colour a byte a channel, premultiplied by how much it hides, and the depth alone at full precision.
        shaders[key] = Shader(lamp_shading() + tangent_seen() + placing(Plane) + source(colour.source), ga,
                              dict(Point=Point, Plane=Plane, Tangent=solids.Patch), dict(oversampling=float(samples_per_side())) | colour.constants,
                              targets=("rgba8unorm", "r32float"))
    return shaders[key]


def flat(points: Point, faces: np.ndarray) -> dict:
    """Each face's tangent plane at all three corners: the join of its corners, as it is."""
    corners = points[faces]
    joined = corners[:, 0] & corners[:, 1] & corners[:, 2]                    # [faces] Patch-
    return dict(a=joined, b=joined, c=joined)


def smooth(points: Point, mesh: Mesh) -> dict:
    """At each corner, the tangent planes of the faces around its vertex summed: each face's the join of its corners,
    so that a larger face counts for more, as the picture's lighting reads only their slopes."""
    # The mesh's incidence has scalar cells: carry them into the scene's algebra with its topology intact.
    incidence = mesh.incidence[2, 0]
    incidence = SparseExtensor(solids.context.include_from_algebra(incidence.cells), incidence.rows, incidence.columns, incidence.shape)
    corners = incidence * points.field()                                    # [3] Point-[faces]
    joined = corners[0] & corners[1] & corners[2]                            # Patch-[faces]
    # Read incidence backwards to sum each vertex's neighbouring tangents, then forwards to its face corners.
    tangents = incidence.sum(axis=0).reverse() * joined                      # Patch-[vertices]
    corners = (incidence * tangents).batch()                                 # [3, faces] Patch-
    return dict(a=corners[0], b=corners[1], c=corners[2])


class Samples(NamedTuple):
    """A mesh at every sample of a view's pixels, `oversampling` to a side, drawn on the GPU by the picture that
    composes it, where its samples stay: the shader drawing it, the vertices of its triangles, three to a face, and the
    values the shader reads."""
    shader: Shader
    n_vertices: int
    values: dict


def samples(points: Point, faces: np.ndarray, values: np.ndarray, normals: dict, view: View,
            face_values: np.ndarray, colour: Colour, gloss: float) -> Samples:
    """The mesh at every sample of the view's pixels, for `solids.layer` to compose: its colour, premultiplied by how
    much it hides, and its depth along the view, both zero where it shows nothing; from up to four values at each
    corner of each face, `[faces, 3, values]` or one row for all, and at each face, coloured by `colour`, with a glint
    of `gloss`."""
    def padded(entries: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        entries = np.broadcast_to(entries, shape + (np.shape(entries)[-1],))
        return np.concatenate([entries, np.zeros(shape + (4 - entries.shape[-1],))], axis=-1)
    from examples.util.wgsl.simplices import view_camera
    return Samples(shader(colour), 3 * len(faces),
                   dict(points=points, faces=faces, values=padded(values, (len(faces), 3)).reshape(-1, 4),
                        face_values=padded(face_values, (len(faces),)), normals=normals,
                        **view_camera(view).uniforms(8 * view.half.max()), **lighting(gloss), **steps(view)))


def layer(points: Point, faces: np.ndarray, values: np.ndarray, normals: dict, view: View,
          face_values: np.ndarray, colour: Colour, gloss: float) -> tuple[np.ndarray, np.ndarray]:
    """The mesh as the view sees it: its colour, `[rows, columns, 4]`, and its depth along the view, `[rows, columns]`,
    each weighted by coverage; its samples, as `samples`, averaged over each pixel on the GPU."""
    return solids.layer([], view, inks=[], translucent=[],
                        meshes=[(samples(points, faces, values, normals, view, face_values, colour, gloss), 1.0)])


def circles(rims: Point, pivots: Line, n_points: int) -> Point:
    """Closed circles, as polylines `[..., n_points + 1]`: each point of a rim `[...]` turned a full turn about its
    pivot `[...]`, the line through the circle's centre square to it."""
    turns = (pivots.normalized()[..., None] * (np.linspace(0, 2 * np.pi, n_points + 1) / 2)).exp()   # [..., n + 1] Motor
    return turns >> rims[..., None]


def draw(points: Point, faces: np.ndarray, values: np.ndarray, normals: dict, view: View, background: np.ndarray,
         face_values: np.ndarray, colour: Colour, gloss: float) -> np.ndarray:
    """The mesh as the view sees it on the background, `[rows, columns, 3]`; as `layer`."""
    return solids.draw([], view, background, inks=[], translucent=[],
                       meshes=[(samples(points, faces, values, normals, view, face_values, colour, gloss), 1.0)])
