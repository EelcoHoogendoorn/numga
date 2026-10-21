"""Triangle meshes in PGA3D, rasterized on the GPU: lit by the lamp of `solids`, seen by its camera along
parallel rays, on a plain background.

Each corner of a face reads its point, its colour and the face's shading normal there, and places itself on
the image by its pairings with the camera's own planes; the normals are flat, the face's own, or smooth, the
faces around each vertex averaged, as the caller builds them.
"""

from __future__ import annotations

from functools import cache

import numpy as np

from examples.wgsl import Shader
from examples.wgsl.library import compose
from examples.wgsl.solids import Motor, Plane, Point, ambient, fill_tilt, ga, gloss, lamp, mv, origin, shininess

# The intrinsic unit: the volume four points span, from the origin along each axis.
unit = origin & mv.point([1, 0, 0]) & mv.point([0, 1, 0]) & mv.point([0, 0, 1])

def source(colour: str) -> str:
    """WGSL drawing a mesh, coloured by `colour`, WGSL defining `fn colour(values: vec4<f32>, face_values:
    vec4<f32>, corner: vec3<f32>, front: bool) -> vec3<f32>`: from each pixel's values, the corners' blended
    across the face, the face's own, how near the pixel lies to each corner, and whether the camera sees the side
    the face's normal points to. It writes the colour and the depth, each weighted by how much of a pixel the
    mesh covers."""
    return colour + """
var<storage, read> points: array<Point>;
var<storage, read> faces: array<vec3<u32>>;
// Values at each corner of each face, three to a face, and each face's own.
var<storage, read> values: array<vec4<f32>>;
var<storage, read> face_values: array<vec4<f32>>;
// Each face's shading normal at its three corners: planes through the origin, square to it.
struct Corners {
    a: Plane,
    b: Plane,
    c: Plane,
}
var<storage, read> normals: array<Corners>;
// The camera of `solids`: looking along its own -z, its rays parallel, the image half as wide and high as these.
var<uniform> camera: Motor;
var<uniform> half_size: vec2<f32>;
var<uniform> depth: f32;
var<uniform> lamp: Point;
var<uniform> ambient: f32;
var<uniform> fill_tilt: f32;
var<uniform> gloss: f32;
var<uniform> shininess: f32;

// Interpolated at every sample of a pixel, so its colour is supersampled along with its edges.
struct Corner {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(perspective, sample) values: vec4<f32>,
    @location(1) @interpolate(flat) face_values: vec4<f32>,
    @location(2) @interpolate(perspective, sample) normal: vec3<f32>,
    @location(3) @interpolate(perspective, sample) corner: vec3<f32>,
    @location(4) @interpolate(perspective, sample) depth: f32,
}

// A pixel's colour, and its depth along the view in the depth target's first channel.
struct Shade {
    @location(0) colour: vec4<f32>,
    @location(1) depth: vec4<f32>,
}

// A corner on the image: the point in the camera's frame, read by its planes.
@vertex
fn place(@builtin(vertex_index) index: u32) -> Corner {
    let face = faces[index / 3];
    let vertex = face[index % 3];
    let seen = camera << points[vertex];
    let weight = mv.w & seen;
    let position = vec4<f32>((mv.x & seen) / weight / half_size.x, (mv.y & seen) / weight / half_size.y,
                             0.5 - (mv.z & seen) / weight / depth, 1.0);
    let corners = normals[index / 3];
    let normal = select(select(corners.c, corners.b, index % 3 == 1), corners.a, index % 3 == 0);
    let corner = vec3<f32>(f32(index % 3 == 0), f32(index % 3 == 1), f32(index % 3 == 2));
    // The normal crosses to the fragments as the numbers that interpolate: its inner products with the axes' planes.
    return Corner(position, values[index], face_values[index / 3], vec3<f32>(normal | mv.x, normal | mv.y, normal | mv.z),
                  corner, -(mv.z & seen) / weight);
}

// A pixel of a face: its colour, lit by the lamp through its normal blended across the face, with a glint.
@fragment
fn shade(corner: Corner) -> Shade {
    let blended = mv.x * corner.normal.x + mv.y * corner.normal.y + mv.z * corner.normal.z;
    let direction = camera >> -mv.z.dual();
    // Facing the viewer: a surface is lit on the side the camera sees.
    let front = (blended & direction) <= 0;
    let facing = blended * select(-1.0, 1.0, front);
    let normal = facing / (facing | facing).square_root();
    let light = max(normal & lamp, 0.0);
    let glint = pow(max(-((normal >> direction).dual() & lamp), 0.0), shininess) * select(0.0, 1.0, light > 0);
    let fill = ambient * (1 + fill_tilt * (normal | mv.z));
    let lit = colour(corner.values, corner.face_values, corner.corner, front) * (fill + (1 - ambient) * light) + gloss * glint;
    return Shade(vec4<f32>(lit, 1.0), vec4<f32>(corner.depth, 0.0, 0.0, 1.0));
}
"""


# The vertices' own colours, blended across each face.
plain = """
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec3<f32> {
    return values.rgb;
}
"""


# Each face's own colour.
by_face = """
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec3<f32> {
    return face_values.rgb;
}
"""


# --- plumbing -------------------------------------------------------------------------
@cache
def shader(colour: str) -> Shader:
    return Shader(source(colour), ga, dict(Point=Point, Plane=Plane, Motor=Motor))


def flat(points: Point, faces: np.ndarray) -> dict:
    """Each face's own normal at all three corners."""
    normal = face_normals(points, faces)                                      # [faces] Plane
    return dict(a=normal, b=normal, c=normal)


def smooth(points: Point, faces: np.ndarray) -> dict:
    """At each corner, the unit normals of the faces around its vertex, summed and made unit."""
    # Summed at each vertex: a scatter of the faces' coefficients, the one step outside the algebra.
    summed = np.zeros((len(points), 4))
    np.add.at(summed, faces.reshape(-1), np.repeat(np.asarray(face_normals(points, faces).kernel), 3, axis=0))
    corners = mv.vector(summed)[faces]                                        # [faces, 3] Plane
    corners = corners / (corners | corners).square_root()
    return dict(a=corners[:, 0], b=corners[:, 1], c=corners[:, 2])


def face_normals(points: Point, faces: np.ndarray) -> Plane:
    """Each face's unit normal, as the plane through the origin square to it `[faces]`: the join of its
    corners, intrinsic as they are, made extrinsic by the intrinsic unit, and moved to the origin."""
    corners = points[faces]
    joined = (corners[:, 0] & corners[:, 1] & corners[:, 2]) * unit           # [faces] Plane
    through_origin = joined - mv.w * (joined & origin)
    return through_origin / (through_origin | through_origin).square_root()


def frames_along(points: np.ndarray) -> np.ndarray:
    """A normal at every vertex of polylines `[..., n, 3]` that turns as little as the curve allows: the previous
    vertex's normal reflected twice, in the plane between the vertices and in the plane between their tangents."""
    tangents = np.gradient(points, axis=-2)
    tangents /= np.linalg.norm(tangents, axis=-1, keepdims=True)
    # Any normal at the first vertex: its tangent crossed with the axis it leans on least.
    least = np.eye(3)[np.argmin(np.abs(tangents[..., 0, :]), axis=-1)]
    normals = [np.cross(tangents[..., 0, :], least)]
    for index in range(1, points.shape[-2]):
        step = points[..., index, :] - points[..., index - 1, :]
        reflected = normals[-1] - step * (2 * np.sum(step * normals[-1], axis=-1, keepdims=True)
                                          / np.sum(step * step, axis=-1, keepdims=True))
        turned = tangents[..., index - 1, :] - step * (2 * np.sum(step * tangents[..., index - 1, :], axis=-1, keepdims=True)
                                                        / np.sum(step * step, axis=-1, keepdims=True))
        between = tangents[..., index, :] - turned
        normals.append(reflected - between * (2 * np.sum(between * reflected, axis=-1, keepdims=True)
                                              / np.maximum(np.sum(between * between, axis=-1, keepdims=True), 1e-30)))
    normals = np.stack(normals, axis=-2)
    return normals / np.linalg.norm(normals, axis=-1, keepdims=True)


def tube(points: np.ndarray, radius: float, n_around: int) -> tuple[Point, np.ndarray]:
    """A tube of a radius about polylines `[curves, n, 3]`: rings of points about each vertex, each its normal turned
    about the line along the curve there, flattened from `[curves, n, around]`, and its triangles, indices into
    them."""
    centres = mv.point(points)                                                 # [curves, n] Point
    rims = mv.point(points + radius * frames_along(points))                    # [curves, n] Point
    # The line along the curve at each vertex: the join of two of its points, made extrinsic by the intrinsic unit.
    axes = (centres & mv.point(points + np.gradient(points, axis=-2))) * unit     # [curves, n] Line
    axes = axes / axes.norm()
    turns = (axes[..., None] * (np.linspace(0, 2 * np.pi, n_around, endpoint=False) / 2)).exp()   # [curves, n, around] Motor
    rings = turns >> rims[..., None]                                           # [curves, n, around] Point
    # Closed around: each ring's first vertex again after its last.
    index = np.arange(np.prod(rings.shape)).reshape(rings.shape)
    index = np.concatenate([index, index[..., :1]], axis=-1)                   # [curves, n, around + 1]
    corners = np.stack([index[:, :-1, :-1], index[:, 1:, :-1], index[:, 1:, 1:], index[:, :-1, 1:]], axis=-1)
    return rings.reshape(-1), corners[..., [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3)


def layer(points: Point, faces: np.ndarray, values: np.ndarray, normals: dict, camera: Motor, half: np.ndarray,
          elevation: float, azimuth: float, pixels: int, face_values: np.ndarray = np.zeros(4),
          colour: str = plain, gloss: float = gloss) -> tuple[np.ndarray, np.ndarray]:
    """The mesh as the camera sees it, `pixels` across: its colour, `[rows, columns, 4]`, and its depth along the
    view, `[rows, columns]`, each weighted by coverage; from up to four values at each corner of each face,
    `[faces, 3, values]` or one row for all, and at each face, coloured by `colour`, with a glint of `gloss`."""
    def padded(entries: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        entries = np.broadcast_to(entries, shape + (np.shape(entries)[-1],))
        return np.concatenate([entries, np.zeros(shape + (4 - entries.shape[-1],))], axis=-1)
    shape = (int(round(pixels * half[1] / half[0])), pixels)
    colours, depths = shader(colour).draw(shape, 3 * len(faces), points=points, faces=faces,
                                          values=padded(values, (len(faces), 3)).reshape(-1, 4),
                                          face_values=padded(face_values, (len(faces),)), normals=normals, camera=camera,
                                          half_size=half, depth=8 * half.max(), lamp=lamp(elevation, azimuth),
                                          ambient=ambient, fill_tilt=fill_tilt, gloss=gloss, shininess=shininess)
    return colours, depths[..., 0]


def draw(points: Point, faces: np.ndarray, values: np.ndarray, normals: dict, camera: Motor, half: np.ndarray,
         elevation: float, azimuth: float, pixels: int, background: np.ndarray, face_values: np.ndarray = np.zeros(4),
         colour: str = plain, gloss: float = gloss) -> np.ndarray:
    """The mesh as the camera sees it on the background, `[rows, columns, 3]`; as `layer`."""
    return compose([(*layer(points, faces, values, normals, camera, half, elevation, azimuth, pixels, face_values, colour,
                            gloss), 1.0)], background)
