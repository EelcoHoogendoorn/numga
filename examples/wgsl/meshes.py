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
from examples.wgsl.solids import Motor, Plane, Point, ambient, ga, gloss, lamp, mv, origin, shininess

# The intrinsic unit: the volume four points span, from the origin along each axis.
unit = origin & mv.point([1, 0, 0]) & mv.point([0, 1, 0]) & mv.point([0, 0, 1])

source = """
var<storage, read> points: array<Point>;
var<storage, read> faces: array<vec3<u32>>;
var<storage, read> colours: array<vec3<f32>>;
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
var<uniform> gloss: f32;
var<uniform> shininess: f32;

struct Corner {
    @builtin(position) position: vec4<f32>,
    @location(0) colour: vec3<f32>,
    @location(1) normal: vec3<f32>,
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
    // The normal crosses to the fragments as the numbers that interpolate: its inner products with the axes' planes.
    return Corner(position, colours[vertex], vec3<f32>(normal | mv.x, normal | mv.y, normal | mv.z));
}

// A pixel of a face: its colour, lit by the lamp through its normal blended across the face, with a glint.
@fragment
fn shade(corner: Corner) -> @location(0) vec4<f32> {
    let blended = mv.x * corner.normal.x + mv.y * corner.normal.y + mv.z * corner.normal.z;
    let direction = camera >> -mv.z.dual();
    // Facing the viewer: a surface is lit on the side the camera sees.
    let facing = blended * select(1.0, -1.0, (blended & direction) > 0);
    let normal = facing / (facing | facing).square_root();
    let light = max(normal & lamp, 0.0);
    let glint = pow(max(-((normal >> direction).dual() & lamp), 0.0), shininess) * select(0.0, 1.0, light > 0);
    return vec4<f32>(corner.colour * (ambient + (1 - ambient) * light) + gloss * glint, 1.0);
}
"""


# --- plumbing -------------------------------------------------------------------------
@cache
def shader() -> Shader:
    return Shader(source, ga, dict(Point=Point, Plane=Plane, Motor=Motor))


def flat(points: Point, faces: np.ndarray) -> dict:
    """Each face's own normal at all three corners."""
    normal = face_normals(points, faces)                                      # [faces] Plane
    return dict(a=normal, b=normal, c=normal)


def smooth(points: Point, faces: np.ndarray) -> dict:
    """At each corner, the unit normals of the faces around its vertex, summed and made unit."""
    # Summed at each vertex: a scatter of the faces' coefficients, the one step outside the algebra.
    summed = np.zeros((len(points), 4))
    np.add.at(summed, faces.reshape(-1), np.repeat(np.asarray(face_normals(points, faces).kernel), 3, axis=0))
    vertex = mv.vector(summed)                                                # [points] Plane
    corners = (vertex / (vertex | vertex).square_root())[faces]               # [faces, 3] Plane
    return dict(a=corners[:, 0], b=corners[:, 1], c=corners[:, 2])


def face_normals(points: Point, faces: np.ndarray) -> Plane:
    """Each face's unit normal, as the plane through the origin square to it `[faces]`: the join of its
    corners, intrinsic as they are, made extrinsic by the intrinsic unit, and moved to the origin."""
    corners = points[faces]
    joined = (corners[:, 0] & corners[:, 1] & corners[:, 2]) * unit           # [faces] Plane
    through_origin = joined - mv.w * (joined & origin)
    return through_origin / (through_origin | through_origin).square_root()


def draw(points: Point, faces: np.ndarray, colours: np.ndarray, normals: dict, camera: Motor, half: np.ndarray,
         elevation: float, azimuth: float, pixels: int, background: np.ndarray) -> np.ndarray:
    """The mesh as the camera sees it, `[rows, columns, 3]`, `pixels` across, on the background."""
    shape = (int(round(pixels * half[1] / half[0])), pixels)
    drawn = shader().draw(shape, 3 * len(faces), points=points, faces=faces, colours=np.broadcast_to(colours, (len(points), 3)),
                          normals=normals, camera=camera, half_size=half, depth=8 * half.max(), lamp=lamp(elevation, azimuth),
                          ambient=ambient, gloss=gloss, shininess=shininess)
    # The colour is premultiplied by each pixel's coverage; the background shows through the rest.
    shown = drawn[..., :3] + (1 - drawn[..., 3:]) * background
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)
