"""Drawing for developability: flat-shaded surfaces, each face lit by its own normal, so that the
patches read as flat planes and the creases between them as sharp changes of shade.

The surface is rasterized on the GPU, by a shader whose geometry is written in numga: each corner
reads its face's three vertices, lights the face by the normal its edges meet in, `(b - a) & (c - a)`,
and places itself on the image by its pairings with the camera's own planes.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import cache

import numpy as np

from examples.surfaces.developability import core
from examples.util.wgsl import Shader
from examples.util.wgsl.library import samples_per_side

# The surface's colour; the direction toward the light, up and off to one side, a sum of the directions along the
# axes, and the share of light that falls everywhere; the camera's elevation and azimuth, its distance and half its
# view; the frames' size and background.
surface_colour = np.array([0.90, 0.89, 0.86])
light_direction = core.mv.z.dual() * 0.6 - core.mv.x.dual() * 0.4 - core.mv.y.dual() * 0.7      # [] Point-
ambient = 0.25
elevation, azimuth = np.radians(24.0), np.radians(54.0)
distance, half_view = 8.0, 0.17
frame_shape = (480, 480)
background = np.array([1.0, 1.0, 1.0])

source = """
var<storage, read> points: array<Point>;
var<storage, read> faces: array<vec3<u32>>;
var<uniform> turn: Rotor;
var<uniform> eye: Point;
var<uniform> lamp: Point;
var<uniform> half_view: f32;
var<uniform> pixels: vec2<u32>;
var<uniform> colour: vec3<f32>;
var<uniform> ambient: f32;

// The depths the image holds, from the eye along the view.
const near = 0.1;
const far = 100.0;

// A corner of a face on the image, and its face's colour, the same at every corner.
struct Corner {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) colour: vec3<f32>,
}

@vertex
fn place(@builtin(vertex_index) index: u32) -> Corner {
    let face = faces[index / 3];
    let a = points[face.x];
    let b = points[face.y];
    let c = points[face.z];
    // Two edges meet in the face's normal, twice its area long, and the normal pairs with the lamp's direction.
    let normal = ((b - a) & (c - a)).normalized();
    let lit = ambient + (1 - ambient) * max(normal & lamp, 0.0);
    // The corner seen from the eye, in the camera's own frame, read by the camera's planes.
    let seen = turn << (points[face[index % 3]] - eye);
    let depth = -(mv.z & seen);
    let aspect = f32(pixels.x) / f32(pixels.y);
    let position = vec4<f32>((mv.x & seen) / (half_view * aspect), (mv.y & seen) / half_view,
                             (depth - near) * far / (far - near), depth);
    return Corner(position, colour * lit);
}

@fragment
fn shade(corner: Corner) -> @location(0) vec4<f32> {
    return vec4<f32>(corner.colour, 1.0);
}
"""


# --- plumbing -------------------------------------------------------------------------
@cache
def shader() -> Shader:
    return Shader(source, core.ga, dict(Point=core.Point, Rotor=core.ga.gatype.rotor()), {}, targets=("rgba32float",))


def coordinates(points: core.Point) -> np.ndarray:
    """The x, y and z coordinates of points, antivectors."""
    return points.cast(core.ga.subspace("yz zx xy")).kernel


def camera() -> tuple[core.Extensor, core.Point]:
    """The camera's turn, from one looking along its own -z with y up, and its eye."""
    pitch = (core.mv.yz * (-(np.pi / 2 - elevation) / 2)).exp()
    turn = (core.mv.xy * (-azimuth / 2)).exp() * pitch
    return turn, (turn >> core.mv.z.dual()) * distance


def draw(mesh: core.Mesh, vertices: core.Point) -> np.ndarray:
    """One surface on the background, every face shaded flat by the light falling on it, `[rows, columns, 3]`."""
    turn, eye = camera()
    # Drawn at every sample of each pixel, as many to a side as the traced pictures take, and averaged on the GPU: the
    # colour premultiplied by each pixel's coverage, the background showing through the rest.
    drawn, = shader().draw_averaged(frame_shape, samples_per_side(), 3 * len(mesh.faces),
                                    points=vertices.batch(), faces=mesh.faces, turn=turn, eye=eye,
                                    lamp=light_direction.normalized(), half_view=half_view, colour=surface_colour,
                                    ambient=ambient)
    shown = drawn[..., :3] + (1 - drawn[..., 3:]) * background
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)


def animate_flow(mesh: core.Mesh, states: Iterable[tuple[core.Point, core.Scalar]]) -> Iterator[np.ndarray]:
    """Frames of a surface flowing towards developable patches."""
    for vertices, _ in states:
        yield draw(mesh, vertices)
