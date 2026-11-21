"""Scalar fields over planar triangulated complexes are shaded, banded, and contoured directly on the GPU in the
vertices' own algebra. Vertex values and vertex colours interpolate linearly across each triangle by its hat functions,
while fragment-stage screen derivatives (`fwidth`) evaluate constant-pixel-width level curves (`level_line` and
`level_mark`) regardless of field gradient steepness without extracting contour polylines on the host.

Vertex colours are scaled by `opacity` down to zero, and `Bands`, `Lines`, and `Level` specifications composite
premultiplied ink over them in order so multiple contour and shading passes stack cleanly in `composited`. Vertex
positions pair directly against a `Camera` of `points.algebra`, compiling one cached shader per `(Point, Plane)` pair
without converting points into another algebra.
"""
from __future__ import annotations

from functools import cache
from typing import NamedTuple

import numpy as np

from numga.extensor import Extensor
from numga.gatype import GAType
from numga.topology import Topology
from numga.wgsl import Shader, level_lines, samples_per_side

from .camera import Camera, check_projection, placing, reading
from .drawing import Draw


class Lines(NamedTuple):
    """Periodic level curves repeat every `spacing` units of the scalar field passing through `through`, drawn `width`
    pixels wide in RGBA `ink`."""
    spacing: float
    through: float
    ink: np.ndarray
    width: float


class Level(NamedTuple):
    """A single level curve follows `field == value`, drawn `width` pixels wide in RGBA `ink`."""
    value: float
    ink: np.ndarray
    width: float


class Bands(NamedTuple):
    """Piecewise-constant intervals of width `spacing` starting at `start` fill with RGBA `colours` (`[bands, 4]`),
    clamping values below the first band and above the last."""
    start: float
    spacing: float
    colours: np.ndarray


source = """
var<storage, read> points: array<Point>;
var<storage, read> faces: array<vec3<u32>>;
var<storage, read> values: array<f32>;
var<storage, read> colours: array<vec3<f32>>;
var<uniform> opacity: f32;
// Periodic line sets pack (spacing, through, sample_width, 0) beside RGBA line_inks.
var<storage, read> lines: array<vec4<f32>>;
var<storage, read> line_inks: array<vec4<f32>>;
// Single levels pack (value, sample_width, 0, 0) beside RGBA level_inks.
var<storage, read> levels: array<vec4<f32>>;
var<storage, read> level_inks: array<vec4<f32>>;
// Band specs pack (start, spacing, colour_offset, n_bands) indexing into RGBA band_colours.
var<storage, read> bands: array<vec4<f32>>;
var<storage, read> band_colours: array<vec4<f32>>;

struct Corner {
    @builtin(position) position: vec4<f32>,
    value: f32,
    colour: vec3<f32>,
}

@vertex
fn place(@builtin(vertex_index) index: u32) -> Corner {
    let face = faces[index / 3];
    let vertex = face[index % 3];
    return Corner(clip(points[vertex]), values[vertex], colours[vertex]);
}

@fragment
fn shade(corner: Corner) -> @location(0) vec4<f32> {
    // Evaluate fwidth once in uniform control flow across the triangle's samples.
    let rate = fwidth(corner.value);
    // Accumulate premultiplied RGBA so sample averaging and alpha compositing commute.
    var picture = vec4<f32>(corner.colour * opacity, opacity);
    for (var index = 0u; index < arrayLength(&bands); index++) {
        let band = bands[index];
        let step = clamp(floor((corner.value - band.x) / band.y), 0.0, band.w - 1.0);
        let filling = band_colours[u32(band.z + step)];
        picture = vec4<f32>(filling.rgb * filling.a, filling.a) + picture * (1.0 - filling.a);
    }
    for (var index = 0u; index < arrayLength(&lines); index++) {
        let set = lines[index];
        let ink = line_inks[index];
        let cover = ink.a * level_line((corner.value - set.y) / set.x, rate / set.x, set.z);
        picture = vec4<f32>(ink.rgb * cover, cover) + picture * (1.0 - cover);
    }
    for (var index = 0u; index < arrayLength(&levels); index++) {
        let level = levels[index];
        let ink = level_inks[index];
        let cover = ink.a * level_mark(corner.value - level.x, rate, level.y);
        picture = vec4<f32>(ink.rgb * cover, cover) + picture * (1.0 - cover);
    }
    return picture;
}
"""

# Zero-alpha sentinels keep WGSL storage buffers non-empty when bands, lines, or levels are omitted.
no_lines = Lines(1.0, 0.0, np.zeros(4), 0.0)
no_level = Level(0.0, np.zeros(4), 0.0)
no_bands = Bands(0.0, 1.0, np.zeros((1, 4)))


@cache
def shader(Point: GAType, Plane: GAType) -> Shader:
    """Compile and cache the contour and band shader for vertex type `Point` and camera plane type `Plane`."""
    return Shader(level_lines() + placing(Plane) + source, Point.algebra, dict(Point=Point, Plane=Plane), {})


def picture(points: Extensor, topology: Topology, values: np.ndarray, colours: np.ndarray, camera: Camera,
            lines: tuple[Lines, ...] = (), levels: tuple[Level, ...] = (), bands: tuple[Bands, ...] = (),
            opacity: float = 1.0) -> Draw | np.ndarray:
    """Vertex `values` (`[V]`) and `colours` (`[V, 3]`) over `topology`'s triangles produce a `Draw` pass (or a
    transparent `[rows, columns, 4]` array when `topology` has no triangles) with `bands`, `lines`, and `levels`
    composited over the base shading."""
    check_projection(points, camera)
    side = samples_per_side()
    if topology.n_elements[2] == 0:
        return np.zeros(camera.shape + (4,))
    lines, levels, bands = lines or (no_lines,), levels or (no_level,), bands or (no_bands,)
    starts = np.cumsum([0] + [len(band.colours) for band in bands[:-1]])
    return Draw(shader(**reading(points, camera)), camera.shape, 3 * topology.n_elements[2], dict(
        points=points, faces=topology.elements[2],
        values=np.asarray(values, dtype=np.float32), colours=np.broadcast_to(colours, (len(values), 3)),
        opacity=opacity, **camera.uniforms(1.0),
        lines=np.array([[set.spacing, set.through, set.width * side, 0] for set in lines]),
        line_inks=np.array([set.ink for set in lines]),
        levels=np.array([[level.value, level.width * side, 0, 0] for level in levels]),
        level_inks=np.array([level.ink for level in levels]),
        bands=np.array([[band.start, band.spacing, start, len(band.colours)] for band, start in zip(bands, starts)]),
        band_colours=np.concatenate([band.colours for band in bands])))

