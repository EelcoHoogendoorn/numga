"""A scalar field on a planar mesh, coloured, banded and contoured on the GPU.

The field and its colours at the vertices are interpolated across each triangle by its hats. The fragment stage fills
bands between levels and draws level lines where the field crosses a level, `level_line` and `level_mark` from the
library, one width in pixels however steep the field: no line is extracted as geometry. The colours may be translucent,
down to none, and each band, line and level lays its ink over them with its own opacity, so that pictures stack.
"""
from __future__ import annotations

from functools import cache
from typing import NamedTuple

import numpy as np
from matplotlib.axes import Axes

from examples.util.wgsl import Shader, meshes, pinned, solids
from examples.util.wgsl.flow import axes_view
from examples.util.wgsl.library import level_lines, samples_per_side

Mesh = meshes.Mesh


class Lines(NamedTuple):
    """Level lines every `spacing` of the field, through the level `through`, in `ink`, colour and opacity, `width`
    pixels wide."""
    spacing: float
    through: float
    ink: np.ndarray
    width: float


class Level(NamedTuple):
    """One level line, where the field takes `value`, in `ink`, colour and opacity, `width` pixels wide."""
    value: float
    ink: np.ndarray
    width: float


class Bands(NamedTuple):
    """The field's range filled in bands `spacing` wide from `start`, each in its colour and opacity `[bands, 4]`, the
    first and last reaching on below and above."""
    start: float
    spacing: float
    colours: np.ndarray


source = """
var<storage, read> points: array<Point>;
var<storage, read> faces: array<vec3<u32>>;
// The field at the vertices, and the colours there.
var<storage, read> values: array<f32>;
var<storage, read> colours: array<vec3<f32>>;
var<uniform> camera: Motor;
var<uniform> half_size: vec2<f32>;
// How much the colours at the vertices cover, under the bands and lines laid over them.
var<uniform> opacity: f32;
// Sets of lines: spacing, the level they pass through, and width in samples; and their inks.
var<storage, read> lines: array<vec4<f32>>;
var<storage, read> line_inks: array<vec4<f32>>;
// Single levels: the field's value there, and width in samples; and their inks.
var<storage, read> levels: array<vec4<f32>>;
var<storage, read> level_inks: array<vec4<f32>>;
// Bands: where they start, their spacing, their first colour and how many there are; and the colours.
var<storage, read> bands: array<vec4<f32>>;
var<storage, read> band_colours: array<vec4<f32>>;

struct Corner {
    @builtin(position) position: vec4<f32>,
    @location(0) value: f32,
    @location(1) colour: vec3<f32>,
}

@vertex
fn place(@builtin(vertex_index) index: u32) -> Corner {
    let face = faces[index / 3];
    let vertex = face[index % 3];
    let seen = camera << points[vertex];
    let framed = (axes & seen) / (mv.w & seen);
    return Corner(vec4<f32>(framed.xy / half_size, 0.5, 1.0), values[vertex], colours[vertex]);
}

@fragment
fn shade(corner: Corner) -> @location(0) vec4<f32> {
    // The field's change across one sample, taken once, where every sample of the triangle runs alike.
    let rate = fwidth(corner.value);
    // Colour and coverage, the colour weighted by its coverage, so that samples average and layers stack as they
    // should: each layer lays its ink over what lies beneath by its own coverage.
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

# Stand-ins for no lines, levels or bands, so that the shader's arrays are never empty: clear ink, and no colour.
no_lines = Lines(1.0, 0.0, np.zeros(4), 0.0)
no_level = Level(0.0, np.zeros(4), 0.0)
no_bands = Bands(0.0, 1.0, np.zeros((1, 4)))


@cache
def shader() -> Shader:
    return Shader(level_lines() + source, solids.ga, dict(Point=solids.Point, Motor=solids.Motor),
                  dict(axes=solids.axes), targets=("rgba8unorm",))


def picture(points: solids.Point, mesh: Mesh, values: np.ndarray, colours: np.ndarray, view: solids.View,
            lines: tuple[Lines, ...] = (), levels: tuple[Level, ...] = (), bands: tuple[Bands, ...] = (),
            opacity: float = 1.0) -> np.ndarray:
    """The field `[vertices]` in its colours `[vertices, 3]`, covering by `opacity`, over the mesh at its points, its
    bands filled and its lines and levels drawn over them, in that order: `[rows, columns, 4]`, colour and coverage,
    transparent where no triangle lies."""
    side = samples_per_side()
    lines, levels, bands = lines or (no_lines,), levels or (no_level,), bands or (no_bands,)
    starts = np.cumsum([0] + [len(band.colours) for band in bands[:-1]])
    image, = shader().draw_averaged(
        view.shape, side, 3 * len(mesh.faces), points=points, faces=mesh.faces,
        values=np.asarray(values, dtype=np.float32), colours=np.broadcast_to(colours, (len(values), 3)),
        camera=view.camera, half_size=view.half, opacity=opacity,
        lines=np.array([[set.spacing, set.through, set.width * side, 0] for set in lines]),
        line_inks=np.array([set.ink for set in lines]),
        levels=np.array([[level.value, level.width * side, 0, 0] for level in levels]),
        level_inks=np.array([level.ink for level in levels]),
        bands=np.array([[band.start, band.spacing, start, len(band.colours)] for band, start in zip(bands, starts)]),
        band_colours=np.concatenate([band.colours for band in bands]))
    # The samples averaged with their colours weighted by coverage; the colour where anything covers, divided back.
    coverage = image[..., 3:]
    return np.concatenate((image[..., :3] / np.where(coverage > 0, coverage, 1), coverage), axis=-1)


def overlay(ax: Axes, points: solids.Point, mesh: Mesh, values: np.ndarray, colours: np.ndarray,
            lines: tuple[Lines, ...] = (), levels: tuple[Level, ...] = (), bands: tuple[Bands, ...] = (),
            opacity: float = 1.0):
    """The field drawn into a panel at its own pixels, as `picture` draws it; the image, for an animation to set its
    data."""
    view = axes_view(ax)
    image = picture(points, mesh, values, colours, view, lines, levels, bands, opacity)
    return pinned(ax, image, (*ax.get_xlim(), *ax.get_ylim()))
