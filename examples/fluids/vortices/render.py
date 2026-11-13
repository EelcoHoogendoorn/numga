"""Vorticity, swirl, trails, and worldline braids around moving planar vortices."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.image import AxesImage

from numga import stack

from examples.fluids.vortices import core
from examples.util.wgsl import Shader, fitted, pinned, show, simplices, solids, supersampled
from examples.util.wgsl.camera import Camera

# Fields: the figures' resolution, and the pixels between two panels side by side.
dpi = 100
gap = 8
# The fields' colours: white at zero, deepening with magnitude without saturating early, red for counterclockwise
# vorticity and blue for clockwise vorticity, orange where swirl exceeds strain and purple where strain dominates.
calm = np.ones(3)
turning = np.array(to_rgb("#b2182b"))
counter = np.array(to_rgb("#2166ac"))
swirling = np.array(to_rgb("#b35806"))
strained = np.array(to_rgb("#542788"))
# The trails: colour, opacity, and width in pixels.
trail_colour = np.array(to_rgb("#0f172a"))
trail_opacity = 0.5
trail_width = 1.1
# Worldlines: one colour per vortex, strand width in pixels, view elevation and azimuth in radians,
# margin around a braid, and square panel size in inches.
strand_colours = np.array([to_rgb(colour) for colour in ("#c84b31", "#2d6a9f", "#4c956c", "#e0a458", "#8e7dbe")])
strand_width = 1.6
elevation, azimuth = np.radians([18.0, 30.0])
margin = 0.1
panel_size = 3.0
background = np.ones(3)


# --- plumbing -------------------------------------------------------------------------
def positions(points: core.Point) -> np.ndarray:
    """Horizontal and vertical coordinates of each point read by pairing with the coordinate lines."""
    return (stack((core.mv.x, core.mv.y)) & points[..., None]).to_array()


def extent(points: core.Point) -> tuple[float, float, float, float]:
    """Bounding box of the grid cells centred on the sample points."""
    half_cell = (points[1, 1] - points[0, 0]) / 2                              # [] Point-
    low, high = positions(points[0, 0] - half_cell), positions(points[-1, -1] + half_cell)
    return low[0], high[0], low[1], high[1]


def colour_image(
    values: np.ndarray,
    scale: float,
    positive: np.ndarray,
    negative: np.ndarray,
    n_samples_per_side: int,
) -> np.ndarray:
    """Supersampled RGB image shading signed grid values from white toward `positive` or `negative`."""
    depth = 1 - np.exp(-np.abs(values) / scale)
    samples = calm + depth[..., None] * (np.where(values[..., None] > 0, positive, negative) - calm)
    n_rows, n_columns = values.shape[0] // n_samples_per_side, values.shape[1] // n_samples_per_side
    blocks = samples.reshape(n_rows, n_samples_per_side, n_columns, n_samples_per_side, 3)
    return blocks.mean(axis=(1, 3))[::-1]


def field(
    ax: Axes,
    points: core.Point,
    values: np.ndarray,
    scale: float,
    positive: np.ndarray,
    negative: np.ndarray,
    n_samples_per_side: int,
) -> AxesImage:
    """Axes image displaying a supersampled scalar field pixel-for-pixel on the sample grid."""
    return pinned(ax, colour_image(values, scale, positive, negative, n_samples_per_side), extent(points))


def side_by_side(points: core.Point, n_samples_per_side: int) -> plt.Figure:
    """Two-panel figure sized pixel-for-pixel to the downsampled grid."""
    n_rows, n_columns = points.shape[0] // n_samples_per_side, points.shape[1] // n_samples_per_side
    result = plt.figure(figsize=((2 * n_columns + gap) / dpi, n_rows / dpi), dpi=dpi)
    result.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    return result


def vorticity(derivative: core.Even) -> np.ndarray:
    """Signed vorticity read from the bivector part of the flow derivative."""
    return (core.mv.xy.dual() & derivative).to_array()


def panels(
    figure: plt.Figure,
    points: core.Point,
    derivative: core.Even,
    swirl: core.Scalar,
    vorticity_scale: float,
    swirl_scale: float,
    n_samples_per_side: int,
) -> list[AxesImage]:
    """Vorticity and swirl panels drawn side by side on `figure`."""
    left, right = figure.subplots(1, 2)
    return [
        field(left, points, vorticity(derivative), vorticity_scale, turning, counter, n_samples_per_side),
        field(right, points, swirl.to_array(), swirl_scale, swirling, strained, n_samples_per_side),
    ]


def figure(
    points: core.Point,
    derivative: core.Even,
    swirl: core.Scalar,
    vorticity_scale: float,
    swirl_scale: float,
    n_samples_per_side: int,
) -> plt.Figure:
    """Side-by-side vorticity and swirl figure over the sample grid."""
    result = side_by_side(points, n_samples_per_side)
    panels(result, points, derivative, swirl, vorticity_scale, swirl_scale, n_samples_per_side)
    return result


def braids(worldlines: list[core.Point], height: float) -> plt.Figure:
    """Side-by-side 3D worldline braids in the comoving frame with time running upward."""
    result = plt.figure(figsize=(panel_size * len(worldlines), panel_size))
    for ax, centres in zip(result.subplots(1, len(worldlines)), worldlines):
        n_times, n_vortices = centres.shape
        comoving = (centres - centres.mean(axis=-1)[..., None]).reshape(-1)     # [n_times * n_vortices] Point-
        # The braid width is the larger of its spans across x and y.
        axes = stack((core.mv.x, core.mv.y))                                    # [2] Vector+
        heights = axes[:, None] & comoving                                      # [2, n_times * n_vortices] Scalar+
        lowest, highest = comoving[heights.argmin(axis=-1)], comoving[heights.argmax(axis=-1)]   # [2] Point-, Point-
        spans = (axes & highest) - (axes & lowest)                              # [2] Scalar+
        width = spans.max()                                                     # [] Scalar+
        # Lay the fluid plane on z = 0 of 3D space with time running upward along z.
        times = solids.context.include_from_algebra(width) * np.linspace(0, height, n_times)[:, None]   # [n_times, 1] Scalar+
        plane_offsets = solids.context.include_from_algebra(comoving.reshape(centres.shape))            # [n_times, n_vortices] Point-
        strand_points = solids.origin + plane_offsets + solids.mv.z.dual() * times                      # [n_times, n_vortices] Point-
        view = fitted(solids.view([strand_points], 0.0, elevation, azimuth, margin, 1), ax)
        indices = np.arange(n_times * n_vortices).reshape(n_times, n_vortices)
        strand_topology = simplices.Topology(indices.size, np.stack([indices[:-1], indices[1:]], axis=-1).reshape(-1, 2))
        tubes = simplices.tubes(
            strand_points.reshape(-1),
            strand_topology,
            np.tile(strand_colours[:n_vortices], (n_times - 1, 1)),
            strand_width,
            view,
            gloss=solids.gloss,
        )
        show(ax, solids.draw([], view, background, inks=[], translucent=[], meshes=[(tubes, 1.0)]), view)
    return result


# --- vorticity on the GPU ---------------------------------------------------------------
# Each pixel sample sums the closed-form Rosenhead-Moore vorticity `circulation * core_radius**2 / pi / blob**2`
# over the vortices and their periodic copies.
source = '''
alias Sample = vec3<f32>;
var<storage, read> centres: array<Point>;
var<storage, read> circulations: array<Circulation>;
var<storage, read> copies: array<Point>;
var<uniform> core_squared: f32;
var<uniform> middle: Point;
var<uniform> half_height: f32;
var<uniform> scale: f32;

fn sample(place: vec2<f32>) -> Sample {
    let point = middle + (directions * screen(place)).sum() * half_height;
    var vorticity = circulations[0] * 0.0;
    for (var vortex = 0u; vortex < arrayLength(&centres); vortex++) {
        for (var copy = 0u; copy < arrayLength(&copies); copy++) {
            let separation = point - (centres[vortex] + copies[copy]);
            let blob = separation.dot(separation) + core_squared;
            vorticity = vorticity + circulations[vortex] * (core_squared / pi / (blob * blob));
        }
    }
    let size = plane_unit & vorticity;
    return mix(calm, select(counter, turning, size > 0), 1 - exp(-abs(size) / scale));
}
'''


@cache
def shader() -> Shader:
    """Compiled WGSL shader evaluating closed-form blob vorticity over a planar window."""
    mv = core.mv
    return Shader(
        supersampled() + source,
        core.ga,
        dict(Point=core.Point, Circulation=core.Circulation),
        dict(
            directions=stack((mv.x, mv.y)).dual(),
            plane_unit=mv.xy.dual(),
            pi=np.pi,
            calm=calm,
            turning=turning,
            counter=counter,
        ),
    )


def vorticity_image(
    centres: core.Point,
    circulations: core.Circulation,
    core_radius: float,
    copies: core.Point,
    middle: core.Point,
    half_height: float,
    shape: tuple[int, int],
    scale: float,
) -> np.ndarray:
    """Floating-point RGB image of the vorticity field on a window centred on `middle`."""
    return shader()(
        shape,
        centres=centres,
        circulations=circulations,
        copies=copies,
        core_squared=core_radius**2,
        middle=middle,
        half_height=half_height,
        scale=scale,
    )["image"]


def byte_image(image: np.ndarray) -> np.ndarray:
    """Unsigned 8-bit RGB image clipped to unit range."""
    return np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)


def animate(
    circulations: core.Circulation,
    core_radius: float,
    copies: core.Point,
    half_height: float,
    shape: tuple[int, int],
    scale: float,
    frames: Iterable[core.Point],
) -> Iterator[np.ndarray]:
    """Yield comoving vorticity frames centred on the mean vortex position."""
    for centres in frames:
        yield byte_image(
            vorticity_image(centres, circulations, core_radius, copies, centres.mean(axis=-1), half_height, shape, scale)
        )


def window(
    circulations: core.Circulation,
    core_radius: float,
    copies: core.Point,
    middle: core.Point,
    half_height: float,
    shape: tuple[int, int],
    scale: float,
    frames: Iterable[core.Point],
) -> Iterator[np.ndarray]:
    """Yield fixed-window vorticity frames overlaid with each vortex's path trail."""
    n_rows, n_columns = shape
    x_line, y_line = core.mv.x, core.mv.y
    middle_readings = np.array([(x_line & middle).to_array(), (y_line & middle).to_array()])
    camera = Camera(
        x_line,
        y_line,
        0 * x_line,
        0 * x_line,
        1.0,
        middle_readings,
        np.array([half_height * n_columns / n_rows, half_height]),
        n_columns,
    )
    history: list[core.Point] = []
    for centres in frames:
        history.append(centres)
        n_points = len(history) + 1
        n_vortices = centres.shape[0]
        paths = stack([history[0]] + history, axis=1)                           # [n_vortices, n_points] Point-
        trails = simplices.composited([
            simplices.strokes(
                paths.reshape(-1),
                simplices.Topology.chains([n_points] * n_vortices),
                trail_colour,
                trail_width,
                camera,
                opacity=trail_opacity,
            )
        ])
        image = vorticity_image(centres, circulations, core_radius, copies, middle, half_height, shape, scale)
        yield byte_image(image * (1 - trails[..., 3:]) + trails[..., :3] * trails[..., 3:])
