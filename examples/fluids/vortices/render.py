"""Vorticity and swirl around moving vortices."""

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
from examples.util.wgsl import Shader, fitted, pinned, planar, show, solids, strokes, supersampled

# Fields: the figures' resolution, and the pixels between two panels side by side.
dpi = 100
gap = 8
# The fields' colours: white at zero, deepening with the field's size, without end, so that where the vortices
# overlap their fields add up deeper; the vorticity red where the fluid turns counterclockwise and blue where it
# turns clockwise, the swirl orange where the fluid turns faster than it stretches and purple where it stretches.
calm = np.ones(3)
turning = np.array(to_rgb("#b2182b"))
counter = np.array(to_rgb("#2166ac"))
swirling = np.array(to_rgb("#b35806"))
strained = np.array(to_rgb("#542788"))
# The trails: their colour, opacity and width in pixels.
trail_colour = np.array(to_rgb("#0f172a"))
trail_opacity = 0.5
trail_width = 1.1
# Worldlines: one colour per vortex, the strands' width in pixels, the view as elevation and azimuth in radians,
# azimuth zero looking along +y, the margin around a braid as a share of its larger side, and the size in inches of
# each braid's square panel.
strand_colours = np.array([to_rgb(colour) for colour in ("#c84b31", "#2d6a9f", "#4c956c", "#e0a458", "#8e7dbe")])
strand_width = 1.6
elevation, azimuth = np.radians([18.0, 30.0])
margin = 0.1
panel_size = 3.0
background = np.ones(3)


# --- plumbing -------------------------------------------------------------------------
def positions(points: core.Point) -> np.ndarray:
    """x and y of each point, on the last axis: its pairings with the lines x and y."""
    return (stack((core.mv.x, core.mv.y)) & points[..., None]).to_array()


def extent(points: core.Point) -> tuple[float, float, float, float]:
    """The box the grid's cells cover: its samples sit at the cell centres, half a cell inside."""
    coordinates = positions(points)
    half_cell = (coordinates[1, 1] - coordinates[0, 0]) / 2
    low, high = coordinates[0, 0] - half_cell, coordinates[-1, -1] + half_cell
    return low[0], high[0], low[1], high[1]


def coloured(values: np.ndarray, scale: float, positive: np.ndarray, negative: np.ndarray, n_samples_per_side: int
             ) -> np.ndarray:
    """Values on the grid `[rows, columns]`, rows upward as the grid holds them, from white towards the colour of
    their sign, two thirds of the way at `scale`, each pixel the average of its block of samples,
    `n_samples_per_side` along each side: the picture's RGB `[rows, columns, 3]`, rows downward."""
    depth = 1 - np.exp(-np.abs(values) / scale)
    samples = calm + depth[..., None] * (np.where(values[..., None] > 0, positive, negative) - calm)
    rows, columns = values.shape[0] // n_samples_per_side, values.shape[1] // n_samples_per_side
    blocks = samples.reshape(rows, n_samples_per_side, columns, n_samples_per_side, 3)
    return blocks.mean(axis=(1, 3))[::-1]


def field(ax: Axes, points: core.Point, values: np.ndarray, scale: float, positive: np.ndarray, negative: np.ndarray,
          n_samples_per_side: int) -> AxesImage:
    """A field sampled on the grid of points, a block of samples a pixel, held pixel for pixel."""
    return pinned(ax, coloured(values, scale, positive, negative, n_samples_per_side), extent(points))


def side_by_side(points: core.Point, n_samples_per_side: int) -> plt.Figure:
    """A figure with room for two panels of the grid's pixels side by side."""
    rows, columns = points.shape[0] // n_samples_per_side, points.shape[1] // n_samples_per_side
    result = plt.figure(figsize=((2 * columns + gap) / dpi, rows / dpi), dpi=dpi)
    result.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    return result


def vorticity(derivative: core.Even) -> np.ndarray:
    """The derivative's bivector part, the vorticity, read by its pairing with the unit scalar."""
    return (derivative & core.mv.scalar([1])).to_array()


def panels(figure: plt.Figure, points: core.Point, derivative: core.Even, swirl: core.Scalar,
           vorticity_scale: float, swirl_scale: float, n_samples_per_side: int) -> list[AxesImage]:
    left, right = figure.subplots(1, 2)
    return [field(left, points, vorticity(derivative), vorticity_scale, turning, counter, n_samples_per_side),
            field(right, points, swirl.to_array(), swirl_scale, swirling, strained, n_samples_per_side)]


def figure(points: core.Point, derivative: core.Even, swirl: core.Scalar, vorticity_scale: float, swirl_scale: float,
           n_samples_per_side: int) -> plt.Figure:
    result = side_by_side(points, n_samples_per_side)
    panels(result, points, derivative, swirl, vorticity_scale, swirl_scale, n_samples_per_side)
    return result


def braids(worldlines: list[core.Point], height: float) -> plt.Figure:
    """Worldlines `[times, vortices]` side by side, each a braid of strokes through the plane and time, one
    colour per vortex, drawn following the vortices' mean position. Time runs upward and is stretched to
    `height` times the braid's width: it has no scale of its own against distance."""
    result = plt.figure(figsize=(panel_size * len(worldlines), panel_size))
    for ax, centres in zip(result.subplots(1, len(worldlines)), worldlines):
        following = (centres - centres.mean(axis=-1)[..., None]).reshape(-1)    # [times * vortices] Point-
        # The braid's width: the larger of its spans across x and y, between the lines that touch it on either side.
        across = stack((core.mv.x, core.mv.y))                                  # [2] Plane+
        heights = across[:, None] & following                                   # [2, times * vortices] Scalar+
        lowest, highest = following[heights.argmin(axis=-1)], following[heights.argmax(axis=-1)]   # [2] Point-, Point-
        spans = (across & highest) - (across & lowest)
        width = spans.max()
        # The plane laid on z = 0 of space, and time running up, along z.
        times = solids.context.include_from_algebra(width) * np.linspace(0, height, centres.shape[0])[:, None]  # [times, 1] Scalar+
        on_plane = solids.context.include_from_algebra(following.reshape(centres.shape))  # [times, vortices] Point-
        drawn = solids.origin + on_plane + solids.mv.z.dual() * times           # [times, vortices] Point-
        view = fitted(solids.view([drawn], 0.0, elevation, azimuth, margin, 1), ax)
        # Each vortex's strand, its points through time and its colour.
        strands = [(drawn[:, vortex], colour) for vortex, colour in zip(range(drawn.shape[1]), strand_colours)]
        lines = strokes.polylines(strands, strand_width, view)
        show(ax, solids.draw([], view, background, inks=[], translucent=[], meshes=[(lines, 1.0)]), view)
    return result


# --- vorticity on the GPU ---------------------------------------------------------------
# Each sample of a window sums the vorticity of every vortex blob and its copies, `circulation * core_radius**2 / pi /
# blob**2`, the bivector part of the flow's derivative, in closed form.
source = """
alias Sample = vec3<f32>;
// The vortices: their centres, their circulations and the offsets their pattern repeats at; the square of their core's
// radius, the middle of the window, half its height, and the vorticity at which the colours are two thirds deep.
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
            // The squared distance from the vortex, softened by its core.
            let separation = point - (centres[vortex] + copies[copy]);
            let blob = separation.dot(separation) + core_squared;
            vorticity = vorticity + circulations[vortex] * (core_squared / pi / (blob * blob));
        }
    }
    // White at zero, deepening towards the colour of the sense of the turn without end.
    let size = vorticity & one;
    return mix(calm, select(counter, turning, size > 0), 1 - exp(-abs(size) / scale));
}
"""


@cache
def shader() -> Shader:
    """The shader of the vorticity around the vortices: the unit directions along x and y, which place a sample, the
    scalar one, which reads the vorticity's size, and the colours, given as its constants."""
    mv = core.mv
    return Shader(supersampled() + source, core.ga, dict(Point=core.Point, Circulation=core.Circulation),
                  dict(directions=stack((mv.x, mv.y)).dual(), one=mv.scalar([1]), pi=np.pi, calm=calm, turning=turning,
                       counter=counter))


def vorticity_image(vortices: core.Vortices, middle: core.Point, half_height: float, shape: tuple[int, int],
                    scale: float) -> np.ndarray:
    """The vorticity on a window about its middle, half its height across, of `shape` rows and columns, coloured two
    thirds deep at `scale`: its RGB `[rows, columns, 3]`, from zero to one."""
    return shader()(shape, centres=vortices.centres, circulations=vortices.circulations, copies=vortices.copies,
                    core_squared=vortices.core_radius**2, middle=middle, half_height=half_height, scale=scale)["image"]


def as_bytes(image: np.ndarray) -> np.ndarray:
    return np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)


def animate(frames: Iterable[core.Vortices], half_height: float, shape: tuple[int, int], scale: float
            ) -> Iterator[np.ndarray]:
    """The vorticity on a window that follows the vortices' mean place, frame by frame."""
    for vortices in frames:
        yield as_bytes(vorticity_image(vortices, vortices.centres.mean(axis=-1), half_height, shape, scale))


def window(frames: Iterable[core.Vortices], middle: core.Point, half_height: float, shape: tuple[int, int],
           scale: float) -> Iterator[np.ndarray]:
    """The vorticity on one fixed window of the plane, frame by frame, each vortex trailing the path it has taken,
    stroked over it."""
    rows, columns = shape
    # The window's opposite corners, half its height above and below the middle and as far to either side as its
    # columns of square pixels reach, framed pixel for pixel in a room of its own shape.
    reach = core.mv.x.dual() * (half_height * columns / rows) + core.mv.y.dual() * half_height   # [] Point-
    corners = planar.origin + planar.context.include_from_algebra(stack([middle - reach, middle + reach]))   # [2] Point-
    view = planar.view([corners], 0.0, (columns, rows))
    travelled = []
    for vortices in frames:
        travelled.append(vortices.centres)
        # Each vortex's path so far, from where it started, which is the whole path on the first frame.
        offsets = stack([travelled[0]] + travelled, axis=1)                    # [vortices, frames + 1] Point-
        paths = planar.origin + planar.context.include_from_algebra(offsets)             # [vortices, frames + 1] Point-
        trails = planar.Strokes(paths, trail_width, trail_colour, trail_opacity, planar.SOLID, planar.Cap.ROUND)
        yield planar.draw(view, [trails], vorticity_image(vortices, middle, half_height, shape, scale))
