"""Drawing for two spins: the plane each spin turns in, as a point of the light sphere of its unit planes,
entanglement and the largest Bell combination over the exchange, the second spin steered by measurements
of the first, the energies of the four states of definite spin in a field, and the pair on the
singlet-triplet qubit's sphere.

Each spin's plane is drawn as a point of the three-dimensional space of its planes, with coordinates
along yz, zx and xy of its own space."""

from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import Extensor, stack
from examples.util import charts
from examples.util.animation import capture
from examples.util.wgsl import fitted, labels, meshes, show, solids, strokes
from examples.util.wgsl.scene import Scene

sm = solids.mv
FIRST, SECOND, TOGETHER = "#c0392b", "#2e86c1", "#7d3c98"
FOUR_NAMES = ("singlet", "triplet, zero spin", "triplet, both up", "triplet, both down")
# The sphere a spin's Bloch vector reaches, faint, how much each of its surfaces hides, and its equator and two
# meridians inked on it; the names of the qubit's poles and the dot where its paths start.
SPHERE_COLOUR, SPHERE_OPACITY = np.array([0.70, 0.76, 0.86]), 0.09
CIRCLE_COLOUR, CIRCLE_OPACITY = np.full(3, 0.6), 0.6
LABEL_COLOUR, START_COLOUR = "0.35", np.full(3, 0.1)
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the half width of the square view
# about the centre, and the pixels across it, until the view is fitted to its axes.
ELEVATION, AZIMUTH = np.radians(18.0), np.radians(-55.0 + 90.0)
REACH, N_PIXELS = 1.3, 300
# The widths of the great circles, the paths and the arrows' shafts, the arrowheads' length and width, and the radii
# of the dots, the largest landing dot and the start, in radii of the sphere.
CIRCLE_WIDTH, PATH_WIDTH, ARROW_WIDTH, HEAD_LENGTH, HEAD_WIDTH = 0.00891, 0.026, 0.0223, 0.111, 0.0669
DOT_RADIUS, LANDING_RADIUS, START_RADIUS = 0.0186, 0.0334, 0.026


class Reading(NamedTuple):
    """How a spin is read: the unit elements of the first and the second spin's own space, by name, and what the
    size of a spin read so is called."""
    first: str
    second: str
    size: str


# A spin read as the plane it turns in, and, in the twin notebook of space, as a Bloch vector.
SPIN_PLANES = Reading("yz zx xy", "YZ ZX XY", "size of each spin's plane")
BLOCH_VECTORS = Reading("x y z", "X Y Z", "each Bloch vector's length")


def coordinates(spins: Extensor, names: str) -> np.ndarray:
    """The coordinates `[..., 3]` of spins along the named unit elements of their own algebra: their dot products
    with them."""
    mv = spins.context.multivector
    units = stack([getattr(mv, name) for name in names.split()])
    return units.dot(spins[..., None]).to_array()


def sizes(spins: Extensor, names: str) -> np.ndarray:
    """The size of each spin: the length of its coordinates along the named unit elements."""
    return np.linalg.norm(coordinates(spins, names), axis=-1)


def plain_axes(ax: plt.Axes, across: str, up: str) -> None:
    """Axes named by what runs along them, the quantity's zero as the horizontal one, without numbers."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_position(("data", 0))
    ax.set(xticks=[], yticks=[], xlabel=across, ylabel=up)
    # The horizontal axis's name below the plot, wherever its zero lies.
    ax.xaxis.set_label_coords(0.5, -0.03)
    ax.xaxis.label.set_va("top")


def bell_bounds(ax: plt.Axes, angles: np.ndarray) -> None:
    """The largest Bell combination over the exchange against its two bounds, the only values marked: 2 for spins
    that carry their own answers, dashed, and 2 sqrt 2 for any pair, dotted."""
    ax.axhline(2.0, color="0.5", linestyle="--", linewidth=charts.FRAME_WIDTH)
    ax.axhline(2 * np.sqrt(2), color="0.5", linestyle=":", linewidth=charts.FRAME_WIDTH)
    ax.set(xlim=(angles[0], angles[-1]), ylim=(0.0, 3.0))
    plain_axes(ax, "exchange angle", "largest Bell combination")
    ax.set(yticks=[2.0, 2 * np.sqrt(2)], yticklabels=["2", r"$2\sqrt{2}$"])


def ball(ax: plt.Axes) -> solids.View:
    """The view of the unit sphere: square about its centre, fitted to the matplotlib axes it is shown in."""
    return fitted(solids.View(solids.aim(solids.origin, ELEVATION, AZIMUTH), np.full(2, REACH), N_PIXELS), ax)


def sphere_scene(view: solids.View, n_parts: int) -> Scene:
    """The unit sphere, lightly, set up for frames of `n_parts` opaque parts among it: a faint translucent ball with
    its equator and two meridians, the great circles where the planes x, y and z cut it, inked on it."""
    sphere = [(solids.ball(solids.origin, 1.0)[None, None], SPHERE_COLOUR, SPHERE_OPACITY)]
    # The ball is the first translucent part, after the opaque ones.
    circles = solids.Ink(n_parts, solids.half_space(solids.axes), solids.everywhere[None].broadcast_to((3,)),
                         CIRCLE_COLOUR, CIRCLE_WIDTH / view.pixel, CIRCLE_OPACITY)
    return Scene(view, np.ones(3), [], [], inks=[circles], translucent=sphere)


def on_sphere(parts: list[tuple[solids.Quadric, np.ndarray]], view: solids.View,
              lines: list[tuple[meshes.Samples, float]] = ()) -> np.ndarray:
    """Solids and lines among the unit sphere, as `sphere_scene` sets it up."""
    return sphere_scene(view, len(parts)).draw(parts, inks=[], translucent=[], meshes=lines)


def arrow(tip: np.ndarray, colour: str) -> list[tuple[solids.Quadric, np.ndarray]]:
    """An arrow from the centre to a tip `[3]`, in a colour."""
    shaft, head = solids.arrows(solids.origin, sm.point(tip), ARROW_WIDTH / 2, HEAD_LENGTH, HEAD_WIDTH / 2)
    return [(shaft[None], np.array(to_rgb(colour))), (head[None], np.array(to_rgb(colour)))]


def dots(centres: np.ndarray, radii: np.ndarray) -> solids.Quadric:
    """Balls about points `[dots, 3]`, each of its own radius `[dots]`, `[dots, 1]` quadrics."""
    shapes = solids.ellipsoid(radii[:, None] * np.ones(3))                     # [dots] Plane+ <- Point-
    return solids.placed(shapes, solids.translator(sm.point(centres) - solids.origin))[:, None]


def draw_pair(tips: list[tuple[np.ndarray, str]]) -> plt.Figure:
    """Each spin's arrow `[3]` on its sphere, in its colour, side by side."""
    figure = plt.figure(figsize=(6.0, 3.0))
    for column, (tip, colour) in enumerate(tips):
        ax = figure.add_axes((column / 2, 0.0, 0.5, 1.0))
        view = ball(ax)
        show(ax, on_sphere(arrow(tip, colour), view), view)
    return figure


def draw_state(first: Extensor, second: Extensor, reading: Reading) -> plt.Figure:
    """Each spin on its sphere as the reading places it, the first left and the second right."""
    return draw_pair([(coordinates(first, reading.first), FIRST), (coordinates(second, reading.second), SECOND)])


@charts.style()
def draw_exchange(angles: np.ndarray, bells: Extensor, first: Extensor, reading: Reading) -> plt.Figure:
    """Over the exchange: the size of each spin, and the largest Bell combination against the bound for
    spins that carry their own answers and the bound for any pair."""
    # A smooth curve in each of two panels, wide enough for their long labels.
    figure, (size, bell) = charts.subplots(640, 260, ncols=2, layout="constrained")
    size.plot(angles, sizes(first, reading.first), color=FIRST)
    size.set(xlim=(angles[0], angles[-1]), ylim=(0.0, 1.05))
    plain_axes(size, "exchange angle", reading.size)
    bell.plot(angles, bells.to_array(), color=TOGETHER)
    bell_bounds(bell, angles)
    return figure


def steering_frames(points: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> Iterator[np.ndarray]:
    """Points `[measured, 3]` on the left sphere, each its coordinates read as red, green and blue, and on the right
    where each lands `[frames, measured, 3]`, its dot's area in proportion to its weight `[frames, measured]`; one
    frame per entry of the leading axis."""
    colours = np.clip((points + 1) / 2, 0.0, 1.0)
    figure = plt.figure(figsize=(7.5, 3.75), dpi=80)
    measured_ax, landed_ax = (figure.add_axes((column / 2, 0.0, 0.5, 1.0)) for column in range(2))
    view = ball(measured_ax)
    show(measured_ax, on_sphere([(solids.ball(sm.point(points), DOT_RADIUS)[:, None], colours)], view), view)

    # The sphere the dots land on, set up once.
    landing = sphere_scene(view, 1)

    def landed(target: np.ndarray, weight: np.ndarray) -> np.ndarray:
        return landing.draw([(dots(target, LANDING_RADIUS * np.sqrt(weight)), colours)], inks=[], translucent=[],
                            meshes=[])

    image = show(landed_ax, landed(targets[0], weights[0]), view)
    for target, weight in zip(targets, weights):
        image.set_data(landed(target, weight))
        yield capture(figure)
    plt.close(figure)


def animate_steering(measured: Extensor, probability: Extensor, steered: Extensor, reading: Reading
                     ) -> Iterator[np.ndarray]:
    """The first spin's measurements on its sphere, and where finding +1 in each leaves the second spin,
    one frame per entry of the leading axis. Each measurement keeps its colour, its coordinates read as
    red, green and blue, and each landing point is sized by how likely it is."""
    points = coordinates(measured, reading.first).reshape(-1, 3)              # [measured, 3]
    targets = coordinates(steered, reading.second).reshape(len(steered), -1, 3)   # [frames, measured, 3]
    weights = probability.to_array().reshape(len(probability), -1)             # [frames, planes]
    return steering_frames(points, targets, weights)


@charts.style()
def draw_entanglement(angles: np.ndarray, singular_values: Extensor, first: Extensor, reading: Reading) -> plt.Figure:
    """The correlation map's singular values over the exchange, and the size of the first spin."""
    values = singular_values.to_array()                                        # [angles, 3]
    # Three smooth curves under a legend of their long names, two to a row.
    figure, ax = charts.subplots(480, 300, layout="constrained")
    ax.plot(angles, values[:, 0], color=TOGETHER, label="largest singular value")
    ax.plot(angles, values[:, 1], color=TOGETHER, linestyle="--", label="the other two: the concurrence")
    ax.plot(angles, sizes(first, reading.first), color=FIRST, label=reading.size)
    ax.set(xlim=(angles[0], angles[-1]), ylim=(0.0, 1.05))
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2)
    plain_axes(ax, "exchange angle", "")
    return figure


@charts.style()
def draw_levels(fields: np.ndarray, energies: Extensor, spins: Extensor) -> plt.Figure:
    """The energies of the four states of definite spin in the field's plane as the field rises, the
    lowest traced over them, and the total spin in the field's plane of the lowest."""
    values = energies.to_array()                                               # [fields, states]
    lowest = np.argmin(values, axis=-1)                                        # [fields]
    # Four straight levels and a step, side by side, over the levels' legend in one row.
    figure, (levels, spin) = charts.subplots(680, 320, ncols=2, layout="constrained")
    for column, (name, colour) in enumerate(zip(FOUR_NAMES, (TOGETHER, "0.55", FIRST, SECOND))):
        levels.plot(fields, values[:, column], color=colour, label=name)
    # The lowest level traced over them as a broad faint band, three curves wide.
    levels.plot(fields, values.min(axis=-1), color="0.1", linewidth=charts.points(3 * charts.LINE_PX), alpha=0.25)
    levels.set(xlim=(fields[0], fields[-1]))
    figure.legend(*levels.get_legend_handles_labels(), frameon=False, loc="outside lower center", ncol=len(FOUR_NAMES))
    plain_axes(levels, "field", "energy")
    spin.plot(fields, spins.to_array()[lowest], color="0.2")
    spin.set(xlim=(fields[0], fields[-1]), ylim=(-0.1, 2.1))
    plain_axes(spin, "field", "total spin, lowest state")
    return figure


@charts.style()
def draw_qubit(across: Extensor, turned: Extensor, balance: Extensor, times: np.ndarray,
               back_in_singlet: Extensor, exchange_rates: np.ndarray) -> plt.Figure:
    """The pair on the singlet-triplet qubit's sphere over time, one path per exchange rate: singlet at
    the top, the triplet of zero spin at the bottom, up-down and down-up on the horizontal axis through
    the front. Beside it, how likely the pair is found back in the singlet."""
    paths = np.stack([across.to_array(), turned.to_array(), balance.to_array()], axis=-1)   # [rates, times, 3]
    shades = (TOGETHER, FIRST, SECOND)
    # The sphere beside the curves, the column's width but the room the legend needs past them.
    figure = plt.figure(figsize=(760 / charts.DPI, 320 / charts.DPI))
    ax = figure.add_axes((0.0, 0.0, 0.45, 1.0))
    view = ball(ax)
    tracks = strokes.polylines([(sm.point(path), np.array(to_rgb(colour))) for path, colour in zip(paths, shades)],
                               PATH_WIDTH / view.pixel, view)
    start = [(solids.ball(sm.point(paths[0, 0]), START_RADIUS)[None, None], START_COLOUR)]
    show(ax, on_sphere(start, view, [(tracks, 1.0)]), view)
    # Each pole named just past the sphere, where the camera sees it.
    places = np.array([(0, 0, 1.15), (0, 0, -1.2), (1.25, 0, 0), (-1.25, 0, 0)])
    labels(ax, view, sm.point(places), ["singlet", "triplet", "down-up", "up-down"], ha="center", va="center",
           color=LABEL_COLOUR)
    curves = figure.add_axes((0.53, 0.18, 0.44, 0.7))
    for rate, curve, colour in zip(exchange_rates, back_in_singlet.to_array(), shades):
        curves.plot(times, curve, color=colour, label=f"exchange rate {rate:g}")
    curves.set(xlim=(times[0], times[-1]), ylim=(-0.05, 1.05))
    # The legend in one row above the curves, flush with their right edge, wider than they are.
    curves.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1), ncol=3)
    plain_axes(curves, "time", "back in the singlet")
    return figure
