"""Drawing for two spins: each spin's Bloch vector on a light sphere, entanglement and the largest Bell
combination over the exchange, the second spin steered by measurements of the first, the energies of
the four states of definite spin in a field, and the pair on the singlet-triplet qubit's sphere."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import Extensor
from examples.util.animation import capture
from examples.quantum.two_spins import core
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
REACH, N_PIXELS = 1.3, 700
# The widths of the great circles, the paths and the arrows' shafts, the arrowheads' length and width, and the radii
# of the dots, the largest landing dot and the start, in radii of the sphere.
CIRCLE_WIDTH, PATH_WIDTH, ARROW_WIDTH, HEAD_LENGTH, HEAD_WIDTH = 0.00891, 0.026, 0.0223, 0.111, 0.0669
DOT_RADIUS, LANDING_RADIUS, START_RADIUS = 0.0186, 0.0334, 0.026


def components(vectors: core.First | core.Second, names: str) -> np.ndarray:
    """The coordinates of vectors along the named directions."""
    return vectors.cast(vectors.algebra.subspace(names)).kernel


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
    ax.axhline(2.0, color="0.5", linestyle="--", linewidth=0.8)
    ax.axhline(2 * np.sqrt(2), color="0.5", linestyle=":", linewidth=0.8)
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
    shapes = solids.ellipsoid(radii[:, None] * np.ones(3))                    # [dots] Quadric
    return solids.placed(shapes, solids.translator(sm.point(centres) - solids.origin))[:, None]


def draw_pair(tips: list[tuple[np.ndarray, str]]) -> plt.Figure:
    """Each spin's arrow `[3]` on its sphere, in its colour, side by side."""
    figure = plt.figure(figsize=(7, 3.4))
    for column, (tip, colour) in enumerate(tips):
        ax = figure.add_axes((column / 2, 0.0, 0.5, 1.0))
        view = ball(ax)
        show(ax, on_sphere(arrow(tip, colour), view), view)
    return figure


def draw_state(first: core.First, second: core.Second) -> plt.Figure:
    """Each spin's Bloch vector on its sphere, the first left and the second right."""
    return draw_pair([(components(first, "x y z"), FIRST), (components(second, "X Y Z"), SECOND)])


def draw_exchange(angles: np.ndarray, bells: core.Scalar, first: core.First) -> plt.Figure:
    """Over the exchange: each spin's Bloch length, and the largest Bell combination against the bound
    for spins that carry their own answers and the bound for any pair."""
    figure, (length, bell) = plt.subplots(1, 2, figsize=(11, 3.0), layout="constrained")
    length.plot(angles, np.sqrt((first | first).to_array()), color=FIRST, linewidth=1.5)
    length.set(xlim=(angles[0], angles[-1]), ylim=(0.0, 1.05))
    plain_axes(length, "exchange angle", "each Bloch vector's length")
    bell.plot(angles, bells.to_array(), color=TOGETHER, linewidth=1.5)
    bell_bounds(bell, angles)
    return figure


def steering_frames(points: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> Iterator[np.ndarray]:
    """Points `[measured, 3]` on the left sphere, each its coordinates read as red, green and blue, and on the right
    where each lands `[frames, measured, 3]`, its dot's area in proportion to its weight `[frames, measured]`; one
    frame per entry of the leading axis."""
    colours = np.clip((points + 1) / 2, 0.0, 1.0)
    figure = plt.figure(figsize=(8, 4.2), dpi=80)
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


def animate_steering(directions: core.First, probability: core.Scalar, steered: core.Second) -> Iterator[np.ndarray]:
    """The first spin's measurement directions on its sphere, and where finding +1 along each leaves
    the second spin, one frame per entry of the leading axis. Each direction keeps its colour, its
    coordinates read as red, green and blue, and each landing point is sized by how likely it is."""
    points = components(directions, "x y z").reshape(-1, 3)                  # [directions, 3]
    targets = components(steered, "X Y Z").reshape(len(steered), -1, 3)       # [frames, directions, 3]
    weights = probability.kernel.reshape(len(probability), -1)                # [frames, directions]
    return steering_frames(points, targets, weights)


def draw_entanglement(angles: np.ndarray, singular_values: core.Scalar, first: core.First) -> plt.Figure:
    """The correlation map's singular values over the exchange, and the first spin's Bloch length."""
    values = singular_values.to_array()                                         # [angles, 3]
    figure, ax = plt.subplots(figsize=(8, 3.0), layout="constrained")
    ax.plot(angles, values[:, 0], color=TOGETHER, linewidth=1.5, label="largest singular value")
    ax.plot(angles, values[:, 1], color=TOGETHER, linewidth=1.5, linestyle="--", label="the other two: the concurrence")
    ax.plot(angles, np.sqrt((first | first).to_array()), color=FIRST, linewidth=1.5, label="each Bloch vector's length")
    ax.set(xlim=(angles[0], angles[-1]), ylim=(0.0, 1.05))
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=3)
    plain_axes(ax, "exchange angle", "")
    return figure


def draw_levels(fields: np.ndarray, energies: core.Scalar, spins: core.Scalar) -> plt.Figure:
    """The energies of the four states of definite spin along z as the field rises, the lowest traced
    over them, and the total spin along z of the lowest."""
    values = energies.to_array()                                                # [fields, 4]
    lowest = np.argmin(values, axis=-1)                                         # [fields]
    figure, (levels, spin) = plt.subplots(1, 2, figsize=(11, 3.4), layout="constrained")
    for column, (name, colour) in enumerate(zip(FOUR_NAMES, (TOGETHER, "0.55", FIRST, SECOND))):
        levels.plot(fields, values[:, column], color=colour, linewidth=1.5, label=name)
    levels.plot(fields, values.min(axis=-1), color="0.1", linewidth=4.0, alpha=0.25)
    levels.set(xlim=(fields[0], fields[-1]))
    levels.legend(frameon=False, fontsize=8, loc="upper left")
    plain_axes(levels, "field along z", "energy")
    spin.plot(fields, spins.to_array()[lowest], color="0.2", linewidth=1.5)
    spin.set(xlim=(fields[0], fields[-1]), ylim=(-0.1, 2.1))
    plain_axes(spin, "field along z", "total spin along z, lowest state")
    return figure


def draw_qubit(across: core.Scalar, turned: core.Scalar, balance: core.Scalar, times: np.ndarray,
               back_in_singlet: core.Scalar, exchange_rates: np.ndarray) -> plt.Figure:
    """The pair on the singlet-triplet qubit's sphere over time, one path per exchange rate: singlet at
    the top, the triplet of zero spin at the bottom, up-down and down-up on the horizontal axis through
    the front. Beside it, how likely the pair is found back in the singlet."""
    paths = np.stack([across.to_array(), turned.to_array(), balance.to_array()], axis=-1)   # [rates, times, 3]
    shades = (TOGETHER, FIRST, SECOND)
    figure = plt.figure(figsize=(11, 4.2))
    ax = figure.add_axes((0.0, 0.0, 0.45, 1.0))
    view = ball(ax)
    tracks = strokes.polylines([(sm.point(path), np.array(to_rgb(colour))) for path, colour in zip(paths, shades)],
                               PATH_WIDTH / view.pixel, view)
    start = [(solids.ball(sm.point(paths[0, 0]), START_RADIUS)[None, None], START_COLOUR)]
    show(ax, on_sphere(start, view, [(tracks, 1.0)]), view)
    # Each pole named just past the sphere, where the camera sees it.
    places = np.array([(0, 0, 1.15), (0, 0, -1.2), (1.25, 0, 0), (-1.25, 0, 0)])
    labels(ax, view, sm.point(places), ["singlet", "triplet", "down-up", "up-down"], ha="center", va="center",
           fontsize=9, color=LABEL_COLOUR)
    curves = figure.add_axes((0.53, 0.18, 0.44, 0.7))
    for rate, curve, colour in zip(exchange_rates, back_in_singlet.to_array(), shades):
        curves.plot(times, curve, color=colour, linewidth=1.5, label=f"exchange rate {rate:g}")
    curves.set(xlim=(times[0], times[-1]), ylim=(-0.05, 1.05))
    curves.legend(frameon=False, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=3)
    plain_axes(curves, "time", "back in the singlet")
    return figure


# --- spin planes ----------------------------------------------------------------------
# In two copies of spacetime each spin shows itself as the plane it turns in, drawn as a point of the
# three-dimensional space of its planes, with coordinates along yz, zx and xy of its own space.
FIRST_PLANES, SECOND_PLANES = "yz zx xy", "YZ ZX XY"


def plane_sizes(planes: Extensor, names: str) -> np.ndarray:
    """The size of each plane: the length of its coordinates along the named planes."""
    return np.linalg.norm(components(planes, names), axis=-1)


def draw_planes(first: Extensor, second: Extensor) -> plt.Figure:
    """Each spin's plane in the space of its planes, the first left and the second right."""
    return draw_pair([(components(first, FIRST_PLANES), FIRST), (components(second, SECOND_PLANES), SECOND)])


def draw_exchange_planes(angles: np.ndarray, bells: Extensor, first: Extensor) -> plt.Figure:
    """Over the exchange: the size of each spin's plane, and the largest Bell combination against the
    bound for spins that carry their own answers and the bound for any pair."""
    figure, (size, bell) = plt.subplots(1, 2, figsize=(11, 3.0), layout="constrained")
    size.plot(angles, plane_sizes(first, FIRST_PLANES), color=FIRST, linewidth=1.5)
    size.set(xlim=(angles[0], angles[-1]), ylim=(0.0, 1.05))
    plain_axes(size, "exchange angle", "size of each spin's plane")
    bell.plot(angles, bells.to_array(), color=TOGETHER, linewidth=1.5)
    bell_bounds(bell, angles)
    return figure


def animate_steering_planes(planes: Extensor, probability: Extensor, steered: Extensor) -> Iterator[np.ndarray]:
    """The first spin's measurement planes in the space of its planes, and where finding +1 in each
    leaves the second spin's plane, one frame per entry of the leading axis. Each measurement plane keeps
    its colour, its coordinates read as red, green and blue, and each landing point is sized by how
    likely it is."""
    points = components(planes, FIRST_PLANES).reshape(-1, 3)                  # [planes, 3]
    targets = components(steered, SECOND_PLANES).reshape(len(steered), -1, 3)  # [frames, planes, 3]
    weights = probability.kernel.reshape(len(probability), -1)                # [frames, planes]
    return steering_frames(points, targets, weights)


def draw_entanglement_planes(angles: np.ndarray, singular_values: Extensor, first: Extensor) -> plt.Figure:
    """The correlation map's singular values over the exchange, and the size of the first spin's plane."""
    values = singular_values.to_array()                                         # [angles, 3]
    figure, ax = plt.subplots(figsize=(8, 3.0), layout="constrained")
    ax.plot(angles, values[:, 0], color=TOGETHER, linewidth=1.5, label="largest singular value")
    ax.plot(angles, values[:, 1], color=TOGETHER, linewidth=1.5, linestyle="--", label="the other two: the concurrence")
    ax.plot(angles, plane_sizes(first, FIRST_PLANES), color=FIRST, linewidth=1.5, label="size of each spin's plane")
    ax.set(xlim=(angles[0], angles[-1]), ylim=(0.0, 1.05))
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1), ncol=3)
    plain_axes(ax, "exchange angle", "")
    return figure
