"""Drawing for two spins: each spin's Bloch vector on a light sphere, entanglement and the largest Bell
combination over the exchange, the second spin steered by measurements of the first, the energies of
the four states of definite spin in a field, and the pair on the singlet-triplet qubit's sphere."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.animation import capture
from examples.quantum.two_spins import core

FIRST, SECOND, TOGETHER = "#c0392b", "#2e86c1", "#7d3c98"
OUTLINE = "0.75"
FOUR_NAMES = ("singlet", "triplet, zero spin", "triplet, both up", "triplet, both down")


def components(vectors: core.First | core.Second, names: str) -> np.ndarray:
    """The coordinates of vectors along the named directions."""
    return vectors.cast(vectors.algebra.subspace(names)).kernel


def ball(ax) -> None:
    """The unit sphere, lightly: its equator and two meridians."""
    turn = np.linspace(0, 2 * np.pi, 120)
    circle, zero = np.stack([np.cos(turn), np.sin(turn)]), np.zeros_like(turn)
    for x, y, z in ((circle[0], circle[1], zero), (circle[0], zero, circle[1]), (zero, circle[0], circle[1])):
        ax.plot(x, y, z, color=OUTLINE, linewidth=0.7)
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
    ax.set_box_aspect((1, 1, 1), zoom=1.3)
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-55)


def arrow(ax, tip: np.ndarray, colour: str) -> None:
    ax.plot(*np.stack([np.zeros(3), tip]).T, color=colour, linewidth=2.5)
    ax.scatter(*tip, color=colour, s=25, depthshade=False)


def draw_state(first: core.First, second: core.Second) -> plt.Figure:
    """Each spin's Bloch vector on its sphere, the first left and the second right."""
    figure = plt.figure(figsize=(7, 3.4))
    for column, (tip, colour) in enumerate(((components(first, "x y z"), FIRST), (components(second, "X Y Z"), SECOND))):
        ax = figure.add_axes((column / 2, 0.0, 0.5, 1.0), projection="3d")
        ball(ax)
        arrow(ax, tip, colour)
    return figure


def draw_exchange(angles: np.ndarray, bells: core.Scalar, first: core.First) -> plt.Figure:
    """Over the exchange: each spin's Bloch length, and the largest Bell combination against the bound
    for spins that carry their own answers and the bound for any pair."""
    figure, (length, bell) = plt.subplots(1, 2, figsize=(11, 3.0), layout="constrained")
    length.plot(angles, np.sqrt((first | first).to_array()), color=FIRST, linewidth=1.5)
    length.set(xlim=(angles[0], angles[-1]), ylim=(0.0, 1.05), xlabel="exchange angle (rad)", ylabel="each Bloch vector's length")
    bell.plot(angles, bells.to_array(), color=TOGETHER, linewidth=1.5)
    bell.axhline(2.0, color="0.5", linestyle="--", linewidth=0.8)
    bell.axhline(2 * np.sqrt(2), color="0.5", linestyle=":", linewidth=0.8)
    bell.set(xlim=(angles[0], angles[-1]), ylim=(1.9, 2.95), xlabel="exchange angle (rad)", ylabel="largest Bell combination")
    return figure


def animate_steering(directions: core.First, probability: core.Scalar, steered: core.Second) -> list[np.ndarray]:
    """The first spin's measurement directions on its sphere, and where finding +1 along each leaves
    the second spin, one frame per entry of the leading axis. Each direction keeps its colour, its
    coordinates read as red, green and blue, and each landing point is sized by how likely it is."""
    points = components(directions, "x y z").reshape(-1, 3)                  # [directions, 3]
    colours = np.clip((points + 1) / 2, 0.0, 1.0)
    figure = plt.figure(figsize=(8, 4.2), dpi=80)
    measured, landed = (figure.add_axes((column / 2, 0.0, 0.5, 1.0), projection="3d") for column in range(2))
    for ax in (measured, landed):
        ball(ax)
    measured.scatter(*points.T, c=colours, s=16, depthshade=False)
    targets = components(steered, "X Y Z").reshape(len(steered), -1, 3)       # [frames, directions, 3]
    weights = probability.kernel.reshape(len(probability), -1)                # [frames, directions]
    dots = landed.scatter(*targets[0].T, c=colours, s=40 * weights[0], depthshade=False)
    frames = []
    for target, weight in zip(targets, weights):
        dots._offsets3d = tuple(target.T)
        dots.set_sizes(40 * weight)
        frames.append(capture(figure))
    plt.close(figure)
    return frames


def draw_entanglement(angles: np.ndarray, singular_values: core.Scalar, first: core.First) -> plt.Figure:
    """The correlation map's singular values over the exchange, and the first spin's Bloch length."""
    values = singular_values.to_array()                                         # [angles, 3]
    figure, ax = plt.subplots(figsize=(8, 3.0), layout="constrained")
    ax.plot(angles, values[:, 0], color=TOGETHER, linewidth=1.5, label="largest singular value")
    ax.plot(angles, values[:, 1], color=TOGETHER, linewidth=1.5, linestyle="--", label="the other two: the concurrence")
    ax.plot(angles, np.sqrt((first | first).to_array()), color=FIRST, linewidth=1.5, label="each Bloch vector's length")
    ax.set(xlim=(angles[0], angles[-1]), ylim=(0.0, 1.05), xlabel="exchange angle (rad)")
    ax.legend(frameon=False, loc="center right")
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
    levels.set(xlabel="field along z", ylabel="energy", xlim=(fields[0], fields[-1]))
    levels.legend(frameon=False, fontsize=8)
    spin.plot(fields, spins.to_array()[lowest], color="0.2", linewidth=1.5)
    spin.set(xlabel="field along z", ylabel="total spin along z, lowest state", xlim=(fields[0], fields[-1]), ylim=(-0.1, 2.1))
    return figure


def draw_qubit(across: core.Scalar, turned: core.Scalar, balance: core.Scalar, times: np.ndarray,
               back_in_singlet: core.Scalar, exchange_rates: np.ndarray) -> plt.Figure:
    """The pair on the singlet-triplet qubit's sphere over time, one path per exchange rate: singlet at
    the top, the triplet of zero spin at the bottom, up-down and down-up on the horizontal axis through
    the front. Beside it, how likely the pair is found back in the singlet."""
    paths = np.stack([across.to_array(), turned.to_array(), balance.to_array()], axis=-1)   # [rates, times, 3]
    shades = (TOGETHER, FIRST, SECOND)
    figure = plt.figure(figsize=(11, 4.2))
    sphere = figure.add_axes((0.0, 0.0, 0.45, 1.0), projection="3d")
    ball(sphere)
    for place, text in (((0, 0, 1.15), "singlet"), ((0, 0, -1.25), "triplet"), ((1.3, 0, 0), "down-up"), ((-1.3, 0, 0), "up-down")):
        sphere.text(*place, text, ha="center", fontsize=9, color="0.35")
    for path, colour in zip(paths, shades):
        sphere.plot(*path.T, color=colour, linewidth=2.0)
    sphere.scatter(*paths[0, 0], color="0.1", s=30, depthshade=False)
    curves = figure.add_axes((0.53, 0.18, 0.44, 0.7))
    for rate, curve, colour in zip(exchange_rates, back_in_singlet.to_array(), shades):
        curves.plot(times, curve, color=colour, linewidth=1.5, label=f"exchange rate {rate:g}")
    curves.set(xlim=(times[0], times[-1]), ylim=(-0.05, 1.05), xlabel="time, in units of the field difference", ylabel="back in the singlet")
    curves.legend(frameon=False, fontsize=8, loc="lower left")
    return figure
