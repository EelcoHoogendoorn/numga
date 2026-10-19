"""Drawing for the Hopf fibration: the spins on the sphere beside their fibres in space, each spin and
its fibre in the same colour."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

from examples.animation import capture
from examples.math.hopf import core

# Projected fibres are drawn out to this distance from the origin; the fibres near the projection
# point run off towards infinity.
LIMIT = 2.6


def components(bivectors: core.Bivector) -> np.ndarray:
    """The yz, zx and xy components of bivectors, drawn along x, y and z."""
    return bivectors.cast(core.ga.subspace("yz zx xy")).kernel


def spinor_components(spinors: core.Even) -> np.ndarray:
    """The scalar, yz, zx and xy components of spinors."""
    return spinors.cast(core.ga.subspace("1 yz zx xy")).kernel


def colour(spins: core.Bivector) -> np.ndarray:
    """An RGB colour for each spin: its azimuth as the hue, darker towards the south pole."""
    x, y, z = np.moveaxis(components(spins), -1, 0)
    hue = (np.arctan2(y, x) / (2 * np.pi)) % 1.0
    return hsv_to_rgb(np.stack([hue, np.full_like(hue, 0.75), 0.55 + 0.4 * (1 + z) / 2], axis=-1))


def clipped(points: np.ndarray) -> np.ndarray:
    """Points beyond the drawing limit replaced by NaN, which breaks the line there."""
    return np.where(np.linalg.norm(points, axis=-1, keepdims=True) > LIMIT, np.nan, points)


def panels(figure: plt.Figure, extent: np.ndarray):
    """A small sphere of spins on the left and space on the right, its box fitted to the given
    half-widths along x, y and z so that the drawing fills it."""
    base = figure.add_axes((0.0, 0.2, 0.28, 0.6), projection="3d")
    space = figure.add_axes((0.24, 0.0, 0.76, 1.0), projection="3d")
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 36), np.linspace(0, np.pi, 18))
    base.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v), color="0.85", linewidth=0.3)
    base.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1))
    base.set_box_aspect((1, 1, 1), zoom=1.5)
    space.set(xlim=(-extent[0], extent[0]), ylim=(-extent[1], extent[1]), zlim=(-extent[2], extent[2]))
    space.set_box_aspect(tuple(extent), zoom=1.3)
    for ax in (base, space):
        ax.set_axis_off()
        ax.view_init(elev=22, azim=-55)
    return base, space


def extent(*drawn: np.ndarray) -> np.ndarray:
    """The half-widths along x, y and z that hold everything drawn."""
    points = np.concatenate([d.reshape(-1, 3) for d in drawn])
    return np.nanmax(np.abs(points), axis=0) * 1.02


def draw_fibres(base, space, spins: core.Bivector, projected: core.Bivector, width: float) -> None:
    """Each spin as a dot on the sphere and its fibre as a line in space, in the spin's colour."""
    points, curves, colours = components(spins), clipped(components(projected)), colour(spins)
    points, curves, colours = points.reshape(-1, 3), curves.reshape(-1, curves.shape[-2], 3), colours.reshape(-1, 3)
    base.scatter(*points.T, c=colours, s=12, depthshade=False)
    for curve, rgb in zip(curves, colours):
        space.plot(*curve.T, color=rgb, linewidth=width)


def draw_tori(spins: core.Bivector, projected: core.Bivector) -> plt.Figure:
    """The fibres over circles of spins: each circle's fibres fill a torus, the tori nested, and any two
    fibres linked once."""
    figure = plt.figure(figsize=(11, 6))
    base, space = panels(figure, extent(clipped(components(projected))))
    draw_fibres(base, space, spins, projected, 0.9)
    return figure


def draw_lift(loop: core.Bivector, path: core.Bivector, fibre: core.Bivector) -> plt.Figure:
    """A loop of spins on the sphere, and in space the spinor carried around it: it leaves its
    fibre, drawn grey, and comes back onto it further along."""
    track, circle = clipped(components(path)), clipped(components(fibre))
    figure = plt.figure(figsize=(11, 6))
    base, space = panels(figure, extent(track, circle))
    base.plot(*components(loop).T, color="#c0392b", linewidth=1.5)
    space.plot(*circle.T, color="0.6", linewidth=1.2)
    space.plot(*track.T, color="#c0392b", linewidth=1.5)
    space.scatter(*track[[0, -1]].T, color=["#2e86c1", "#c0392b"], s=30, depthshade=False)
    return figure


def animate_sweep(stream: Iterable[tuple[core.Bivector, core.Bivector]]) -> list[np.ndarray]:
    """Fibres added one by one as their spin spirals over the sphere, the newest drawn heavier; one frame
    per spin, in a box that holds every fibre of the stream."""
    seen = list(stream)
    points = np.stack([components(spin) for spin, _ in seen])                   # [frames, 3]
    colours = np.stack([colour(spin) for spin, _ in seen])                      # [frames, 3]
    curves = [clipped(components(projected)) for _, projected in seen]          # [frames] of [samples + 1, 3]
    figure = plt.figure(figsize=(9, 5.5))
    base, space = panels(figure, extent(*curves))
    # Every spin's dot at once, each grown to its size from its own frame on.
    dots = base.scatter(*points.T, c=colours, s=0, depthshade=False)
    frames = []
    for count, (curve, rgb) in enumerate(zip(curves, colours), start=1):
        dots.set_sizes(12 * (np.arange(len(seen)) < count))
        newest, = space.plot(*curve.T, color=rgb, linewidth=2.2)
        frames.append(capture(figure))
        newest.set_linewidth(0.6)
    plt.close(figure)
    return frames
