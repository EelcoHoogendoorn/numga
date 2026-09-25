"""Drawing for the spin groups: the table of signatures, and the orbits of the two isoclinic flows on
the three-sphere, seen through the stereographic projection."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

from examples.animation import capture
from examples.math.spin_groups import core


def components(points: core.Extensor) -> np.ndarray:
    """The x, y and z components of points in space."""
    return points.cast(points.algebra.subspace("x y z")).kernel


def blades(value: core.Extensor) -> str:
    """A multivector as the sum of its nonzero blades, for printing."""
    return " + ".join(f"{coefficient:g} {name}" for coefficient, name in zip(value.kernel, value.subspace.blade_names)
                      if abs(coefficient) > 1e-12)


def table(rows: Iterable[tuple], centres: Iterable[tuple]) -> str:
    """The signatures as text: generators, rotations and boosts, the invariant form as a multiple of
    the inner product, and the pseudoscalar's square. In even dimensions the pseudoscalar is a spinor
    and either splits the spinors in two or acts on them as the complex unit; in odd dimensions it is
    not, and the spinors are one piece."""
    spinors = {}
    for p, q, action in centres:
        values = action.to_array()
        spinors[p, q] = "two halves" if np.all(np.abs(values.imag) < 1e-9) else "complex"
    lines = [f"{'(p, q)':8s}{'planes':>8s}{'rotations':>11s}{'boosts':>8s}{'form / inner':>14s}{'I * I':>7s}  spinors"]
    for p, q, factor, signs, square in rows:
        values = np.real(signs.to_array())
        lines.append(f"{str((p, q)):8s}{len(values):8d}{int((values > 0).sum()):11d}{int((values < 0).sum()):8d}"
                     f"{float(factor.to_array()):14.1f}{float(square.to_array()):+7.0f}  {spinors[p, q] if (p + q) % 2 == 0 else 'one piece'}")
    return "\n".join(lines)


def colours(shape: tuple[int, int]) -> np.ndarray:
    """A colour per orbit: the start's position around the xy plane as the hue, its height as the value."""
    hue = np.broadcast_to(np.linspace(0.0, 1.0, shape[1], endpoint=False), shape)
    value = np.broadcast_to(np.linspace(0.95, 0.55, shape[0])[:, None], shape)
    return hsv_to_rgb(np.stack([hue, np.full(shape, 0.75), value], axis=-1))


def panels(figure: plt.Figure, extent: np.ndarray, elevations: tuple[float, ...]):
    """Views of space side by side, filling the figure, each fitted to the given half-widths and seen
    from its own elevation."""
    count = len(elevations)
    axes = [figure.add_axes((index / count, 0.0, 1 / count, 1.0), projection="3d") for index in range(count)]
    for ax, elevation in zip(axes, elevations):
        ax.set(xlim=(-extent[0], extent[0]), ylim=(-extent[1], extent[1]), zlim=(-extent[2], extent[2]))
        ax.set_box_aspect(tuple(extent), zoom=1.3)
        ax.set_axis_off()
        ax.view_init(elev=elevation, azim=-55)
    return axes


def extent(*drawn: np.ndarray) -> np.ndarray:
    """The half-widths along x, y and z that hold everything drawn."""
    points = np.concatenate([d.reshape(-1, 3) for d in drawn])
    return np.nanmax(np.abs(points), axis=0) * 1.02


def draw_orbits(ax, orbits: np.ndarray, width: float) -> None:
    """Each orbit as a closed curve, in the colour of its start."""
    rgb = colours(orbits.shape[:2])
    for curve, colour in zip(orbits.reshape(-1, *orbits.shape[-2:]), rgb.reshape(-1, 3)):
        ax.plot(*curve.T, color=colour, linewidth=width)


# One view for all three families.
ELEVATIONS = (24, 24, 24)


def draw_flows(left: core.Extensor, right: core.Extensor, knotted: core.Extensor) -> plt.Figure:
    """The orbits of the two isoclinic flows, each a family of circles filling nested tori and linked
    in opposite senses, beside the orbits of a rotation combining them at different rates: trefoil
    knots, one on each torus."""
    drawn = [components(orbits) for orbits in (left, right, knotted)]
    figure = plt.figure(figsize=(16, 5.5))
    for ax, orbits, width in zip(panels(figure, extent(*drawn), ELEVATIONS), drawn, (0.7, 0.7, 1.4)):
        draw_orbits(ax, orbits, width)
    return figure


def animate_flows(left: core.Extensor, right: core.Extensor, knotted: core.Extensor, frames: int) -> list[np.ndarray]:
    """The points of the three-sphere carried along every flow at once, their orbits drawn faintly;
    six points ride along each knot."""
    drawn = [components(orbits) for orbits in (left, right, knotted)]
    box = extent(*drawn)
    length = drawn[0].shape[2] - 1
    steps = np.linspace(0, length, frames, endpoint=False).astype(int)
    riders = np.arange(6) * length // 6
    images = []
    for step in steps:
        figure = plt.figure(figsize=(13, 4.4))
        axes = panels(figure, box, ELEVATIONS)
        for ax, orbits in zip(axes, drawn[:2]):
            draw_orbits(ax, orbits, 0.35)
            ax.scatter(*orbits[:, :, step].reshape(-1, 3).T, c=colours(orbits.shape[:2]).reshape(-1, 3), s=16,
                       depthshade=False)
        knots = drawn[2]
        draw_orbits(axes[2], knots, 0.6)
        positions = knots[:, :, (step + riders) % length].reshape(-1, 3)
        axes[2].scatter(*positions.T, c=np.repeat(colours(knots.shape[:2]).reshape(-1, 3), 6, axis=0), s=16, depthshade=False)
        images.append(capture(figure))
        plt.close(figure)
    return images
