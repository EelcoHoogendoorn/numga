"""Drawing for the spin groups: the table of signatures, and the orbits of the two isoclinic flows on
the three-sphere, seen through the stereographic projection: tubes and dots drawn on the GPU."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

from examples.math.spin_groups import core
from examples.wgsl import compose, meshes, show, solids

sm = solids.mv
BACKGROUND = np.ones(3)
# One view for all three families: the elevation and azimuth in radians, azimuth zero looking along +y, the margin
# around the orbits as a share of the larger side, and the pixels across a panel of the still and of a frame.
ELEVATION, AZIMUTH = np.radians(24.0), np.radians(35.0)
MARGIN = 0.14
PANEL_PIXELS, FRAME_PIXELS = 540, 440
# The points around a tube; the widths in pixels of the circles and the knots in the still, and of the faint orbits
# in the animation, and their opacity; the riders' radius in pixels.
N_AROUND = 8
CIRCLE_WIDTH, KNOT_WIDTH, FAINT_WIDTH, FAINT_OPACITY = 2.0, 3.2, 1.6, 0.45
RIDER_RADIUS = 3.2


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


def view(pixels: int, *drawn: np.ndarray) -> solids.View:
    """One view for every panel, `pixels` across, framing everything drawn `[..., 3]`."""
    return solids.view([sm.point(points) for points in drawn], 0.0, ELEVATION, AZIMUTH, MARGIN, pixels)


def orbit_layer(orbits: np.ndarray, width: float, view: solids.View) -> tuple[np.ndarray, np.ndarray]:
    """Each orbit `[heights, azimuths, steps + 1, 3]` as a tube of a width in pixels, in the colour of its start."""
    curves = zip(orbits.reshape(-1, *orbits.shape[-2:]), colours(orbits.shape[:2]).reshape(-1, 3))
    return meshes.tubes(list(curves), width, view, N_AROUND)


def rider_layer(positions: np.ndarray, rgb: np.ndarray, view: solids.View) -> tuple[np.ndarray, np.ndarray]:
    """A dot at each position `[n, 3]`, in its colour `[n, 3]`."""
    balls = solids.ball(sm.point(positions), RIDER_RADIUS * view.pixel)[:, None]
    return solids.layer([(balls, rgb)], view)


def draw_flows(left: core.Extensor, right: core.Extensor, knotted: core.Extensor) -> plt.Figure:
    """The orbits of the two isoclinic flows, each a family of circles filling nested tori and linked
    in opposite senses, beside the orbits of a rotation combining them at different rates: trefoil
    knots, one on each torus."""
    drawn = [components(orbits) for orbits in (left, right, knotted)]
    panel = view(PANEL_PIXELS, *drawn)
    figure = plt.figure(figsize=(16, 5.5))
    for index, (orbits, width) in enumerate(zip(drawn, (CIRCLE_WIDTH, CIRCLE_WIDTH, KNOT_WIDTH))):
        ax = figure.add_axes((index / len(drawn), 0.0, 1 / len(drawn), 1.0))
        show(ax, compose([(*orbit_layer(orbits, width, panel), 1.0)], BACKGROUND), panel)
    return figure


def animate_flows(left: core.Extensor, right: core.Extensor, knotted: core.Extensor, frames: int) -> list[np.ndarray]:
    """The points of the three-sphere carried along every flow at once, their orbits drawn faintly;
    six points ride along each knot. The three panels side by side, seen by one camera."""
    drawn = [components(orbits) for orbits in (left, right, knotted)]
    frame = view(FRAME_PIXELS, *drawn)
    faint = [(*orbit_layer(orbits, FAINT_WIDTH, frame), FAINT_OPACITY) for orbits in drawn]
    length = drawn[0].shape[2] - 1
    steps = np.linspace(0, length, frames, endpoint=False).astype(int)
    riders = np.arange(6) * length // 6
    knots = drawn[2]
    images = []
    for step in steps:
        # Every start carried along the two flows, and six points along each knot.
        riding = [(orbits[:, :, step].reshape(-1, 3), colours(orbits.shape[:2]).reshape(-1, 3)) for orbits in drawn[:2]]
        riding.append((knots[:, :, (step + riders) % length].reshape(-1, 3),
                       np.repeat(colours(knots.shape[:2]).reshape(-1, 3), len(riders), axis=0)))
        images.append(np.concatenate([compose([orbits, (*rider_layer(*points, frame), 1.0)], BACKGROUND)
                                      for orbits, points in zip(faint, riding)], axis=1))
    return images
