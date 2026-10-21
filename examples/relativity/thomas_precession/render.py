"""Drawing for Thomas precession: velocities in the Poincaré disk, a particle bouncing in a box, a
storage ring, and the anomalous precession rate against the electric field."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

from numga import stack

from examples.animation import capture
from examples.relativity.thomas_precession import core

INK, FAINT, FILL, SPIN = "#1e293b", "#cbd5e1", "#dbeafe", "#ea580c"
orbit_colours = ("#0f766e", "#2563eb", "#7c3aed")
line_colours = ("#64748b", "#ea580c")
# Arrow lengths: in the Poincaré disk, around the orbits, and as a share of the box's or ring's half
# width; the view's reach around the orbits.
DISK_ARROW, ORBIT_ARROW, SCENE_ARROW = 0.16, 0.09, 0.3
orbit_reach = 0.66
# The window of the disk on the rocket's triangle at its largest: the centre, and its half width.
TRIANGLE_CENTRE, TRIANGLE_REACH = (0.5, 0.2), 0.62
FRAME_INCHES, DPI = 5.0, 90


# --- plumbing -------------------------------------------------------------------------
def disk(velocities: core.Antivector) -> np.ndarray:
    """Velocities as points of the Poincaré disk, their spatial part over one plus their time part,
    on the last axis."""
    t, x, y = np.moveaxis((stack((core.mv.t, core.mv.x, core.mv.y)) & velocities[..., None]).to_array(), -1, 0)
    return np.stack([x, y], axis=-1) / (1 + t)[..., None]


def planar(directions: core.Antivector) -> np.ndarray:
    """x and y of directions and positions, on the last axis: their pairings with the slices x and y."""
    return (stack((core.mv.x, core.mv.y)) & directions[..., None]).to_array()


def unit(directions: core.Antivector) -> np.ndarray:
    """The directions in the plane, on the last axis."""
    xy = planar(directions)
    return xy / np.linalg.norm(xy, axis=-1, keepdims=True)


def spin_axes(spins: core.Bivector) -> np.ndarray:
    """The arrow each spin plane turns about, drawn by the right-hand rule, in the plane, on the last axis."""
    # The spins along x and y.
    planes = stack((core.about(core.mv.x.dual()), core.about(core.mv.y.dual())))   # [2] Bivector
    xy = (spins[..., None] | planes.inverse()).to_array()
    return xy / np.linalg.norm(xy, axis=-1, keepdims=True)


def arrows(ax: plt.Axes, tails: np.ndarray, directions: np.ndarray, length: float, colour: str, width: float):
    return ax.quiver(tails[..., 0], tails[..., 1], length * directions[..., 0], length * directions[..., 1],
                     angles="xy", scale_units="xy", scale=1, color=colour, width=width, zorder=4)


def canvas(centre: tuple[float, float], reach: float) -> tuple[plt.Figure, plt.Axes]:
    """A square view of half width `reach` about `centre`."""
    figure, ax = plt.subplots(figsize=(FRAME_INCHES, FRAME_INCHES), dpi=DPI)
    ax.set(xlim=(centre[0] - reach, centre[0] + reach), ylim=(centre[1] - reach, centre[1] + reach), aspect="equal")
    ax.set_axis_off()
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return figure, ax


def disk_canvas(centre: tuple[float, float], reach: float) -> tuple[plt.Figure, plt.Axes]:
    """An empty Poincaré disk, in a square view of half width `reach` about `centre`: its rim is the
    speed of light."""
    figure, ax = canvas(centre, reach)
    ax.add_patch(plt.Circle((0, 0), 1, facecolor="none", edgecolor=FAINT, linewidth=1.5))
    return figure, ax


def animate_two_boosts(frames: Iterable[tuple[core.Antivector, core.Antivector, core.Antivector]]) -> list[np.ndarray]:
    """Each frame: the triangle of velocities `[samples]`, rest to the first boost to the second and
    straight back; its corners `[3]`, with the rocket's first axis at each as the observer at rest
    reads it `[3]`, and at the last corner that axis as the straight boost from rest would carry it."""
    images = []
    for loop, corners, axes in frames:
        figure, ax = disk_canvas(TRIANGLE_CENTRE, TRIANGLE_REACH)
        outline, tails = disk(loop), disk(corners)
        ax.fill(*outline.T, color=FILL, zorder=1)
        ax.plot(*outline.T, color=INK, linewidth=1.5, zorder=2)
        arrows(ax, tails[-1], np.array([1.0, 0.0]), DISK_ARROW, FAINT, 0.012)
        arrows(ax, tails, unit(axes), DISK_ARROW, SPIN, 0.012)
        ax.plot(*tails.T, "o", color=INK, markersize=4, zorder=5)
        images.append(capture(figure))
        plt.close(figure)
    return images


def animate_orbits(loops: core.Antivector, states: Iterable[tuple[core.Antivector, core.Bivector]]) -> list[np.ndarray]:
    """The circles of velocities `[steps + 1, orbits]`, and each frame's velocities and spins as the
    observer at rest reads them `[orbits]`, the first frame's spins kept faint at the start."""
    outlines = disk(loops)
    states = iter(states)
    first = next(states)
    starts, start_directions = disk(first[0]), spin_axes(first[1])
    images = []
    for velocities, spins in chain([first], states):
        figure, ax = disk_canvas((0, 0), orbit_reach)
        points, directions = disk(velocities), spin_axes(spins)
        for orbit, colour in enumerate(orbit_colours):
            ax.plot(*outlines[:, orbit].T, color=colour, linewidth=1.2, alpha=0.5, zorder=2)
            arrows(ax, starts[orbit], start_directions[orbit], ORBIT_ARROW, FAINT, 0.010)
            arrows(ax, points[orbit], directions[orbit], ORBIT_ARROW, colour, 0.011)
        ax.plot(*points.T, "o", color=INK, markersize=4, zorder=5)
        images.append(capture(figure))
        plt.close(figure)
    return images


def animate_box(half_width: float, walls: core.Antivector, states: Iterable[tuple[core.Antivector, core.Bivector]]
                ) -> list[np.ndarray]:
    """A square box, the diamond its particle bounces along between the midpoints of its `walls`
    `[4]`, and each frame's position and spin as the observer at rest reads it."""
    corners = half_width * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]])
    diamond = planar(walls)[[0, 1, 2, 3, 0]]
    images = []
    for position, spin in states:
        figure, ax = canvas((0, 0), 1.15 * half_width)
        ax.plot(*corners.T, color=INK, linewidth=2.0, zorder=2)
        ax.plot(*diamond.T, color=FAINT, linewidth=1.0, zorder=1)
        point = planar(position)
        arrows(ax, point, spin_axes(spin), SCENE_ARROW * half_width, SPIN, 0.012)
        ax.plot(*point, "o", color=INK, markersize=7, zorder=5)
        images.append(capture(figure))
        plt.close(figure)
    return images


def animate_ring(ring: core.Antivector, states: Iterable[tuple[core.Antivector, core.Antivector, core.Bivector]]) -> list[np.ndarray]:
    """The ring `[samples]`, and each frame's position, velocity and spin as the observer at rest reads
    it: the momentum's direction faint, the spin solid."""
    outline = planar(ring)
    radius = np.linalg.norm(outline, axis=-1).max()
    images = []
    for position, velocity, spin in states:
        figure, ax = canvas((0, 0), 1.3 * radius)
        ax.plot(*outline.T, color=FAINT, linewidth=2.0, zorder=1)
        point = planar(position)
        arrows(ax, point, unit(velocity), SCENE_ARROW * radius, line_colours[0], 0.010)
        arrows(ax, point, spin_axes(spin), SCENE_ARROW * radius, SPIN, 0.012)
        ax.plot(*point, "o", color=INK, markersize=7, zorder=5)
        images.append(capture(figure))
        plt.close(figure)
    return images


def draw_magic(electric: np.ndarray, ratios: core.Scalar) -> plt.Figure:
    """The anomalous precession rate over the anomaly times the magnetic field `[gammas, fields]`,
    against the electric field's strength."""
    figure, ax = plt.subplots(figsize=(6, 3.6), dpi=110, layout="constrained")
    for values, colour in zip(ratios.cast(core.ga.subspace.scalar()).kernel[..., 0], line_colours):
        ax.plot(electric, values, "o-", color=colour, linewidth=2, markersize=4)
    ax.axhline(1.0, color=FAINT, linewidth=1.0, zorder=0)
    ax.set(xlabel="electric field over magnetic field", ylabel="anomalous rate over anomaly × field")
    ax.spines[["top", "right"]].set_visible(False)
    return figure
