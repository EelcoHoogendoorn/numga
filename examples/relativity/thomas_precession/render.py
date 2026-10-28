"""Drawing for Thomas precession: velocities in the Poincaré disk, a particle bouncing in a box, a
storage ring, and the anomalous precession rate against the electric field."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

from numga import stack

from examples.util import charts
from examples.util.animation import capture
from examples.relativity.thomas_precession import core

INK, FAINT, FILL, SPIN = "#1e293b", "#cbd5e1", "#dbeafe", "#ea580c"
orbit_colours = ("#0f766e", "#2563eb", "#7c3aed")
line_colours = ("#64748b", "#ea580c")
# Arrow lengths: in the Poincaré disk, around the orbits, and as a share of the box's or ring's half
# width; the view's half width around the orbits.
DISK_ARROW, ORBIT_ARROW, SCENE_ARROW = 0.16, 0.09, 0.3
orbit_half_width = 0.66
# The window of the disk on the rocket's triangle at its largest: the centre, and its half width.
TRIANGLE_CENTRE, TRIANGLE_HALF_WIDTH = (0.5, 0.2), 0.62
FRAME_INCHES, DPI = 10 / 3, 90


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


def spin_axes(spins: core.Spin) -> np.ndarray:
    """The arrow of each spin, in the plane, on the last axis: its pairings with the planes yz and zx."""
    planes = stack((core.mv.yz, core.mv.zx))                                    # [2] Bivector+
    xy = (planes & spins[..., None]).to_array()
    return xy / np.linalg.norm(xy, axis=-1, keepdims=True)


def arrows(ax: plt.Axes, tails: np.ndarray, directions: np.ndarray, length: float, colour: str, width: float):
    return ax.quiver(tails[..., 0], tails[..., 1], length * directions[..., 0], length * directions[..., 1],
                     angles="xy", scale_units="xy", scale=1, color=colour, width=width, zorder=4)


def aim(quiver, tails: np.ndarray, directions: np.ndarray, length: float) -> None:
    """The arrows of `arrows` moved to other tails and directions."""
    quiver.set_offsets(np.reshape(tails, (-1, 2)))
    quiver.set_UVC(*(length * np.reshape(directions, (-1, 2))).T)


def canvas(centre: tuple[float, float], half_width: float) -> tuple[plt.Figure, plt.Axes]:
    """A square view of half width `half_width` about `centre`."""
    figure, ax = plt.subplots(figsize=(FRAME_INCHES, FRAME_INCHES), dpi=DPI)
    ax.set(xlim=(centre[0] - half_width, centre[0] + half_width), ylim=(centre[1] - half_width, centre[1] + half_width),
           aspect="equal")
    ax.set_axis_off()
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return figure, ax


def disk_canvas(centre: tuple[float, float], half_width: float) -> tuple[plt.Figure, plt.Axes]:
    """An empty Poincaré disk, in a square view of half width `half_width` about `centre`: its rim is the
    speed of light."""
    figure, ax = canvas(centre, half_width)
    ax.add_patch(plt.Circle((0, 0), 1, facecolor="none", edgecolor=FAINT, linewidth=1.0))
    return figure, ax


def animate_two_boosts(frames: Iterable[tuple[core.Antivector, core.Antivector, core.Antivector]]) -> Iterator[np.ndarray]:
    """Each frame: the triangle of velocities `[samples]`, rest to the first boost to the second and
    straight back; its corners `[3]`, with the rocket's first axis at each as the observer at rest
    reads it `[3]`, and at the last corner that axis as the straight boost from rest would carry it."""
    frames = iter(frames)
    loop, corners, axes = next(frames)
    # The disk set up once, the triangle and its arrows drawn for the first frame and moved for each.
    figure, ax = disk_canvas(TRIANGLE_CENTRE, TRIANGLE_HALF_WIDTH)
    outline, tails = disk(loop), disk(corners)
    fill, = ax.fill(*outline.T, color=FILL, zorder=1)
    edge, = ax.plot(*outline.T, color=INK, linewidth=1.0, zorder=2)
    straight = arrows(ax, tails[-1], np.array([1.0, 0.0]), DISK_ARROW, FAINT, 0.012)
    rocket_axes = arrows(ax, tails, unit(axes), DISK_ARROW, SPIN, 0.012)
    dots, = ax.plot(*tails.T, "o", color=INK, markersize=3, zorder=5)
    for loop, corners, axes in chain([(loop, corners, axes)], frames):
        outline, tails = disk(loop), disk(corners)
        fill.set_xy(outline)
        edge.set_data(*outline.T)
        aim(straight, tails[-1], np.array([1.0, 0.0]), DISK_ARROW)
        aim(rocket_axes, tails, unit(axes), DISK_ARROW)
        dots.set_data(*tails.T)
        yield capture(figure)
    plt.close(figure)


def animate_orbits(loops: core.Antivector, states: Iterable[tuple[core.Antivector, core.Spin]]) -> Iterator[np.ndarray]:
    """The circles of velocities `[steps + 1, orbits]`, and each frame's velocities and spins as the
    observer at rest reads them `[orbits]`, the first frame's spins kept faint at the start."""
    outlines = disk(loops)
    states = iter(states)
    first = next(states)
    starts, start_directions = disk(first[0]), spin_axes(first[1])
    # The disk, the orbits and the first frame's spins set up once; the spins and the dots moved for each frame.
    figure, ax = disk_canvas((0, 0), orbit_half_width)
    spin_arrows = []
    for orbit, colour in enumerate(orbit_colours):
        ax.plot(*outlines[:, orbit].T, color=colour, linewidth=0.8, alpha=0.5, zorder=2)
        arrows(ax, starts[orbit], start_directions[orbit], ORBIT_ARROW, FAINT, 0.010)
        spin_arrows.append(arrows(ax, starts[orbit], start_directions[orbit], ORBIT_ARROW, colour, 0.011))
    dots, = ax.plot(*starts.T, "o", color=INK, markersize=3, zorder=5)
    for velocities, spins in chain([first], states):
        points, directions = disk(velocities), spin_axes(spins)
        for orbit, arrow in enumerate(spin_arrows):
            aim(arrow, points[orbit], directions[orbit], ORBIT_ARROW)
        dots.set_data(*points.T)
        yield capture(figure)
    plt.close(figure)


def animate_box(half_width: float, walls: core.Antivector, states: Iterable[tuple[core.Antivector, core.Spin]]
                ) -> Iterator[np.ndarray]:
    """A square box, the diamond its particle bounces along between the midpoints of its `walls`
    `[4]`, and each frame's position and spin as the observer at rest reads it."""
    corners = half_width * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]])
    diamond = planar(walls)[[0, 1, 2, 3, 0]]
    # The box and the diamond set up once; the spin and the particle moved for each frame.
    figure, ax = canvas((0, 0), 1.15 * half_width)
    ax.plot(*corners.T, color=INK, linewidth=1.3, zorder=2)
    ax.plot(*diamond.T, color=FAINT, linewidth=0.7, zorder=1)
    spin_arrow = arrows(ax, np.zeros(2), np.zeros(2), SCENE_ARROW * half_width, SPIN, 0.012)
    particle, = ax.plot(0, 0, "o", color=INK, markersize=5, zorder=5)
    for position, spin in states:
        point = planar(position)
        aim(spin_arrow, point, spin_axes(spin), SCENE_ARROW * half_width)
        particle.set_data(*point[:, None])
        yield capture(figure)
    plt.close(figure)


def animate_ring(ring: core.Antivector, states: Iterable[tuple[core.Antivector, core.Antivector, core.Spin]]) -> Iterator[np.ndarray]:
    """The ring `[samples]`, and each frame's position, velocity and spin as the observer at rest reads
    it: the momentum's direction faint, the spin solid."""
    outline = planar(ring)
    radius = np.linalg.norm(outline, axis=-1).max()
    # The ring set up once; the momentum, the spin and the particle moved for each frame.
    figure, ax = canvas((0, 0), 1.3 * radius)
    ax.plot(*outline.T, color=FAINT, linewidth=1.3, zorder=1)
    momentum_arrow = arrows(ax, np.zeros(2), np.zeros(2), SCENE_ARROW * radius, line_colours[0], 0.010)
    spin_arrow = arrows(ax, np.zeros(2), np.zeros(2), SCENE_ARROW * radius, SPIN, 0.012)
    particle, = ax.plot(0, 0, "o", color=INK, markersize=5, zorder=5)
    for position, velocity, spin in states:
        point = planar(position)
        aim(momentum_arrow, point, unit(velocity), SCENE_ARROW * radius)
        aim(spin_arrow, point, spin_axes(spin), SCENE_ARROW * radius)
        particle.set_data(*point[:, None])
        yield capture(figure)
    plt.close(figure)


@charts.style()
def draw_magic(electric: np.ndarray, ratios: core.Scalar) -> plt.Figure:
    """The anomalous precession rate over the anomaly times the magnetic field `[gammas, fields]`,
    against the electric field's strength."""
    # Half the column: two lines of dots, and the height the long vertical label needs.
    figure, ax = charts.subplots(420, 300, layout="constrained")
    for values, colour in zip(ratios.to_array(), line_colours):
        ax.plot(electric, values, "o-", color=colour)
    ax.axhline(1.0, color=FAINT, linewidth=charts.FRAME_WIDTH, zorder=0)
    # The rate from zero, and one, where the magic Lorentz factor holds it, the only value marked.
    ax.set(ylim=(0, None), xticks=[], yticks=[1.0], xlabel="electric field over magnetic field",
           ylabel="anomalous rate over anomaly × field")
    ax.spines[["top", "right"]].set_visible(False)
    return figure
