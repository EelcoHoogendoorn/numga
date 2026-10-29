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
def disk(velocities: core.Antivector) -> core.Antivector:
    """Velocities as points of the Poincaré disk: where the line from minus the lab's time through each
    velocity meets the lab's slice of space, a share `1 / (1 + (mv.t & velocity))` of the way."""
    return (velocities + core.time) / (1 + (core.mv.t & velocities)) - core.time   # [...] Antivector-


def unit(directions: core.Antivector) -> core.Antivector:
    """The part of directions and velocities in the lab's slice of space, at unit length: a spatial
    direction squares to a positive scalar."""
    spatial = directions - (core.mv.t & directions) * core.time                 # [...] Antivector-
    return spatial / spatial.squared().square_root()                            # [...] Antivector-


def unit_spins(spins: core.Spin) -> core.Spin:
    """The spins at unit size: a patch of a plane of space squares to a positive scalar."""
    return spins / spins.squared().square_root()                                # [...] Spin-


def planar(directions: core.Antivector) -> np.ndarray:
    """x and y of directions and positions, on the last axis: their pairings with the slices x and y."""
    return (stack((core.mv.x, core.mv.y)) & directions[..., None]).to_array()


def spin_planar(spins: core.Spin) -> np.ndarray:
    """The arrow of each spin in the plane, on the last axis: its pairings with the planes yz and zx, the
    planes its patch turns in when it points along x and along y."""
    return (stack((core.mv.yz, core.mv.zx)) & spins[..., None]).to_array()


def arrows(ax: plt.Axes, tails: np.ndarray, vectors: np.ndarray, colour: str, width: float):
    return ax.quiver(tails[..., 0], tails[..., 1], vectors[..., 0], vectors[..., 1],
                     angles="xy", scale_units="xy", scale=1, color=colour, width=width, zorder=4)


def aim(quiver, tails: np.ndarray, vectors: np.ndarray) -> None:
    """The arrows of `arrows` moved to other tails and arrow vectors."""
    quiver.set_offsets(np.reshape(tails, (-1, 2)))
    quiver.set_UVC(*np.reshape(vectors, (-1, 2)).T)


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
    # The straight boost from rest keeps the rocket's x axis along the lab's.
    straight_axis = planar(DISK_ARROW * core.mv.x.dual())
    outline, tails = planar(disk(loop)), planar(disk(corners))
    fill, = ax.fill(*outline.T, color=FILL, zorder=1)
    edge, = ax.plot(*outline.T, color=INK, linewidth=1.0, zorder=2)
    straight = arrows(ax, tails[-1], straight_axis, FAINT, 0.012)
    rocket_axes = arrows(ax, tails, planar(DISK_ARROW * unit(axes)), SPIN, 0.012)
    dots, = ax.plot(*tails.T, "o", color=INK, markersize=3, zorder=5)
    for loop, corners, axes in chain([(loop, corners, axes)], frames):
        outline, tails = planar(disk(loop)), planar(disk(corners))
        fill.set_xy(outline)
        edge.set_data(*outline.T)
        aim(straight, tails[-1], straight_axis)
        aim(rocket_axes, tails, planar(DISK_ARROW * unit(axes)))
        dots.set_data(*tails.T)
        yield capture(figure)
    plt.close(figure)


def animate_orbits(loops: core.Antivector, states: Iterable[tuple[core.Antivector, core.Spin]]) -> Iterator[np.ndarray]:
    """The circles of velocities `[steps + 1, orbits]`, and each frame's velocities and spins as the
    observer at rest reads them `[orbits]`, the first frame's spins kept faint at the start."""
    outlines = planar(disk(loops))
    states = iter(states)
    first = next(states)
    starts, start_spins = planar(disk(first[0])), spin_planar(ORBIT_ARROW * unit_spins(first[1]))
    # The disk, the orbits and the first frame's spins set up once; the spins and the dots moved for each frame.
    figure, ax = disk_canvas((0, 0), orbit_half_width)
    spin_arrows = []
    for orbit, colour in enumerate(orbit_colours):
        ax.plot(*outlines[:, orbit].T, color=colour, linewidth=0.8, alpha=0.5, zorder=2)
        arrows(ax, starts[orbit], start_spins[orbit], FAINT, 0.010)
        spin_arrows.append(arrows(ax, starts[orbit], start_spins[orbit], colour, 0.011))
    dots, = ax.plot(*starts.T, "o", color=INK, markersize=3, zorder=5)
    for velocities, spins in chain([first], states):
        points, spin_vectors = planar(disk(velocities)), spin_planar(ORBIT_ARROW * unit_spins(spins))
        for orbit, arrow in enumerate(spin_arrows):
            aim(arrow, points[orbit], spin_vectors[orbit])
        dots.set_data(*points.T)
        yield capture(figure)
    plt.close(figure)


def animate_box(half_width: float, walls: core.Antivector, states: Iterable[tuple[core.Antivector, core.Spin]]
                ) -> Iterator[np.ndarray]:
    """A square box, the diamond its particle bounces along between the midpoints of its `walls`
    `[4]`, and each frame's position and spin as the observer at rest reads it."""
    # The box's corners, a corner on the diagonal turned by quarter turns all the way round.
    quarters = (core.mv.xy * (-np.arange(5) * np.pi / 4)).exp()                 # [corners + 1] Rotor+
    corners = quarters >> (half_width * (core.mv.x.dual() + core.mv.y.dual()))  # [corners + 1] Antivector-
    diamond = planar(walls)[[0, 1, 2, 3, 0]]
    # The box and the diamond set up once; the spin and the particle moved for each frame.
    figure, ax = canvas((0, 0), 1.15 * half_width)
    ax.plot(*planar(corners).T, color=INK, linewidth=1.3, zorder=2)
    ax.plot(*diamond.T, color=FAINT, linewidth=0.7, zorder=1)
    spin_arrow = arrows(ax, np.zeros(2), np.zeros(2), SPIN, 0.012)
    particle, = ax.plot(0, 0, "o", color=INK, markersize=5, zorder=5)
    for position, spin in states:
        point = planar(position)
        aim(spin_arrow, point, spin_planar(SCENE_ARROW * half_width * unit_spins(spin)))
        particle.set_data(*point[:, None])
        yield capture(figure)
    plt.close(figure)


def animate_ring(ring: core.Antivector, states: Iterable[tuple[core.Antivector, core.Antivector, core.Spin]]) -> Iterator[np.ndarray]:
    """The ring `[samples]`, and each frame's position, velocity and spin as the observer at rest reads
    it: the momentum's direction faint, the spin solid."""
    # The ring's radius: its points lie in the lab's slice, where they square to their squared distance from the axis.
    radius = ring.squared().square_root().to_array().max()
    # The ring set up once; the momentum, the spin and the particle moved for each frame.
    figure, ax = canvas((0, 0), 1.3 * radius)
    ax.plot(*planar(ring).T, color=FAINT, linewidth=1.3, zorder=1)
    momentum_arrow = arrows(ax, np.zeros(2), np.zeros(2), line_colours[0], 0.010)
    spin_arrow = arrows(ax, np.zeros(2), np.zeros(2), SPIN, 0.012)
    particle, = ax.plot(0, 0, "o", color=INK, markersize=5, zorder=5)
    for position, velocity, spin in states:
        point = planar(position)
        aim(momentum_arrow, point, planar(SCENE_ARROW * radius * unit(velocity)))
        aim(spin_arrow, point, spin_planar(SCENE_ARROW * radius * unit_spins(spin)))
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
