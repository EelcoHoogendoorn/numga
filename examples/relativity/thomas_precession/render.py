"""Drawing for Thomas precession: velocities in the Poincaré disk, a particle bouncing in a box, a
storage ring, and the anomalous precession rate against the electric field."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack

from examples.util import charts
from examples.util.animation import capture
from examples.util.wgsl import contours, simplices
from examples.util.wgsl.camera import Camera, flat_panel
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
# In pixels: the outlines' widths, thin, regular and bold; the arrows' widths and the radius of the dots at their tips;
# the dots' radii, small and for a particle. The samples round the disk's rim.
THIN, LINE, BOLD = 1.0, 1.25, 1.6
ARROW_WIDTH, ARROW_TIP = 2.5, 5.0
DOT, PARTICLE = 1.9, 3.1
N_RIM_SAMPLES = 200


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


def spin_arrows(spins: core.Spin) -> core.Antivector:
    """The arrow of each spin in the plane: its pairings with the planes yz and zx, the planes its patch turns in when
    it points along x and along y, as its steps along x and along y."""
    mv = core.mv
    steps = stack((mv.x.dual(), mv.y.dual()))                                  # [2] Antivector-
    return ((stack((mv.yz, mv.zx)) & spins[..., None]) * steps).sum(axis=-1)   # [...] Antivector-


def ink(colour: str) -> np.ndarray:
    """A colour's red, green and blue."""
    return np.array(to_rgb(colour))


def arrows(tails: core.Antivector, steps: core.Antivector, colours: np.ndarray, camera: Camera) -> list[np.ndarray]:
    """Arrows from the tails by the steps `[arrows]`, each in its colour, a stroke and a dot at its tip: their
    pictures."""
    tips = tails + steps                                                      # [arrows] Antivector-
    ends = stack((tails, tips), axis=-1).reshape(-1)                          # [arrows * 2] Antivector-
    n_arrows = tails.shape[0]
    return [simplices.strokes(ends, simplices.Mesh.chains([2] * n_arrows), colours, ARROW_WIDTH, camera,
                              cap=simplices.Cap.BUTT),
            simplices.markers(tips, colours, ARROW_TIP, camera)]


def dots(points: core.Antivector, radius: float, camera: Camera) -> np.ndarray:
    """Dots in ink at points `[points]`: their picture."""
    return simplices.markers(points, ink(INK), radius, camera)


def outline(points: core.Antivector, colour: str, width: float, camera: Camera, opacity: float = 1.0) -> np.ndarray:
    """The polyline through points `[points]`: its picture."""
    return simplices.strokes(points, simplices.Mesh.chains([points.shape[0]]), ink(colour), width, camera,
                             opacity=opacity)


def canvas(centre: tuple[float, float], half_width: float) -> tuple[plt.Figure, plt.Axes, Camera]:
    """A square view of half width `half_width` about `centre`, and its camera, looking down on x and y."""
    figure, ax = plt.subplots(figsize=(FRAME_INCHES, FRAME_INCHES), dpi=DPI)
    ax.set(xlim=(centre[0] - half_width, centre[0] + half_width), ylim=(centre[1] - half_width, centre[1] + half_width),
           aspect="equal")
    ax.set_axis_off()
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return figure, ax, flat_panel(ax, core.mv.x.algebra)


def rim(camera: Camera) -> np.ndarray:
    """The rim of the Poincaré disk, the speed of light: its picture."""
    turns = (core.mv.xy * (-np.linspace(0, 2 * np.pi, N_RIM_SAMPLES) / 2)).exp()
    return outline(turns >> core.mv.x.dual(), FAINT, LINE, camera)


def animate_two_boosts(frames: Iterable[tuple[core.Antivector, core.Antivector, core.Antivector]]) -> Iterator[np.ndarray]:
    """Each frame: the triangle of velocities `[samples]`, rest to the first boost to the second and
    straight back; its corners `[3]`, with the rocket's first axis at each as the observer at rest
    reads it `[3]`, and at the last corner that axis as the straight boost from rest would carry it."""
    # The disk set up once, the triangle and its arrows drawn for each frame.
    figure, ax, camera = canvas(TRIANGLE_CENTRE, TRIANGLE_HALF_WIDTH)
    edge = rim(camera)
    shown = simplices.overlay(ax, [edge])
    # The straight boost from rest keeps the rocket's x axis along the lab's.
    straight_axis = DISK_ARROW * core.mv.x.dual()                             # [] Antivector-
    for loop, corners, axes in frames:
        outline_points, tails = disk(loop), disk(corners)                      # [samples] Antivector-, [3] Antivector-
        # The triangle filled as a fan from rest, its first point, which its two straight sides leave from.
        n_samples = outline_points.shape[0]
        fan = contours.Mesh(n_samples, np.stack((np.zeros(n_samples - 2, dtype=int), np.arange(1, n_samples - 1),
                                                 np.arange(2, n_samples)), axis=-1))
        shown.set_data(simplices.composited([
            edge,
            contours.picture(outline_points, fan, np.zeros(n_samples), ink(FILL), camera),
            outline(outline_points, INK, LINE, camera),
            *arrows(tails[-1:], straight_axis[None], ink(FAINT), camera),
            *arrows(tails, DISK_ARROW * unit(axes), ink(SPIN), camera),
            dots(tails, DOT, camera)]))
        yield capture(figure)
    plt.close(figure)


def animate_orbits(loops: core.Antivector, states: Iterable[tuple[core.Antivector, core.Spin]]) -> Iterator[np.ndarray]:
    """The circles of velocities `[steps + 1, orbits]`, and each frame's velocities and spins as the
    observer at rest reads them `[orbits]`, the first frame's spins kept faint at the start."""
    outlines = disk(loops)                                                     # [steps + 1, orbits] Antivector-
    n_samples, n_orbits = outlines.shape
    colours = np.stack([ink(colour) for colour in orbit_colours])
    states = iter(states)
    first = next(states)
    # The disk, the orbits and the first frame's spins set up once; the spins and the dots drawn for each frame.
    figure, ax, camera = canvas((0, 0), orbit_half_width)
    by_orbit = outlines[np.arange(n_samples)[None, :], np.arange(n_orbits)[:, None]].reshape(-1)   # [orbits * samples] Antivector-
    beneath = [rim(camera),
               simplices.strokes(by_orbit, simplices.Mesh.chains([n_samples] * n_orbits),
                                 np.repeat(colours, n_samples - 1, axis=0), THIN, camera, opacity=0.5),
               *arrows(disk(first[0]), spin_arrows(ORBIT_ARROW * unit_spins(first[1])), ink(FAINT), camera)]
    shown = simplices.overlay(ax, beneath)
    for velocities, spins in chain([first], states):
        points = disk(velocities)                                              # [orbits] Antivector-
        shown.set_data(simplices.composited(beneath + [
            *arrows(points, spin_arrows(ORBIT_ARROW * unit_spins(spins)), colours, camera), dots(points, DOT, camera)]))
        yield capture(figure)
    plt.close(figure)


def animate_box(half_width: float, walls: core.Antivector, states: Iterable[tuple[core.Antivector, core.Spin]]
                ) -> Iterator[np.ndarray]:
    """A square box, the diamond its particle bounces along between the midpoints of its `walls`
    `[4]`, and each frame's position and spin as the observer at rest reads it."""
    # The box's corners, a corner on the diagonal turned by quarter turns all the way round.
    quarters = (core.mv.xy * (-np.arange(5) * np.pi / 4)).exp()                 # [corners + 1] Rotor+
    corners = quarters >> (half_width * (core.mv.x.dual() + core.mv.y.dual()))  # [corners + 1] Antivector-
    # The box and the diamond set up once; the spin and the particle drawn for each frame.
    figure, ax, camera = canvas((0, 0), 1.15 * half_width)
    beneath = [outline(walls[[0, 1, 2, 3, 0]], FAINT, THIN, camera), outline(corners, INK, BOLD, camera)]
    shown = simplices.overlay(ax, beneath)
    for position, spin in states:
        shown.set_data(simplices.composited(beneath + [
            *arrows(position[None], spin_arrows(SCENE_ARROW * half_width * unit_spins(spin))[None], ink(SPIN), camera),
            dots(position[None], PARTICLE, camera)]))
        yield capture(figure)
    plt.close(figure)


def animate_ring(ring: core.Antivector, states: Iterable[tuple[core.Antivector, core.Antivector, core.Spin]]) -> Iterator[np.ndarray]:
    """The ring `[samples]`, and each frame's position, velocity and spin as the observer at rest reads
    it: the momentum's direction faint, the spin solid."""
    # The ring's radius: its points lie in the lab's slice, where they square to their squared distance from the axis.
    radii = ring.squared().square_root()
    radius = radii.max()
    # The ring set up once; the momentum, the spin and the particle drawn for each frame.
    figure, ax, camera = canvas((0, 0), float((1.3 * radius).to_array()))
    beneath = [outline(ring, FAINT, BOLD, camera)]
    shown = simplices.overlay(ax, beneath)
    for position, velocity, spin in states:
        shown.set_data(simplices.composited(beneath + [
            *arrows(position[None], (SCENE_ARROW * radius * unit(velocity))[None], ink(line_colours[0]), camera),
            *arrows(position[None], spin_arrows(SCENE_ARROW * radius * unit_spins(spin))[None], ink(SPIN), camera),
            dots(position[None], PARTICLE, camera)]))
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
