"""Poincaré-disc projections, storage-ring animations, and precession-rate charts visualize relativistic spin transport."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from numga.render import Camera, contours, drawing, simplices
from numga.topology import Topology

from examples.relativity.thomas_precession import core
from examples.util import charts
from examples.util.animation import capture
from examples.util.panels import offset_panel, overlay

ink_colour = "#1e293b"
faint_colour = "#cbd5e1"
fill_colour = "#dbeafe"
spin_colour = "#ea580c"
orbit_colours = ("#0f766e", "#2563eb", "#7c3aed")
line_colours = ("#64748b", "#ea580c")

disk_arrow = 0.16
orbit_arrow = 0.09
scene_arrow = 0.3
orbit_half_width = 0.66
triangle_centre = (0.5, 0.2)
triangle_half_width = 0.62
frame_inches = 10 / 3
frame_dpi = 90

thin_width = 1.0
line_width = 1.25
bold_width = 1.6
arrow_width = 2.5
chevron_length = 10.0
dot_radius = 1.9
particle_radius = 3.1
n_rim_samples = 199


# --- plumbing -------------------------------------------------------------------------
def disk_points(velocities: core.Antivector) -> core.Antivector:
    """Stereographic projection from `-time` maps unit four-velocities (`[...] Antivector-`) onto the Poincaré disc in the laboratory spatial slice."""
    return (velocities + core.time) / (1 + (core.mv.t & velocities)) - core.time   # [...] Antivector-


def unit_directions(directions: core.Antivector) -> core.Antivector:
    """Projecting into the laboratory spatial slice and dividing by the intrinsic norm normalizes spatial directions (`[...] Antivector-`)."""
    spatial = directions - (core.mv.t & directions) * core.time                    # [...] Antivector-
    return spatial / spatial.antinorm()                                            # [...] Antivector-


def unit_spins(spins: core.Spin) -> core.Spin:
    """Dividing by the intrinsic norm normalizes spatial spin patches (`[...] Spin-`)."""
    return spins / spins.antinorm()                                                # [...] Spin-


def spin_arrows(spins: core.Spin) -> core.Antivector:
    """Pairing `spins` (`[...] Spin-`) with the vertical coordinate planes reads out their horizontal rotation-axis vectors (`[...] Antivector-`)."""
    mv = core.mv
    steps = stack((mv.x.dual(), mv.y.dual()))                                      # [2] Antivector-
    return ((stack((mv.yz, mv.zx)) & spins[..., None]) * steps).sum(axis=-1)       # [...] Antivector-


def rgb_colour(colour: str) -> np.ndarray:
    """Converts a hex colour string into a three-component RGB float array."""
    return np.array(to_rgb(colour))


def arrow_layer(tails: core.Antivector, steps: core.Antivector, colours: np.ndarray, camera: Camera) -> np.ndarray:
    """Renders `n_arrows` chevron-tipped strokes from `tails` along `steps` (`[n_arrows] Antivector-`)."""
    n_arrows = tails.shape[0]
    tips = tails + steps                                                           # [n_arrows] Antivector-
    ends = stack((tails, tips), axis=-1).reshape(-1)                               # [n_arrows * 2] Antivector-
    return simplices.strokes(
        ends,
        Topology.chains([2] * n_arrows),
        colours,
        arrow_width,
        camera,
        chevrons=np.ones(n_arrows),
        chevron=chevron_length,
        cap=simplices.Cap.BUTT,
    )


def dot_layer(points: core.Antivector, radius: float, camera: Camera) -> np.ndarray:
    """Renders circular markers of `radius` at `points` (`[...] Antivector-`)."""
    return simplices.markers(points, rgb_colour(ink_colour), radius, camera)


def loop_layer(points: core.Antivector, colour: str, width: float, camera: Camera) -> np.ndarray:
    """Renders a closed polyline through `points` (`[...] Antivector-`) in `colour`."""
    return simplices.strokes(points, Topology.loops([points.shape[0]]), rgb_colour(colour), width, camera)


def planar_canvas(centre: tuple[float, float], half_width: float) -> tuple[plt.Figure, plt.Axes, Camera]:
    """Creates a borderless square figure, axes, and planar camera of `half_width` around `centre`."""
    figure, ax = plt.subplots(figsize=(frame_inches, frame_inches), dpi=frame_dpi)
    ax.set(
        xlim=(centre[0] - half_width, centre[0] + half_width),
        ylim=(centre[1] - half_width, centre[1] + half_width),
        aspect="equal",
    )
    ax.set_axis_off()
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return figure, ax, offset_panel(ax, core.mv.x, core.mv.y)


def disk_rim(camera: Camera) -> np.ndarray:
    """Renders the light-speed boundary circle of the Poincaré disc."""
    turns = (core.mv.xy * (-np.linspace(0, 2 * np.pi, n_rim_samples, endpoint=False) / 2)).exp()   # [n_rim_samples] Rotor+
    rim_points = turns >> core.mv.x.dual()                                         # [n_rim_samples] Antivector-
    return loop_layer(rim_points, faint_colour, line_width, camera)


def two_boosts_animation(
    states: Iterable[tuple[core.Antivector, core.Antivector, core.Antivector]],
) -> Iterator[np.ndarray]:
    """Yields RGB frames showing the hyperbolic velocity triangle, its corners, and the Wigner-rotated rocket axis."""
    figure, ax, camera = planar_canvas(triangle_centre, triangle_half_width)
    rim_picture = disk_rim(camera)
    shown = overlay(ax, [rim_picture])
    straight_axis = disk_arrow * core.mv.x.dual()                                  # [] Antivector-
    for loop, corners, axes in states:
        n_samples = loop.shape[0]
        outline_points = disk_points(loop)                                         # [n_samples] Antivector-
        tails = disk_points(corners)                                               # [3] Antivector-
        fan = Topology(
            n_samples,
            np.stack(
                (
                    np.zeros(n_samples - 2, dtype=int),
                    np.arange(1, n_samples - 1),
                    np.arange(2, n_samples),
                ),
                axis=-1,
            ),
        )
        shown.set_data(drawing.composited([
            rim_picture,
            contours.picture(outline_points, fan, np.zeros(n_samples), rgb_colour(fill_colour), camera),
            loop_layer(outline_points, ink_colour, line_width, camera),
            arrow_layer(tails[-1:], straight_axis[None], rgb_colour(faint_colour), camera),
            arrow_layer(tails, disk_arrow * unit_directions(axes), rgb_colour(spin_colour), camera),
            dot_layer(tails, dot_radius, camera),
        ]))
        yield capture(figure)
    plt.close(figure)


def orbits_animation(
    loops: core.Antivector,
    states: Iterable[tuple[core.Antivector, core.Spin]],
) -> Iterator[np.ndarray]:
    """Yields RGB frames of circular velocity orbits (`[n_samples, n_orbits] Antivector-`) and their precessing laboratory spins."""
    n_samples, n_orbits = loops.shape
    outlines = disk_points(loops)                                                  # [n_samples, n_orbits] Antivector-
    colours = np.stack([rgb_colour(colour) for colour in orbit_colours])
    state_stream = iter(states)
    initial_state = next(state_stream)
    figure, ax, camera = planar_canvas((0, 0), orbit_half_width)
    by_orbit = outlines[np.arange(n_samples)[None, :], np.arange(n_orbits)[:, None]].reshape(-1)   # [n_orbits * n_samples] Antivector-
    beneath = [
        disk_rim(camera),
        simplices.strokes(
            by_orbit,
            Topology.chains([n_samples] * n_orbits),
            np.repeat(colours, n_samples - 1, axis=0),
            thin_width,
            camera,
            opacity=0.5,
        ),
        arrow_layer(
            disk_points(initial_state[0]),
            spin_arrows(orbit_arrow * unit_spins(initial_state[1])),
            rgb_colour(faint_colour),
            camera,
        ),
    ]
    shown = overlay(ax, beneath)
    for velocities, spins in chain([initial_state], state_stream):
        points = disk_points(velocities)                                           # [n_orbits] Antivector-
        shown.set_data(drawing.composited(beneath + [
            arrow_layer(points, spin_arrows(orbit_arrow * unit_spins(spins)), colours, camera),
            dot_layer(points, dot_radius, camera),
        ]))
        yield capture(figure)
    plt.close(figure)


def box_animation(
    half_width: float,
    wall_midpoints: core.Antivector,
    states: Iterable[tuple[core.Antivector, core.Spin]],
) -> Iterator[np.ndarray]:
    """Yields RGB frames of a particle bouncing between `wall_midpoints` (`[4] Antivector-`) inside a square box with its precessing spin."""
    quarter_turns = (core.mv.xy * (-np.arange(4) * np.pi / 4)).exp()               # [4] Rotor+
    corners = quarter_turns >> (half_width * (core.mv.x.dual() + core.mv.y.dual()))   # [4] Antivector-
    square = Topology.loops([4])
    figure, ax, camera = planar_canvas((0, 0), 1.15 * half_width)
    beneath = [
        simplices.strokes(wall_midpoints, square, rgb_colour(faint_colour), thin_width, camera),
        simplices.strokes(corners, square, rgb_colour(ink_colour), bold_width, camera),
    ]
    shown = overlay(ax, beneath)
    for position, spin in states:
        shown.set_data(drawing.composited(beneath + [
            arrow_layer(
                position[None],
                spin_arrows(scene_arrow * half_width * unit_spins(spin))[None],
                rgb_colour(spin_colour),
                camera,
            ),
            dot_layer(position[None], particle_radius, camera),
        ]))
        yield capture(figure)
    plt.close(figure)


def ring_animation(
    ring_points: core.Antivector,
    states: Iterable[tuple[core.Antivector, core.Antivector, core.Spin]],
) -> Iterator[np.ndarray]:
    """Yields RGB frames of a muon circulating around `ring_points` (`[...] Antivector-`) with momentum and spin arrows."""
    radii = ring_points.antinorm()                                                 # [...] Scalar+
    radius = radii.max()                                                           # [] Scalar+
    figure, ax, camera = planar_canvas((0, 0), float((1.3 * radius).to_array()))
    beneath = [loop_layer(ring_points, faint_colour, bold_width, camera)]
    shown = overlay(ax, beneath)
    for position, velocity, spin in states:
        shown.set_data(drawing.composited(beneath + [
            arrow_layer(
                position[None],
                (scene_arrow * radius * unit_directions(velocity))[None],
                rgb_colour(line_colours[0]),
                camera,
            ),
            arrow_layer(
                position[None],
                spin_arrows(scene_arrow * radius * unit_spins(spin))[None],
                rgb_colour(spin_colour),
                camera,
            ),
            dot_layer(position[None], particle_radius, camera),
        ]))
        yield capture(figure)
    plt.close(figure)


@charts.style()
def magic_momentum_figure(electric: np.ndarray, rate_ratios: core.Scalar) -> plt.Figure:
    """Plots normalized anomalous precession `rate_ratios` (`[n_gammas, n_fields] Scalar+`) against the electric-to-magnetic field ratio."""
    figure, ax = charts.subplots(420, 300, layout="constrained")
    for values, colour in zip(rate_ratios.to_array(), line_colours):
        ax.plot(electric, values, "o-", color=colour)
    ax.axhline(1.0, color=faint_colour, linewidth=charts.FRAME_WIDTH, zorder=0)
    ax.set(
        ylim=(0, None),
        xticks=[],
        yticks=[1.0],
        xlabel="electric field over magnetic field",
        ylabel="anomalous rate over anomaly × field",
    )
    ax.spines[["top", "right"]].set_visible(False)
    return figure
