"""Drawing for magnetic resonance: Bloch-ball trajectories, absorption and dispersion lines, and the
spin echo. The Bloch ball is drawn on the GPU, the rest by matplotlib."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.util.animation import capture
from examples.quantum.magnetic_resonance import core
from examples.util.wgsl import labels, show, solids, strokes

sm = solids.mv
COLOURS = ("#c0392b", "#7d3c98", "#2e86c1")
SPHERE_COLOUR, SPHERE_OPACITY = np.array([0.70, 0.76, 0.86]), 0.09
AXIS_COLOUR = np.array([0.45, 0.47, 0.52])
CIRCLE_COLOUR, CIRCLE_OPACITY = np.full(3, 0.75), 0.55
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the half width of the square view
# about the centre, and the pixels across it.
ELEVATION, AZIMUTH = np.radians(18.0), np.radians(40.0)
REACH, N_PIXELS = 1.3, 300
# The widths of the great circles, the paths and the arrows, the arrowheads' length and the dots' radius, in pixels,
# and the size of the axes' names in points; how far the axes' arrows and their names reach, in radii of the ball.
CIRCLE_WIDTH, PATH_WIDTH, ARROW_WIDTH, HEAD_LENGTH, HEAD_WIDTH, DOT_RADIUS = 1.0, 1.4, 1.0, 7.0, 4.0, 2.6
LABEL_SIZE = 8
AXIS_LENGTH, LABEL_DISTANCE = 1.18, 1.26
# The quantity plots' size in inches and their resolution.
LINES_INCHES, SIGNAL_INCHES, DPI = (6.4, 2.6), (4.5, 2.5), 90
# The time the echo's signal is followed for, in µs from the first pulse.
ECHO_SPAN = 8.0


def components(rho: core.State) -> np.ndarray:
    """The Bloch vector's x, y and z components: twice the pairings of the state's spin part, its antibivector part,
    with the planes yz, zx and xy, in the algebra of spacetime as in that of space."""
    mv = rho.context.multivector
    spin_planes = stack((mv.yz, mv.zx, mv.xy))                           # [3] Bivector+
    return 2 * (spin_planes & rho.select.antibivector()[..., None]).to_array()


def draw_nutation(states: Iterable[core.State], steady_states: core.State) -> plt.Figure:
    """Each spin's Bloch vector from equilibrium, nutating about the drive and spiralling into its
    steady state, marked with a dot; the states are consumed as they are yielded. The Bloch ball is
    translucent, with its equator and two meridians, in the frame of its axes, named at their tips."""
    paths, ends = components(stack(list(states))), components(steady_states)  # [times, detunings, 3], [detunings, 3]
    colours = np.array([to_rgb(colour) for colour in COLOURS[: paths.shape[1]]])
    view = solids.View(solids.aim(solids.origin, ELEVATION, AZIMUTH), np.full(2, REACH), N_PIXELS)
    tracks = strokes.polylines(list(zip(sm.point(np.moveaxis(paths, 1, 0)), colours)), PATH_WIDTH, view)
    shafts, heads = solids.frame(AXIS_LENGTH, ARROW_WIDTH / 2 * view.pixel, HEAD_LENGTH * view.pixel,
                                 HEAD_WIDTH / 2 * view.pixel)
    parts = [(shafts, AXIS_COLOUR), (heads, AXIS_COLOUR),
             (solids.ball(sm.point(ends), DOT_RADIUS * view.pixel)[:, None], colours)]
    # The ball, translucent, the first part after the opaque ones; its equator and two meridians inked where the
    # planes x, y and z cut it.
    ball = [(solids.ball(solids.origin, 1.0)[None, None], SPHERE_COLOUR, SPHERE_OPACITY)]
    great_circles = solids.Ink(len(parts), solids.half_space(solids.axes), solids.everywhere[None].broadcast_to((3,)),
                               CIRCLE_COLOUR, CIRCLE_WIDTH, CIRCLE_OPACITY)
    figure = plt.figure(figsize=(N_PIXELS / 100, N_PIXELS / 100), dpi=100)
    ax = figure.add_axes((0, 0, 1, 1))
    show(ax, solids.draw(parts, view, np.ones(3), [great_circles], translucent=ball, meshes=[(tracks, 1.0)]), view)
    # Each axis named just past its arrow's tip, where the camera sees it.
    labels(ax, view, sm.point(LABEL_DISTANCE * np.eye(3)), list("xyz"), ha="center", va="center", fontsize=LABEL_SIZE,
           color=AXIS_COLOUR)
    return figure


def draw_lines(detunings: np.ndarray, drives: np.ndarray, steady_states: core.State) -> plt.Figure:
    """Absorption and dispersion against detuning, one curve per drive strength, each divided by
    the drive so weak drives stay visible: a stronger drive broadens and flattens the line."""
    r = components(steady_states)                                        # [detunings, drives, 3]
    figure, (absorbing, dispersing) = plt.subplots(1, 2, figsize=LINES_INCHES, dpi=DPI)
    for index, (drive, colour) in enumerate(zip(drives, COLOURS)):
        absorbing.plot(detunings, -r[:, index, 1] / drive, color=colour, label=f"drive {drive:g} rad/µs")
        dispersing.plot(detunings, r[:, index, 0] / drive, color=colour)
    absorbing.set_ylabel("absorption, per unit drive")
    dispersing.set_ylabel("dispersion, per unit drive")
    # Each line against its zero, with resonance marked faintly.
    absorbing.set_ylim(bottom=0)
    for ax in (absorbing, dispersing):
        ax.axhline(0, color="black", linewidth=0.8, zorder=0)
        ax.axvline(0, color="0.85", linewidth=0.7, zorder=0)
        ax.set(xticks=[], yticks=[], xlabel="detuning")
        ax.spines[["top", "right"]].set_visible(False)
    absorbing.legend(fontsize=7, frameon=False)
    figure.tight_layout()
    return figure


def signal(ensemble: core.State) -> np.ndarray:
    """The ensemble's mean transverse Bloch vector, its length at each time: what a pick-up coil sees."""
    r = components(ensemble)                                             # [times, spins, 3]
    return np.hypot(r[..., 0].mean(axis=-1), r[..., 1].mean(axis=-1))


def draw_echo(states: Iterable[core.State], dt: float, delay: float, t2: float) -> plt.Figure:
    """The signal after the first pulse, for an ensemble recorded every dt: it vanishes as the spins
    fan out, and returns at twice the delay after the second pulse, reduced only by the true
    dephasing."""
    ensemble = stack(list(states))                                       # [times, spins] State-
    times = np.arange(ensemble.shape[0]) * dt
    figure, ax = plt.subplots(figsize=SIGNAL_INCHES, dpi=DPI)
    ax.plot(times, signal(ensemble), color="#2e86c1", label="mean transverse spin")
    ax.plot(times, np.exp(-times / t2), "--", color="0.5", label="exp(-t / T2)")
    for when, name in ((delay, "π pulse"), (2 * delay, "echo")):
        ax.axvline(when, color="0.7", linewidth=0.8)
        ax.annotate(name, (when, 0.95), textcoords="offset points", xytext=(4, 0), fontsize=8)
    ax.set(xlabel="time", ylabel="mean transverse spin", xlim=(times[0], times[-1]), ylim=(0, 1.05), xticks=[],
           yticks=[])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="center right", fontsize=8, frameon=False)
    figure.tight_layout()
    return figure


def transverse(states: core.State) -> np.ndarray:
    """The length of each state's transverse Bloch vector: what a pick-up coil sees of it."""
    r = components(states)
    return np.hypot(r[..., 0], r[..., 1])


def draw_decays(times: np.ndarray, echoed: core.State, faded: core.State, t2: float) -> plt.Figure:
    """The signal at each time after the first pulse, on a logarithmic time axis: with a half-turn
    pulse halfway, the echo, which fades only at the rate `1 / t2`; without it, the free decay, which
    the uneven field ends within about a microsecond."""
    fine = np.geomspace(times[0], times[-1], 200)
    figure, ax = plt.subplots(figsize=SIGNAL_INCHES, dpi=DPI)
    ax.semilogx(times, transverse(echoed), "o-", color="#2e86c1", label="echo")
    ax.semilogx(times, transverse(faded), "o-", color="#c0392b", label="free decay")
    ax.semilogx(fine, np.exp(-fine / t2), "--", color="0.5", label="exp(-t / T2)")
    ax.set(xlabel="time after the first pulse (µs)", ylabel="transverse spin", ylim=(0, 1.05), yticks=[])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    figure.tight_layout()
    return figure


def animate_echo(detunings: np.ndarray, states: Iterable[core.State], dt: float, delay: float, t2: float,
                 every: int) -> Iterator[np.ndarray]:
    """The spins seen from above the field, fanning out and refocusing, beside the signal so far, over ECHO_SPAN;
    one frame for every few recorded states, each drawn as its state comes."""
    order = np.argsort(detunings)
    span = np.linspace(0.0, ECHO_SPAN)
    figure, (top, trace) = plt.subplots(1, 2, figsize=(9, 4.2), dpi=67, gridspec_kw={"width_ratios": (1, 1.4)})
    top.add_patch(plt.Circle((0, 0), 1.0, fill=False, color="0.8"))
    spins = top.scatter(np.zeros(len(order)), np.zeros(len(order)), c=detunings[order], cmap="coolwarm", s=6)
    top.set_xlim(-1.1, 1.1)
    top.set_ylim(-1.1, 1.1)
    top.set_aspect("equal")
    top.set_axis_off()
    so_far, = trace.plot([], [], color="#2e86c1")
    trace.plot(span, np.exp(-span / t2), "--", color="0.8")
    trace.axvline(delay, color="0.85", linewidth=0.8)
    trace.annotate("π pulse", (delay, 0.95), textcoords="offset points", xytext=(4, 0))
    trace.set_xlim(0.0, ECHO_SPAN)
    trace.set(ylim=(0, 1.05), xticks=[], yticks=[], xlabel="time", ylabel="mean transverse spin")
    trace.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    times, strength = [], []
    for index, rho in enumerate(states):
        times.append(index * dt)
        strength.append(signal(rho))
        if index % every == 0:
            spins.set_offsets(components(rho)[order, :2])
            so_far.set_data(times, strength)
            yield capture(figure)
    plt.close(figure)
