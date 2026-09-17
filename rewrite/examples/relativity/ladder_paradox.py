"""A ladder can fit while moving without being able to stay relaxed inside.

The door-closure snapshot resolves the simultaneity question, but leaves the
physical intuition about stopping unanswered. The ladder's relaxed rest length
is still greater than the barn's length. Stopping it must expose that mismatch:
either it extends beyond the doors, or keeping it short introduces compression.
This example follows the geometry past closure to show both possibilities and
the subsequent elastic response when the compression is allowed to relax.

Stopping changes the slopes of the worldlines while their positions remain
continuous. A simultaneous stop cannot instantly lengthen the ladder: its
short separation becomes compression at rest. Another observer assigns
different times to those same kinks, but agrees on the resulting compression.

A ladder of relaxed proper length L0=1 enters a barn of length 0.8 at 0.8c.
Its barn-frame length is 0.6, so both doors can close with the ladder inside.
An open sandwich transforms the whole scene into the incoming ladder frame:
the two door-closure events are no longer simultaneous.

Closing the doors and stopping the ladder are separate operations. Compare:

* The rigid-to-rigid velocity_step construction distributes a single sharp
  impulse on the rapidity-bisector surface. The before/after proper spacing
  agrees, with no elastic degrees of freedom represented. During stopping,
  the barn-frame length grows from 0.6 to 1; the front crosses the exit.
* Stopping every material point on the barn's t=0 slice instead retains the
  length 0.6 immediately after the stop. At rest, that is actual compression
  relative to the relaxed length 1: the stop has excited a compressive mode.

Within this single-step construction, the bisector timing is the unique
impulse distribution, up to translation, that preserves proper spacing for
the specified before/after velocities. Other timings introduce strain and
excite the extended body's internal dynamics. We illustrate that excitation
through its low-energy elastic analogue: damped ringing about the relaxed
length L0.

The ringing starts at zero velocity, with stored compressive strain. Its
sinusoidal form, frequency and damping are schematic. The large speed and
compression make the spacetime geometry readable; the panel does not model
material damage at those energies. Door crossings depict yielding doors.
Keeping impenetrable doors closed would constrain the subsequent motion.

For the algebraic treatment of stopping-induced compression and relaxation,
and the strain-preserving alternative, see M. Fayngold, "The Dynamics of
Relativistic Length Contraction and the Ehrenfest Paradox", arXiv:0712.3891
(2007; revised 2020), Sec. 1(I) and 1(III). These geometric examples were
developed independently and illustrate the same arguments:
https://arxiv.org/pdf/0712.3891v3#page=3

Coordinates are (ct, x), with c = L0 = 1.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from examples import PLOT_DIR
from examples.relativity.relativistic_impulse import (
    GEOMETRY_ATOL, Vector, VectorMap, mv, velocity_step,
)


REAR, FRONT, IMPULSE = "#2475ba", "#db7220", "#258e59"
DOOR, ELASTIC = "#59616c", "#a64479"


def endpoint_positions(times, events, velocities):
    """Intersect the two piecewise straight end worldlines with time slices."""
    elapsed = np.asarray(times)[..., None] - events[:, 0]
    return events[:, 1] + np.where(elapsed < 0, velocities[0], velocities[1]) * elapsed


def ringing_length(times, damping_ratio=0.25, natural_frequency=5.0):
    """Underdamped mode with L(0)=0.6, L'(0)=0, and L(infinity)=1.

    The default damping ratio is quarter critical. natural_frequency is the
    undamped angular frequency; both decay and oscillation follow from it.
    """
    times = np.asarray(times)
    damping = damping_ratio * natural_frequency
    frequency = natural_frequency * np.sqrt(1.0 - damping_ratio**2)
    return 1.0 - 0.4 * np.exp(-damping * times) * (
        np.cos(frequency * times) + damping / frequency * np.sin(frequency * times)
    )


def style_axis(ax, title, xlim, ylim, primed=False):
    suffix = "′" if primed else ""
    ax.set(xlabel=f"x{suffix} / L₀", ylabel=f"ct{suffix} / L₀",
           xlim=xlim, ylim=ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.13)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, fontsize=12, pad=13)


def barn_background(ax, door_events, door_velocity, times):
    xs = door_events[:, 1] + door_velocity * (times[:, None] - door_events[:, 0])
    ax.fill_betweenx(times, xs[:, 0], xs[:, 1], color=DOOR, alpha=0.07)
    ax.plot(xs, times, color=DOOR, ls="--", lw=1.3, alpha=0.65)
    ax.scatter(door_events[:, 1], door_events[:, 0], marker="s", s=48,
               color=DOOR, edgecolor="white", linewidth=0.7, zorder=6)


def length_bar(ax, positions, time, label, offset=0.09, color="#333333"):
    ax.plot(positions, [time, time], lw=5, color=color, alpha=0.28,
            solid_capstyle="butt")
    ax.text(np.mean(positions), time + offset, label, ha="center", fontsize=10,
            color=color, va="bottom" if offset > 0 else "top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5})


def draw_ladder(beta, barn_length, gamma, moving_length, initial_positions, closure, moving_closure, moving_events, moving_doors, events, velocities, contact_time, plot_path) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 12.0), dpi=170)

    # 1. The ordinary paradox concerns simultaneous door closure, without a stop.
    ax = axes[0, 0]
    times = np.array([-0.6, 0.65])
    barn_background(ax, closure.kernel, 0.0, times)
    for end, color in enumerate((REAR, FRONT)):
        ax.plot(initial_positions[end] + beta * times, times, color=color, lw=2.6)
    ax.axhline(0.0, color=DOOR, lw=1, alpha=0.3)
    length_bar(ax, initial_positions, 0.0, "Moving ladder: 0.60 L₀", offset=0.10)
    ax.annotate("Doors close together", xy=(0.8, 0.0), xytext=(0.43, -0.36),
                ha="center", fontsize=10, color=DOOR,
                arrowprops={"arrowstyle": "->", "color": DOOR})
    ax.text(0.23, 0.51, "Barn: 0.80 L₀", ha="center", color=DOOR, fontsize=10)
    style_axis(ax, "1  ·  In the barn frame: it fits at closure",
               (-0.46, 1.32), (-0.6, 0.65))

    # 2. Boost the SAME closure events and incoming end worldlines.
    ax = axes[0, 1]
    times = np.array([-1.62, 0.30])
    door_beta = moving_doors[1] / moving_doors[0]
    barn_background(ax, moving_closure, door_beta, times)
    for end, color in enumerate((REAR, FRONT)):
        ax.plot([moving_events[end, 1]] * 2, times, color=color, lw=2.6)
    length_bar(ax, moving_events[:, 1], -1.40, "Ladder: 1.00 L₀", offset=0.08)
    barn_slice = moving_closure[:, 1] + door_beta * (-0.72 - moving_closure[:, 0])
    length_bar(ax, barn_slice, -0.72, "Barn: 0.48 L₀", offset=0.08, color=DOOR)
    ax.annotate("Exit closes first", xy=moving_closure[1, ::-1],
                xytext=(1.40, -0.97), fontsize=10, color=DOOR,
                arrowprops={"arrowstyle": "->", "color": DOOR})
    ax.annotate("Entrance closes later", xy=moving_closure[0, ::-1],
                xytext=(0.49, 0.13), fontsize=10, color=DOOR,
                arrowprops={"arrowstyle": "->", "color": DOOR})
    style_axis(ax, "2  ·  In the incoming ladder frame: different times",
               (-0.31, 2.32), (-1.62, 0.30), primed=True)

    # 3. Prescribed staggered braking: no proper-length mismatch to relax.
    ax = axes[1, 0]
    limits = (-0.48, 2.12)
    barn_background(ax, closure.kernel, 0.0, np.array(limits))
    for end, color in enumerate((REAR, FRONT)):
        times = np.array([limits[0], events[end, 0], limits[1]])
        xs = endpoint_positions(times, events, velocities)[:, end]
        ax.plot(xs, times, color=color, lw=2.6)
    ax.plot(events[:, 1], events[:, 0], "--", color=IMPULSE, lw=2.5)
    ax.scatter(events[:, 1], events[:, 0], color=IMPULSE, s=30, zorder=5)
    ax.text(0.30, 0.57, "Kinks joined by\ntheir bisector",
            fontsize=10, color=IMPULSE, ha="center")
    length_bar(ax, initial_positions, 0.0, "0.60 L₀ at closure", offset=-0.13)
    length_bar(ax, endpoint_positions(1.43, events, velocities), 1.43,
               "At rest: 1.00 L₀", offset=0.11)
    ax.scatter([barn_length], [contact_time], marker="*", color="#ba4242", s=125, zorder=7)
    ax.annotate("Front reaches exit", xy=(barn_length, contact_time),
                xytext=(1.28, 0.85), ha="right", fontsize=10, color="#ba4242",
                arrowprops={"arrowstyle": "->", "color": "#ba4242"})
    ax.text(0.59, 1.76, "Relaxed at rest:\ntoo long for the barn", ha="center", fontsize=10)
    style_axis(ax, "3  ·  Strain-preserving stop: no longer fits",
               (-0.44, 1.44), limits)

    # 4. Stopping inside leaves compression, whose relaxation drives ringing.
    ax = axes[1, 1]
    barn_background(ax, closure.kernel, 0.0, np.array(limits))
    times = np.linspace(0.0, limits[1], 1000)
    damping_ratio, natural_frequency = 0.25, 5.0
    damping = damping_ratio * natural_frequency
    frequency = natural_frequency * np.sqrt(1.0 - damping_ratio**2)
    lengths = ringing_length(times, damping_ratio=damping_ratio, natural_frequency=natural_frequency)
    centre = barn_length / 2
    ring_positions = centre + np.array([-0.5, 0.5]) * lengths[:, None]
    # The damping correction makes the initial post-stop end velocities zero.
    length_rates = 0.4 * np.exp(-damping * times) * (
        (frequency**2 + damping**2) / frequency * np.sin(frequency * times)
    )
    for end, color in enumerate((REAR, FRONT)):
        before_times = np.array([limits[0], 0.0])
        ax.plot(initial_positions[end] + beta * before_times, before_times, color=color, lw=2.6)
        ax.plot(ring_positions[:, end], times, color=color, lw=2.6)
        ax.axvline(centre + (end - 0.5), ymin=(0.0 - limits[0]) / np.ptp(limits),
                   color=color, ls=":", lw=1.3, alpha=0.5)
    ax.plot(initial_positions, [0.0, 0.0], "--", color=ELASTIC, lw=2.6)
    ax.scatter(initial_positions, [0.0, 0.0], color=ELASTIC, s=32, zorder=5)
    length_bar(ax, initial_positions, 0.0, "Stopped at 0.60 L₀: compressed",
               offset=-0.14, color=ELASTIC)
    # First contact with both doors, found on the expanding part of the mode.
    first_peak = times <= np.pi / frequency
    ringing_contact = np.interp(barn_length, lengths[first_peak], times[first_peak])
    ax.scatter([0.0, barn_length], [ringing_contact] * 2,
               marker="*", color="#ba4242", s=125, zorder=7)
    ax.annotate("Expansion reaches both doors", xy=(barn_length, ringing_contact),
                xytext=(0.4, 0.50), ha="center", fontsize=10, color="#ba4242",
                arrowprops={"arrowstyle": "->", "color": "#ba4242"})
    length_bar(ax, [centre - 0.5, centre + 0.5], 1.91,
               "Relaxed length: 1.00 L₀", offset=0.08)
    ax.text(0.4, 1.04, "Elastic ringing\nQuarter-critical\ndamping",
            ha="center", fontsize=10, color=ELASTIC)
    style_axis(ax, "4  ·  Stop inside: compression and ringing",
               (-0.44, 1.44), limits)

    fig.suptitle("The ladder paradox: fitting in motion, stopping in a shorter barn", fontsize=17, y=0.98)
    fig.text(0.5, 0.948, "Relaxed ladder L₀ = 1   ·   Barn = 0.8 L₀   ·   Incoming speed = 0.8c",
             ha="center", fontsize=11, color=DOOR)
    legend = [Line2D([], [], color=REAR, lw=2.5, label="Rear end"),
              Line2D([], [], color=FRONT, lw=2.5, label="Front end"),
              Line2D([], [], color=DOOR, ls="--", lw=1.3, label="Door position"),
              Line2D([], [], color=DOOR, marker="s", ls="none", label="Door closes")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.931),
               ncol=4, frameon=False, fontsize=10)
    fig.text(0.5, 0.035,
             "The ladder can fit for an instant; it cannot remain inside at its relaxed rest length.\n"
             "Door crossings assume yielding doors; ringing illustrates an elastic mode, not a collision calculation.",
             ha="center", fontsize=10, color=DOOR, linespacing=1.6)
    fig.tight_layout(rect=(0.015, 0.075, 0.99, 0.90), h_pad=3.2, w_pad=3.0)
    if plot_path:
        fig.savefig(plot_path, bbox_inches="tight")
        print(f"Figure saved to {plot_path}")
    return fig


def main(plot_path: str = str(PLOT_DIR / "ladder_paradox.png")) -> plt.Figure:
    beta, barn_length = 0.8, 0.8
    rapidity = np.arctanh(beta)
    gamma = np.cosh(rapidity)
    moving_length = 1 / gamma
    rear_at_closure = (barn_length - moving_length) / 2
    initial_positions = np.array([rear_at_closure, rear_at_closure + moving_length])
    closure: Vector = mv.vector([[0.0, 0.0], [0.0, barn_length]])
    incoming_events: Vector = mv.vector(np.stack([np.zeros(2), initial_positions], axis=-1))
    incoming_direction: Vector = mv.vector([gamma, gamma * beta])

    # One extensor transforms all events and tangents to the incoming rest frame.
    angle = rapidity / 2
    boost: VectorMap = (angle * mv.tx).exp().normalized().sandwich(Vector)
    moving_closure = boost(closure).kernel
    moving_events = boost(incoming_events).kernel
    moving_direction = boost(incoming_direction).kernel
    moving_doors = boost(mv.vector([1.0, 0.0])).kernel

    # The unchanged-proper-length stop is the same velocity-step building block.
    directions = (-mv.tx * mv.scalar([[rapidity], [0.0]]) / 2).exp().normalized() >> mv.t
    separation = ((directions[0] + directions[1]) * mv.tx) / (1 + (directions[0] | directions[1]))
    steps = mv.scalar([[0.0], [1.0]]) * separation
    steps = steps + mv.vector([0.0, rear_at_closure])
    events = steps.kernel
    velocities = directions.kernel[:, 1] / directions.kernel[:, 0]
    contact_time = (barn_length - initial_positions[1]) / beta

    fig = draw_ladder(beta, barn_length, gamma, moving_length, initial_positions, closure,
                      moving_closure, moving_events, moving_doors, events, velocities,
                      contact_time, plot_path)

    # --- checks -------------------------------------------------------------
    np.testing.assert_allclose(moving_direction, [1.0, 0.0], atol=GEOMETRY_ATOL)
    np.testing.assert_allclose(np.diff(moving_events[:, 1]), 1.0, atol=GEOMETRY_ATOL)
    np.testing.assert_allclose(moving_closure[:, 0], [0.0, -gamma * beta * barn_length])
    np.testing.assert_allclose(
        (boost(closure) | boost(closure)).kernel,
        (closure | closure).kernel, atol=GEOMETRY_ATOL,
    )
    np.testing.assert_allclose(events, [[0.0, 0.1], [0.5, 1.1]], atol=GEOMETRY_ATOL)
    np.testing.assert_allclose(velocities, [beta, 0.0], atol=GEOMETRY_ATOL)
    np.testing.assert_allclose(
        ((directions[0] + directions[1]) | (steps[1] - steps[0])).kernel,
        0.0, atol=GEOMETRY_ATOL,
    )
    np.testing.assert_allclose(endpoint_positions(0.0, events, velocities), initial_positions)
    np.testing.assert_allclose(np.diff(endpoint_positions(0.6, events, velocities)), 1.0)

    return fig


if __name__ == "__main__":
    main()
    plt.show()
