"""Bell's spaceships: synchronized clocks versus strain-preserving impulses.

Two ships start at rest, separated by a rope of relaxed proper length L0=1.
Compare the same ten rapidity increments, from rest to 0.8c, with two timings:

* The strain-preserving impulse train from relativistic_impulse gives the
  rear ship the reference program and delays each matching front impulse to
  the step's rapidity-bisector surface. Applying this prescription throughout
  the connecting material preserves proper spacing between successive inertial
  states. The lab separation contracts, while the final rest separation is
  unchanged.
* Bell's ships execute identical programs on initially synchronized onboard
  clocks. Identical velocity histories give identical time dilation, so equal
  trigger readings occur at equal times in the initial lab. The front worldline
  is a spatial translation of the rear's: lab separation stays unchanged.
  After both ships coast, an equal-time slice in their shared rest frame
  spans a longer segment. The rope must stretch to maintain that connection.

Every impulse changes a worldline's slope without breaking its continuity.
Equal clock programs do not make the rope jump to a new contracted length;
they demand a change in proper spacing. In the final moving frame the same
impulses have staggered times. Both observers agree on the required stretch,
despite describing its development with different simultaneity slices.

The rear has equal proper-time intervals between impulses in both examples.
Bell's ships execute the same evenly spaced onboard-clock program. The
corresponding lab-time intervals grow as the ships speed up. These clocks are
synchronized in the initial lab, not continually resynchronized in successive
moving frames.

The connecting dashed lines are impulse schedules, not propagating signals.
The strain-preserving example prescribes distributed forcing. Bell's example
prescribes ship trajectories; passive-rope stresses and any breaking threshold
require a material model. The final panel shows the required rope extension,
computed by boosting both last-kink events with the same open sandwich.

The finite strain-preserving step has an algebraic counterpart in M. Fayngold,
"The Dynamics of Relativistic Length Contraction and the Ehrenfest Paradox",
arXiv:0712.3891 (2007; revised 2020), Sec. 1(III), Eq. (9):
https://arxiv.org/pdf/0712.3891v3#page=13
The geometric construction and its independent development are described in
relativistic_impulse.

For Bell motion versus continuous Born-rigid acceleration, see Franklin,
European Journal of Physics 31 (2010), sections 2 and 3:
https://arxiv.org/abs/0906.1919

Coordinates are (ct, x), with c = L0 = 1.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from examples import PLOT_DIR
from examples.relativity.relativistic_impulse import (
    GEOMETRY_ATOL, Vector, VectorMap, mv, small_impulses,
)


REAR, FRONT = "#2475ba", "#db7220"
PRESERVED, STRETCHED = "#258e59", "#a64479"


def positions_at(times, events, velocities):
    """Integrate each endpoint's velocity jumps from initial positions (0, 1)."""
    elapsed = np.maximum(np.asarray(times)[..., None, None] - events[:, :, 0], 0.0)
    return np.array([0.0, 1.0]) + (np.diff(velocities)[:, None] * elapsed).sum(axis=-2)


def trigger_readings(events, velocities):
    """Proper clock readings at every kink; both clocks read zero at lab t=0."""
    intervals = np.diff(events[:, :, 0], axis=0)
    elapsed = intervals * np.sqrt(1.0 - velocities[1:-1, None]**2)
    return np.concatenate([events[:1, :, 0], events[:1, :, 0] + np.cumsum(elapsed, axis=0)])


def rope_comparison(ax, gap, y, color, title, label):
    """A spatial slice in the final rest frame, aligned on the rear attachment."""
    ax.text(0.0, y + 0.36, title, color=color, fontsize=11, weight="medium")
    ax.plot([0.0, gap], [y, y], color=color, lw=3)
    for x, ship_color in ((0.0, REAR), (gap, FRONT)):
        ax.scatter([x], [y], marker=">", s=160, facecolor="white",
                   edgecolor=ship_color, linewidth=2.0, zorder=5)
    ax.annotate("", xy=(gap, y - 0.17), xytext=(0.0, y - 0.17),
                arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.2})
    ax.text(gap / 2, y - 0.29, label, color=color, ha="center", va="top", fontsize=10)


def main(plot_path: str = str(PLOT_DIR / "bell_spaceships.png")) -> plt.Figure:
    beta, count, dt = 0.8, 10, 0.1
    rapidity = np.arctanh(beta)
    gamma = np.cosh(rapidity)

    preserved, velocities = small_impulses(rapidity, count=count, dt=dt)
    # Bell's front follows an exact spatial translation of the SAME rear path.
    rear_events = preserved.kernel[:, 0]
    bell: Vector = mv.vector(np.stack([rear_events, rear_events + [0.0, 1.0]], axis=1))
    schedules = ((preserved, PRESERVED), (bell, STRETCHED))
    clocks = [trigger_readings(steps.kernel, velocities) for steps, _ in schedules]
    np.testing.assert_allclose(clocks[1][:, 0], clocks[1][:, 1], atol=GEOMETRY_ATOL)
    np.testing.assert_allclose(clocks[0][:, 0], clocks[1][:, 0], atol=GEOMETRY_ATOL)
    np.testing.assert_allclose(clocks[1][:, 0], np.arange(count) * dt, atol=GEOMETRY_ATOL)

    # Boost the final coasting worldlines, then measure at equal final-frame time.
    angle = rapidity / 2
    final_frame: VectorMap = (angle * mv.tx).exp().normalized().sandwich(Vector)
    final_direction: Vector = mv.vector([gamma, gamma * beta])
    np.testing.assert_allclose(final_frame(final_direction).kernel, [1.0, 0.0], atol=GEOMETRY_ATOL)
    gaps = []
    for steps, _ in schedules:
        # Anchor each view at the rear's final kink; the front's kink need not
        # be simultaneous. Its final worldline is vertical, so x' is constant.
        final_events = final_frame(steps[-1] - steps[-1, 0]).kernel
        gaps.append(final_events[1, 1] - final_events[0, 1])
    np.testing.assert_allclose(gaps, [1.0, gamma], atol=GEOMETRY_ATOL)

    end_time = preserved.kernel[-1, :, 0].max() + 0.53
    sample_time = end_time - 0.22
    lab_gaps = [np.diff(positions_at(sample_time, steps.kernel, velocities))[0]
                for steps, _ in schedules]
    np.testing.assert_allclose(lab_gaps, [1.0 / gamma, 1.0], atol=GEOMETRY_ATOL)
    sample_times = np.linspace(-0.3, end_time, 300)
    np.testing.assert_allclose(np.diff(positions_at(sample_times, bell.kernel, velocities), axis=-1),
                               1.0, atol=GEOMETRY_ATOL)

    fig, axes = plt.subplots(1, 3, figsize=(16.7, 7.5), dpi=170,
                             gridspec_kw={"width_ratios": [1.0, 1.0, 0.85]})
    titles = ("Strain-preserving impulse train", "Bell: identical clock programs")
    for index, ((steps, color), ax) in enumerate(zip(schedules, axes[:2])):
        events = steps.kernel
        for end, ship_color in enumerate((REAR, FRONT)):
            times = np.concatenate([[-0.30], events[:, end, 0], [end_time]])
            xs = positions_at(times, events, velocities)[:, end]
            np.testing.assert_allclose(xs[1:-1], events[:, end, 1], atol=GEOMETRY_ATOL)
            ax.plot(xs, times, color=ship_color, lw=2.6)
        for step in events:
            ax.plot(step[:, 1], step[:, 0], "--", color=color, lw=1.35, alpha=0.8)
            ax.scatter(step[:, 1], step[:, 0], color=color, s=11, zorder=4)

        for time, label in ((-0.18, "Initially: 1.00 L₀"),
                            (sample_time, f"Final lab gap: {lab_gaps[index]:.2f} L₀")):
            endpoints = positions_at(time, events, velocities)
            ax.plot(endpoints, [time, time], color="#333333", lw=5, alpha=0.27)
            ax.text(np.mean(endpoints), time + (0.09 if time > 0 else -0.09),
                    label, ha="center", va="bottom" if time > 0 else "top", fontsize=10,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1})
        note = ("Front impulses follow the bisectors"
                if index == 0 else "Matching clock readings; horizontal timing lines")
        ax.text(0.5, 1.025, note, transform=ax.transAxes, ha="center", va="bottom", fontsize=10, color=color)
        ax.set_title(titles[index], fontsize=12, pad=38)
        ax.set(xlabel="x / L₀", ylabel="ct / L₀", xlim=(-0.20, 3.05), ylim=(-0.40, end_time + 0.18))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0, 1, 2, 3])
        ax.set_yticks([0, 1, 2])
        ax.grid(alpha=0.13)
        ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    ax.set_title("In the final shared rest frame", fontsize=12, pad=38)
    rope_comparison(ax, gaps[0], 2.05, PRESERVED, "Strain-preserving timing",
                    "1.00 L₀ · no extension")
    rope_comparison(ax, gaps[1], 0.82, STRETCHED, "Synchronized-clock timing",
                    f"{gaps[1]:.2f} L₀ · {100 * (gaps[1] - 1):.1f}% extension")
    ax.plot([1.0, 1.0], [0.65, 2.25], color="#888888", ls=":", lw=1.2)
    ax.text(0.0, 2.78, "Same final speed: 0.8c\nDifferent required rope lengths", fontsize=10, color="#59616c")
    ax.text(0.0, 0.04, "Rope tension grows; failure depends\non its strength and elastic response.",
            fontsize=10, color=STRETCHED)
    ax.set(xlim=(-0.22, 2.05), ylim=(-0.35, 3.03))
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    fig.suptitle("Bell’s spaceships: the timing across the rope matters", fontsize=17, y=0.99)
    fig.text(0.5, 0.937, "10 matching impulses   ·   0 → 0.8c   ·   Same rear-ship history in both cases",
             ha="center", fontsize=11, color="#59616c")
    legend = [Line2D([], [], color=REAR, lw=2.6, label="Rear ship"),
              Line2D([], [], color=FRONT, lw=2.6, label="Front ship"),
              Line2D([], [], color="#666666", ls="--", lw=1.4, label="Matching impulse events")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.903),
               ncol=3, frameon=False, fontsize=10)
    fig.text(0.5, 0.045,
             "The bisector train preserves proper spacing. Identical synchronized clock programs preserve the initial lab gap instead.\n"
             "Dashed lines specify timing; they are not signals or sound waves. Rope extension is kinematic; elastic dynamics are not simulated.",
             ha="center", fontsize=10, color="#59616c", linespacing=1.6)
    fig.subplots_adjust(left=0.055, right=0.98, bottom=0.19, top=0.75, wspace=0.28)
    if plot_path:
        fig.savefig(plot_path, bbox_inches="tight")
        print(f"Figure saved to {plot_path}")
    return fig


if __name__ == "__main__":
    main()
    plt.show()
