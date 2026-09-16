"""Strain-preserving impulses, built from a symmetric reversal and its boosts.

An impulse changes a worldline's slope, not its position. In a fixed observer
frame, the body's equal-time length cannot jump while its endpoint worldlines
remain continuous. A sudden velocity change therefore cannot simply replace
the body's old length with the relaxed Lorentz-contracted length of its new
motion. Arbitrary timing across the body generally leaves compression or
stretch, which drives an internal response and may relax to a different length.
Changing observer alone introduces no strain: Lorentz contraction compares
different equal-time cuts of the same worldlines.

Start in the symmetric frame: the rod reverses direction everywhere at once,
keeping the same speed. Each end's incoming and outgoing worldline segments
are mirror images across that instant, and the measured length is unchanged.
Both motions have the same proper spacing, so there is no strain mismatch to
relax. An open sandwich changes the observer of the entire drawing. The line
joining the kinks tilts, becoming the spacelike hyperbolic bisector of the old
and new motion. The observer-frame length now changes continuously as the two
ends reach their kinks at different times.

Each material point makes one sharp jump between specified uniform velocities.
For unequal velocities, unchanged proper spacing on both sides fixes this
kink line up to translation: a strain-preserving, or constant-strain, impulse.
The symmetric reversal, Lorentz transformation and continuity of worldlines
are sufficient to construct it geometrically.

Observers agree on the local impulse events, worldline continuity and material
strain, while assigning different simultaneity slices and measured lengths.
Impulses simultaneous across the body in one frame become a tilted sequence
of events in another. That apparent travel is an ordering of distributed
impulses, not propagation of a signal or sound wave. Both descriptions give
the same physical strain outcome.

Repeatedly applying one fixed boost followed by one fixed translation produces
the train. Each application advances both worldlines to their next kinks.
Events are evenly spaced on the left end's proper clock; their spacing in
observer time grows as the body speeds up.

This uniqueness concerns the prescribed single-step construction. Unchanged
proper spacing and uniform outgoing motion leave no elastic mismatch to relax.
The barn and spaceship examples illustrate compression and stretching from
other timings, with the subsequent response depending on the material.

This geometric presentation was developed independently of Moses Fayngold's
algebraic treatment, but the finite strain-preserving timing is the same:
M. Fayngold, "The Dynamics of Relativistic Length Contraction and the Ehrenfest
Paradox", arXiv:0712.3891 (2007; revised 2020), Sec. 1(III),
"Non-simultaneous braking", Eq. (9):
https://arxiv.org/pdf/0712.3891v3#page=13

Coordinates are (ct, x), with c = proper rod length L0 = 1.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples import PLOT_DIR
from numga import Algebra, NumpyContext, stack


Spacetime = Algebra("t+x-")
ctx = NumpyContext(Spacetime)
mv = ctx.multivector

Scalar = Spacetime.gatype.scalar()
Vector = Spacetime.gatype.vector()
Rotor = Spacetime.gatype.rotor()
VectorMap = Spacetime.gatype((Vector, Vector))

GEOMETRY_ATOL = 1e-9  # Allow for the library's approximate bivector exp().


def strain_preserving_kinks(before: Vector, after: Vector) -> Vector:
    """Join unit future timelike directions with a unit-proper-length step.

    The summed directions give the midpoint observer's time axis. Its
    spacelike perpendicular supplies the kink line; scale it so that both
    rest frames measure the original proper spacing. Anchor the left kink
    at the origin. This construction is specific to the (ct, x) plane.
    """
    separation: Vector = ((before + after) * mv.tx) / (1 + (before | after))
    return mv.scalar([[0.0], [1.0]]) * separation


def velocity_step(
    before_rapidity: float, after_rapidity: float,
) -> tuple[Vector, Vector]:
    """A prescribed rigid-to-rigid step, anchored at the left endpoint's kink.

    Return the two kink events and unit before/after directions. For unequal
    specified velocities, a single sharp jump with unchanged proper spacing
    uniquely fixes the kink surface up to translation. It is horizontal in
    the rapidity-bisector frame, where the velocity reverses direction at the
    same speed without a length change. Rotors supply the two directions;
    their bisector supplies the kink events.

    This kinematic construction represents no flexible modes or sound wave;
    it does not establish uniqueness for general dynamical histories. The
    proper rod length is 1.
    """
    rapidities: Scalar = mv.scalar([[before_rapidity], [after_rapidity]])
    rotors: Rotor = (-rapidities * mv.tx / 2).exp().normalized()
    directions: Vector = rotors >> mv.t
    return strain_preserving_kinks(directions[0], directions[1]), directions


def small_impulses(
    final_rapidity: float, count: int = 10, dt: float = 0.1,
) -> tuple[Vector, np.ndarray]:
    """Compose one fixed boost-plus-translation to generate successive steps.

    Return kink events with batch shape (step, endpoint), and the velocities
    before, between and after the steps. dt is PROPER time between left-end
    kinks. The same affine map advances the kink pair on every iteration;
    its linear part advances the worldline directions. Negative steps need
    sufficient spacing to keep the right-end events in chronological order.
    """
    # Normalize once to remove exp's tiny scale error before reusing the map.
    rotor: Rotor = (-mv.tx * final_rapidity / (2 * count)).exp().normalized()
    boost: VectorMap = rotor.sandwich(Vector)
    before: Vector = mv.vector([1.0, 0.0])
    after: Vector = boost(before)
    events: Vector = strain_preserving_kinks(before, after)
    offset: Vector = dt * after

    def advance(pair: Vector) -> Vector:
        # Both the boost and the translation stay FIXED throughout the train.
        return boost(pair) + offset

    event_history = [events]
    direction_history = [before, after]
    for _ in range(count - 1):
        events = advance(events)
        after = boost(after)
        event_history.append(events)
        direction_history.append(after)

    kinks: Vector = stack(event_history)
    directions: Vector = stack(direction_history)
    velocities = directions.kernel[:, 1] / directions.kernel[:, 0]
    return kinks, velocities


def main(plot_path: str = str(PLOT_DIR / "relativistic_rod_impulse.png")) -> plt.Figure:
    u = 0.5
    half_rapidity = np.arctanh(u)
    gamma_u = np.cosh(half_rapidity)
    symmetric_length = 1.0 / gamma_u

    # Start in the symmetric frame: two identical velocity reversals at t=0.
    kinks, directions = velocity_step(-half_rapidity, half_rapidity)
    before, after = directions
    bisector: Vector = (before + after).normalized()
    np.testing.assert_allclose(bisector.kernel, [1.0, 0.0], atol=GEOMETRY_ATOL)
    np.testing.assert_allclose((kinks | bisector).kernel, 0.0, atol=GEOMETRY_ATOL)

    identity: VectorMap = Spacetime.operator.identity(Vector.output_subspace)
    # Leave the passenger open to obtain the observer change as an extensor.
    angle = -half_rapidity / 2
    rotor: Rotor = (angle * mv.tx).exp().normalized()
    initial_rest_frame: VectorMap = rotor.sandwich(Vector)
    transformed_directions = initial_rest_frame(directions).kernel
    final_beta = 2*u / (1+u*u)
    np.testing.assert_allclose(transformed_directions[:, 1] / transformed_directions[:, 0], [0.0, final_beta], atol=GEOMETRY_ATOL)
    np.testing.assert_allclose(initial_rest_frame(kinks).kernel, [[0.0, 0.0], [0.5, 1.0]], atol=GEOMETRY_ATOL)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 7.1), dpi=160)
    colors = ("#2475ba", "#db7220", "#258e59")
    for index, (ax, frame, title) in enumerate((
        (axes[0], identity, "Symmetric frame: −0.5c → +0.5c"),
        (axes[1], initial_rest_frame, "Boosted frame: 0 → 0.8c"),
    )):
        # Apply the SAME map to the kink events and the before/after directions.
        events = frame(kinks).kernel
        tangents = frame(directions).kernel
        velocities = tangents[:, 1] / tangents[:, 0]
        np.testing.assert_allclose((frame(bisector) | frame(kinks)).kernel, 0.0, atol=GEOMETRY_ATOL)

        def positions(t: float) -> np.ndarray:
            elapsed = t - events[:, 0]
            return events[:, 1] + np.where(elapsed < 0, velocities[0], velocities[1]) * elapsed

        for end, label in enumerate(("Left end", "Right end")):
            times = np.array([-0.60, events[end, 0], 1.45])
            xs = np.array([positions(t)[end] for t in times])
            ax.plot(xs, times, color=colors[end], lw=2.5, label=label)
        ax.plot(events[:, 1], events[:, 0], "--", color=colors[2], lw=2.5, label="Velocity-step surface")
        ax.scatter(events[:, 1], events[:, 0], color=colors[2], s=32, zorder=4)

        for t, prefix in ((-0.30, "Before"), (1.05, "After")):
            a, b = positions(t)
            expected_length = np.sqrt(1.0 - velocities[0 if t < 0 else 1]**2)
            np.testing.assert_allclose(b-a, expected_length, atol=GEOMETRY_ATOL)
            ax.plot([a, b], [t, t], color="#333333", lw=5, alpha=0.35)
            ax.text((a+b)/2, t+(-0.14 if t<0 else 0.12),
                    f"{prefix}: {b-a:.3f} L₀", ha="center", fontsize=10)
        if index == 0:
            ax.text(symmetric_length/2, 0.12, "Simultaneous reversal", ha="center", color=colors[2], fontsize=10)
        else:
            ax.text(0.34, 0.41, "Same line, boosted", color=colors[2], fontsize=10)
        ax.set_title(title, fontsize=12, loc="left", pad=14)
        ax.set(xlabel="x / L₀", ylabel="ct / L₀", xlim=(-0.16, 1.93), ylim=(-0.70, 1.53))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0, 0.5, 1.0, 1.5])
        ax.set_yticks([-0.5, 0, 0.5, 1.0, 1.5])
        ax.grid(alpha=0.15)
        ax.spines[["top", "right"]].set_visible(False)

    # -----------------------------------------------------------------------
    # 3. Ten small impulses from repeated composition of the same affine map.
    # -----------------------------------------------------------------------
    count, dt = 10, 0.1
    steps, velocities = small_impulses(2 * half_rapidity, count=count, dt=dt)
    events = steps.kernel
    ax = axes[2]
    finish_time = events[-1, :, 0].max()
    end_time = finish_time + 0.40

    def stepped_positions(t: float) -> np.ndarray:
        # Start with the initially stationary rod, then sum each velocity jump
        # only after that endpoint has crossed its corresponding step surface.
        elapsed = np.maximum(t - events[:, :, 0], 0.0)
        return np.array([0.0, 1.0]) + (np.diff(velocities)[:, None] * elapsed).sum(axis=0)

    for end, label in enumerate(("Left end", "Right end")):
        times = np.concatenate([[-0.30], events[:, end, 0], [end_time]])
        xs = np.array([stepped_positions(t)[end] for t in times])
        np.testing.assert_allclose(xs[1:-1], events[:, end, 1], atol=GEOMETRY_ATOL)
        ax.plot(xs, times, color=colors[end], lw=2.5, label=label)
    for step in events:
        ax.plot(step[:, 1], step[:, 0], "--", color=colors[2], lw=1.4, alpha=0.75)
        ax.scatter(step[:, 1], step[:, 0], color=colors[2], s=12, zorder=4)

    sample_time = finish_time + 0.20
    a, b = stepped_positions(sample_time)
    np.testing.assert_allclose(b-a, np.sqrt(1-final_beta**2), atol=GEOMETRY_ATOL)
    ax.plot([a, b], [sample_time, sample_time], color="#333333", lw=5, alpha=0.35)
    ax.text((a+b)/2, sample_time+0.12, f"After: {b-a:.3f} L₀", ha="center", fontsize=10)
    ax.text(0.04, 0.96, "Repeat one fixed map\nEqual ticks on the left clock",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.set_title("10 impulses: 0 → 0.8c", fontsize=12, loc="left", pad=14)
    ax.set(xlabel="x / L₀", ylabel="ct / L₀", xlim=(-0.16, 2.7), ylim=(-0.38, end_time+0.30))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.grid(alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Strain-preserving impulses: start with a symmetric reversal", y=0.99)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               bbox_to_anchor=(0.5, 0.945), ncol=3, frameon=False, fontsize=10)
    fig.text(0.5, 0.025,
             "Velocity changes at kinks; length changes continuously along connected worldlines.\n"
             "Different observers take different time slices of the same drawing and agree on the material strain.",
             ha="center", fontsize=9.5, color="#59616c", linespacing=1.5)
    fig.tight_layout(rect=(0, 0.13, 1, 0.86))
    if plot_path:
        fig.savefig(plot_path, bbox_inches="tight")
        print(f"Figure saved to {plot_path}")
    return fig


if __name__ == "__main__":
    main()
    plt.show()
