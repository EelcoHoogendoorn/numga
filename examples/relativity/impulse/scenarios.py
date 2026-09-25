"""One function per figure: the scene of each impulse example, built from `core`.

Each returns the events and worldlines its figure draws. Coordinates are (ct, x), with c = L0 = 1.
"""

from __future__ import annotations

import numpy as np

from numga import stack

from examples.relativity.impulse.core import (
    GEOMETRY_ATOL, Scalar, Vector, VectorMap, boost, coasting, mv, ringing_length,
    small_impulses, velocity_step, worldline_events, worldlines,
)


def times(*values: float) -> Scalar:
    """Observer times as a batch of scalars."""
    return mv.scalar(np.array(values)[:, None])


def impulse():
    """A symmetric reversal, the same drawing boosted, and a train of ten small impulses.

    Returns the worldlines [view, vertex, end], kinks [view, end] and before/after slices
    [view, slice, end] of the symmetric and boosted views, then the train's worldlines, kinks
    [step, end] and final slice [end].
    """
    speed = 0.5
    half_rapidity = np.arctanh(speed)

    # Start in the symmetric frame: two identical velocity reversals at t=0.
    kinks, directions = velocity_step(-half_rapidity, half_rapidity)
    before, after = directions
    bisector: Vector = (before + after).normalized()

    # Leave the passenger open to obtain the observer change as an extensor. Apply each
    # observer map to both events and tangents, preserving incidence.
    identity: VectorMap = mv.rotor() >> Vector
    initial_rest_frame: VectorMap = boost(half_rapidity)
    views: VectorMap = stack((identity, initial_rest_frame))
    view_kinks: Vector = views[:, None](kinks)                          # [view, end]
    view_directions: Vector = views[:, None](directions)                # [view, step + 1]
    view_worldlines: Vector = stack([
        worldlines(view_kinks[view, None], view_directions[view], times(-0.60, 1.45))
        for view in range(2)
    ])
    view_slices: Vector = stack([
        worldline_events(times(-0.30, 1.05), view_kinks[view, None], view_directions[view])
        for view in range(2)
    ])

    # Repeat a fixed boost and fixed translation to build a train of impulses.
    steps, train_directions = small_impulses(2 * half_rapidity, 10, 0.1)
    finish: Scalar = steps[-1, 1] | mv.t                                # the front's last kink ends the train
    train: Vector = worldlines(steps, train_directions, stack((mv.scalar([-0.30]), finish + 0.40)))
    train_slice: Vector = worldline_events(finish + 0.20, steps, train_directions)

    # --- checks -------------------------------------------------------------
    # The kinks are simultaneous for the symmetric observer, whose time axis bisects the motions.
    np.testing.assert_allclose((bisector | mv.t).to_array(), 1.0, atol=GEOMETRY_ATOL)
    np.testing.assert_allclose((kinks | bisector).to_array(), 0.0, atol=GEOMETRY_ATOL)
    # Boosted, the same drawing is a step from rest to the relativistic sum of the speeds,
    # and the kinks are no longer simultaneous.
    boosted: Vector = view_directions[1]
    np.testing.assert_allclose((-(boosted | mv.x) / (boosted | mv.t)).to_array(),
                               [0.0, 2 * speed / (1 + speed**2)], atol=GEOMETRY_ATOL)
    np.testing.assert_allclose((view_kinks[1] | mv.t).to_array(), [0.0, 0.5], atol=GEOMETRY_ATOL)

    return view_worldlines, view_kinks, view_slices, train, steps, train_slice


def ladder():
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

    Returns the geometry of the four panels: the barn frame, the incoming ladder frame,
    the strain-preserving stop, and the stop inside with its ringing.
    """
    beta, barn_length = 0.8, 0.8
    rapidity = np.arctanh(beta)
    moving_length = 1 / np.cosh(rapidity)
    rear_at_closure = (barn_length - moving_length) / 2
    closure: Vector = mv.vector([[0.0, 0.0], [0.0, barn_length]])        # [door] closure events
    incoming: Vector = mv.vector([[0.0, rear_at_closure], [0.0, rear_at_closure + moving_length]])  # [end] at closure
    incoming_direction: Vector = boost(rapidity)(mv.t)
    at_rest: Vector = mv.vector([1.0, 0.0])

    # 1. The ordinary paradox concerns simultaneous door closure, without a stop.
    span = times(-0.6, 0.65)[:, None]
    barn_frame = (coasting(closure, at_rest, span), closure, coasting(incoming, incoming_direction, span), incoming)

    # 2. One extensor transforms all events and tangents to the incoming rest frame.
    to_ladder: VectorMap = boost(-rapidity)
    moving_closure, moving_events = to_ladder(closure), to_ladder(incoming)
    moving_direction, moving_doors = to_ladder(incoming_direction), to_ladder(at_rest)
    span = times(-1.62, 0.30)[:, None]
    ladder_frame = (
        coasting(moving_closure, moving_doors, span), moving_closure, coasting(moving_events, moving_direction, span),
        coasting(moving_events, moving_direction, mv.scalar([-1.40])), coasting(moving_closure, moving_doors, mv.scalar([-0.72])),
    )

    # 3. The unchanged-proper-length stop is the same velocity-step building block.
    limits = times(-0.48, 2.12)
    kinks, directions = velocity_step(rapidity, 0.0)
    steps: Vector = (kinks + mv.vector([0.0, rear_at_closure]))[None]  # [step, end]
    contact_time = (barn_length - rear_at_closure - moving_length) / beta
    stopped: Vector = worldline_events(mv.scalar([1.43]), steps, directions)
    strain_preserving = (
        coasting(closure, at_rest, limits[:, None]), closure, worldlines(steps, directions, limits), steps[0],
        incoming, stopped, worldline_events(mv.scalar([contact_time]), steps, directions)[1],
    )

    # 4. Stopping inside leaves compression, whose relaxation drives ringing about the relaxed
    #    length, centred in the barn. First contact with both doors is found on the expanding
    #    part of the mode.
    damping_ratio, natural_frequency = 0.25, 5.0
    ring_times = np.linspace(0.0, 2.12, 1000)
    lengths = ringing_length(ring_times, damping_ratio, natural_frequency)
    centre = barn_length / 2
    ring: Vector = mv.t * ring_times[:, None] + mv.x * (centre + np.array([-0.5, 0.5]) * lengths[:, None])
    relaxed: Vector = mv.x * (centre + np.array([-0.5, 0.5]))            # [end] relaxed ends at t=0
    first_peak = ring_times <= np.pi / (natural_frequency * np.sqrt(1.0 - damping_ratio**2))
    ringing_contact = np.interp(barn_length, lengths[first_peak], ring_times[first_peak])
    ringing = (
        coasting(closure, at_rest, limits[:, None]), closure,
        coasting(incoming, incoming_direction, times(-0.48, 0.0)[:, None]),
        ring, coasting(relaxed, at_rest, times(0.0, 2.12)[:, None]), incoming,
        coasting(closure, at_rest, mv.scalar([ringing_contact])), coasting(relaxed, at_rest, mv.scalar([1.91])),
    )

    # --- checks -------------------------------------------------------------
    # In its own frame the incoming ladder is at rest with its full proper length,
    # and the exit closes first.
    np.testing.assert_allclose((-(moving_direction | mv.x)).to_array(), 0.0, atol=GEOMETRY_ATOL)
    np.testing.assert_allclose((-((moving_events[1] - moving_events[0]) | mv.x)).to_array(), 1.0, atol=GEOMETRY_ATOL)
    np.testing.assert_allclose((moving_closure | mv.t).to_array(), [0.0, -np.sinh(rapidity) * barn_length],
                               atol=GEOMETRY_ATOL)
    # The strain-preserving stop joins kinks perpendicular to the rapidity bisector; the front
    # stops beyond the exit, and at rest the ladder has its relaxed length.
    np.testing.assert_allclose(((directions[0] + directions[1]) | (steps[0, 1] - steps[0, 0])).to_array(),
                               0.0, atol=GEOMETRY_ATOL)
    np.testing.assert_allclose((-(steps[0] | mv.x)).to_array(), [0.1, 1.1], atol=GEOMETRY_ATOL)
    np.testing.assert_allclose((-((stopped[1] - stopped[0]) | mv.x)).to_array(), 1.0, atol=GEOMETRY_ATOL)

    return barn_frame, ladder_frame, strain_preserving, ringing


def spaceships():
    """Bell's spaceships: synchronized clocks versus strain-preserving impulses.

    Two ships start at rest, separated by a rope of relaxed proper length L0=1.
    Compare the same ten rapidity increments, from rest to 0.8c, with two timings:

    * The strain-preserving impulse train from small_impulses gives the
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
    the core module.

    For Bell motion versus continuous Born-rigid acceleration, see Franklin,
    European Journal of Physics 31 (2010), sections 2 and 3:
    https://arxiv.org/abs/0906.1919

    Returns, per schedule (strain-preserving, Bell), the worldlines [schedule, vertex, end],
    kinks [schedule, step, end], lab slices [schedule, slice, end], and the final rest-frame
    events of both ships [schedule, end].
    """
    beta, count, dt = 0.8, 10, 0.1
    rapidity = np.arctanh(beta)

    preserved, directions = small_impulses(rapidity, count, dt)
    # Bell's front follows an exact spatial translation of the SAME rear path.
    bell: Vector = preserved[:, :1] + mv.vector([[0.0, 0.0], [0.0, 1.0]])
    schedules: Vector = stack((preserved, bell))                          # [schedule, step, end]

    # Boost the final coasting worldlines, then measure at equal final-frame time. Anchor each
    # view at the rear's final kink; the front's kink need not be simultaneous. Its final
    # worldline is vertical, so x' is constant.
    final_frame: VectorMap = boost(-rapidity)
    final_events: Vector = final_frame(schedules[:, -1] - schedules[:, -1, :1])   # [schedule, end]

    end_time: Scalar = (preserved[-1, 1] | mv.t) + 0.53                  # the front's last kink ends the train
    tracks: Vector = stack([worldlines(schedule, directions, stack((mv.scalar([-0.30]), end_time))) for schedule in (preserved, bell)])
    slices: Vector = stack([worldline_events(stack((mv.scalar([-0.18]), end_time - 0.22)), schedule, directions)
                            for schedule in (preserved, bell)])

    # --- checks -------------------------------------------------------------
    # Proper time between kinks: the rear keeps one evenly spaced program in both schedules,
    # and Bell's ships run it on clocks started together at lab t = 0.
    ticks: Scalar = (schedules[:, 1:] - schedules[:, :-1]).norm()          # [schedule, step - 1, end]
    np.testing.assert_allclose(ticks[0, :, 0].to_array(), dt, atol=GEOMETRY_ATOL)
    np.testing.assert_allclose(ticks[1].to_array(), dt, atol=GEOMETRY_ATOL)
    np.testing.assert_allclose((bell[0] | mv.t).to_array(), 0.0, atol=GEOMETRY_ATOL)
    # Both ships end at rest in the final frame: the strain-preserving rope keeps its length,
    # Bell's is stretched by gamma. In the lab it is the other way round.
    np.testing.assert_allclose((-(final_frame(directions[-1]) | mv.x)).to_array(), 0.0, atol=GEOMETRY_ATOL)
    np.testing.assert_allclose((-(final_events[:, 1] | mv.x)).to_array(), [1.0, np.cosh(rapidity)],
                               atol=GEOMETRY_ATOL)
    np.testing.assert_allclose((-((slices[:, 1, 1] - slices[:, 1, 0]) | mv.x)).to_array(),
                               [1.0 / np.cosh(rapidity), 1.0], atol=GEOMETRY_ATOL)
    along = worldline_events(times(*np.linspace(-0.3, 3.0, 300)), bell, directions)
    np.testing.assert_allclose((-((along[:, 1] - along[:, 0]) | mv.x)).to_array(), 1.0, atol=GEOMETRY_ATOL)

    return tracks, schedules, slices, final_events


if __name__ == "__main__":
    from examples.animation import save_figure
    from examples.relativity.impulse import render

    save_figure(render.draw_impulse(*impulse()), "relativistic_rod_impulse")
    save_figure(render.draw_ladder(*ladder()), "ladder_paradox")
    save_figure(render.draw_spaceships(*spaceships()), "bell_spaceships")
