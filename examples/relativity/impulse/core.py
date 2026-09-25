"""Strain-preserving impulses, built from a symmetric reversal and its boosts.

An impulse changes a worldline's slope, not its position. In a fixed observer
frame, the body's equal-time length cannot jump while its endpoint worldlines
remain continuous. A sudden velocity change therefore cannot simply replace
the body's length before the change with the relaxed Lorentz-contracted length
of the motion after it. Arbitrary timing across the body generally leaves
compression or stretch, which drives an internal response and may relax to a
different length.
Changing observer alone introduces no strain: Lorentz contraction compares
different equal-time cuts of the same worldlines.

Start in the symmetric frame: the rod reverses direction everywhere at once,
keeping the same speed. Each end's incoming and outgoing worldline segments
are mirror images across that instant, and the measured length is unchanged.
Both motions have the same proper spacing, so there is no strain mismatch to
relax. An open sandwich changes the observer of the entire drawing. The line
joining the kinks tilts, becoming the spacelike hyperbolic bisector of the
motions before and after. In the boosted drawing the observer-frame length
changes continuously as the two ends reach their kinks at different times.

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

Coordinates are time and position along `mv.t` and `mv.x`, in units where the
speed of light and the proper rod length are both one.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext, concatenate, stack


Spacetime = Algebra("t+x-")
ctx = NumpyContext(Spacetime)
mv = ctx.multivector

Scalar = Spacetime.gatype.scalar()
Vector = Spacetime.gatype.vector()
Rotor = Spacetime.gatype.rotor()
VectorMap = Spacetime.gatype((Vector, Vector))

# Allow for the library's approximate bivector exp().
GEOMETRY_ATOL = 1e-9
# The rod's two ends, in units of its proper length.
ENDS: Scalar = mv.scalar([[0.0], [1.0]])  # [end] Scalar


def boost(rapidity: np.ndarray) -> VectorMap:
    """The Lorentz boost adding `rapidity` to every velocity, as an open sandwich on events and directions.

    Normalizing removes exp's tiny scale error before the map is reused.
    """
    rotor: Rotor = (-mv.tx * rapidity / 2).exp().normalized()
    return rotor.sandwich(Vector)


def strain_preserving_kinks(before: Vector, after: Vector) -> Vector:
    """Join unit future timelike directions with a unit-proper-length step.

    The summed directions give the midpoint observer's time axis. Its
    spacelike perpendicular supplies the kink line; scale it so that both
    rest frames measure the unit proper spacing. Anchor the left kink
    at the origin. This construction is specific to the plane of `mv.t` and `mv.x`.
    """
    separation: Vector = ((before + after) * mv.tx) / (1 + (before | after))
    return ENDS * separation


def velocity_step(before_rapidity: float, after_rapidity: float) -> tuple[Vector, Vector]:
    """A prescribed rigid-to-rigid step, anchored at the left endpoint's kink.

    Return the two kink events and unit before/after directions. For unequal
    specified velocities, a single sharp jump with unchanged proper spacing
    uniquely fixes the kink surface up to translation. It is horizontal in
    the rapidity-bisector frame, where the velocity reverses direction at the
    same speed without a length change. Boosts supply the two directions;
    their bisector supplies the kink events.

    This kinematic construction represents no flexible modes or sound wave;
    it does not establish uniqueness for general dynamical histories. The
    proper rod length is 1.
    """
    directions: Vector = boost(np.array([before_rapidity, after_rapidity]))(mv.t)   # [step + 1] Vector
    return strain_preserving_kinks(directions[0], directions[1]), directions


def small_impulses(final_rapidity: float, count: int, dt: float) -> tuple[Vector, Vector]:
    """Compose one fixed boost-plus-translation to generate successive steps.

    Return kink events with batch shape (step, end), and the directions
    before, between and after the steps. dt is PROPER time between left-end
    kinks. The same affine map advances the kink pair on every iteration;
    its linear part advances the worldline directions. Negative steps need
    sufficient spacing to keep the right-end events in chronological order.
    """
    step: VectorMap = boost(final_rapidity / count)
    before: Vector = mv.vector([1.0, 0.0])
    after: Vector = step(before)
    events: Vector = strain_preserving_kinks(before, after)
    offset: Vector = dt * after

    event_history = [events]
    direction_history = [before, after]
    for _ in range(count - 1):
        # Both the boost and the translation stay FIXED throughout the train.
        events = step(events) + offset
        after = step(after)
        event_history.append(events)
        direction_history.append(after)
    return stack(event_history), stack(direction_history)


def coasting(events: Vector, direction: Vector, times: Scalar) -> Vector:
    """The events at observer times `times` on the straight worldlines through `events` along `direction`."""
    return events + direction * ((times - (events | mv.t)) / (direction | mv.t))


def worldline_events(times: Scalar, kinks: Vector, directions: Vector) -> Vector:
    """Each end's event at observer times `times`, on its piecewise straight worldline: [..., end].

    kinks has batch shape (step, end); directions (step + 1), the motion before,
    between and after the steps. Start on the worldline before the first kink,
    then add each velocity jump only after that end has crossed its kink.
    """
    # Velocities per unit observer time.
    rates: Vector = directions / (directions | mv.t)                     # [step + 1] Vector
    elapsed: Scalar = times.reshape(*times.shape, 1, 1) - (kinks | mv.t)  # [..., step, end]
    jumps: Vector = rates[1:] - rates[:-1]                               # [step]
    return kinks[0] + rates[0] * elapsed[..., 0, :] + (jumps[:, None] * elapsed.clip(0.0, np.inf)).sum(axis=-2)


def worldlines(kinks: Vector, directions: Vector, span: Scalar) -> Vector:
    """Each end's worldline as a polyline over the observer times span [limit], through its kinks: [step + 2, end]."""
    ends: Vector = worldline_events(span, kinks, directions)             # [limit, end]
    return concatenate((ends[:1], kinks, ends[1:]))


def ringing_length(times: np.ndarray, damping_ratio: float, natural_frequency: float) -> np.ndarray:
    """Underdamped mode that starts at length 0.6 with zero rate of change and settles to length 1.

    natural_frequency is the undamped angular frequency; both decay and
    oscillation follow from it and the damping ratio.
    """
    damping = damping_ratio * natural_frequency
    frequency = natural_frequency * np.sqrt(1.0 - damping_ratio**2)
    return 1.0 - 0.4 * np.exp(-damping * times) * (
        np.cos(frequency * times) + damping / frequency * np.sin(frequency * times)
    )
