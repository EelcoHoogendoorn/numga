"""Spacetime worldline geometry models strain-preserving relativistic impulses and endpoint histories.

In the symmetric frame where a rod reverses velocity simultaneously across its length, incoming and outgoing worldlines have identical proper spacing without elastic mismatch. Boosting that reversal tilts the simultaneous impulse line onto the spacelike hyperbolic bisector of the incoming and outgoing four-velocities, defining a finite velocity step that preserves proper spacing across inertial frames; see M. Fayngold, arXiv:0712.3891 (2007; revised 2020), Section 1(III).
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext, concatenate, stack

ga = Algebra("x+t-")
mv = NumpyContext(ga).multivector

Scalar = ga.gatype.scalar()                         # Scalar+
Point = ga.gatype.antivector()                      # Antivector-
Rotor = ga.gatype.rotor()                           # Rotor+
PointMap = ga.gatype((Point, Point))                # Point- <- Point-

time = -mv.t.dual()                                 # [] Point-
space = mv.x.dual()                                 # [] Point-
rod_ends = np.array([0, 1])


def boost_map(rapidity: np.ndarray) -> PointMap:
    """Hyperbolic turns in the spacetime plane build the Lorentz boost map (`[...] Point- <- Point-`)."""
    rotor = (mv.xt * (rapidity / 2)).exp().normalized()                         # [...] Rotor+
    return rotor >> Point                                                       # [...] Point- <- Point-


def strain_preserving_kinks(before: Point, after: Point) -> Point:
    """Spacelike bisector separation places the two rod-end kinks (`[2] Point-`) at unit proper length."""
    separation = ((before + after) * mv.tx) / (1 + (before | after))            # [] Point-
    return rod_ends * separation                                                # [2] Point-


def velocity_step(before_rapidity: float, after_rapidity: float) -> tuple[Point, Point]:
    """A single strain-preserving velocity jump places the endpoint kinks (`[2] Point-`) between incoming and outgoing tangents (`[2] Point-`)."""
    directions = boost_map(np.array([before_rapidity, after_rapidity]))(time)   # [2] Point-
    return strain_preserving_kinks(directions[0], directions[1]), directions


def impulse_train(final_rapidity: float, n_steps: int, proper_step: float) -> tuple[Point, Point]:
    """Repeated application of one boost and proper-time shift generates kink events (`[n_steps, 2] Point-`) and segment tangents (`[n_steps + 1] Point-`)."""
    step_boost = boost_map(np.asarray(final_rapidity / n_steps))                # [] Point- <- Point-
    before = time                                                               # [] Point-
    after = step_boost(before)                                                  # [] Point-
    events = strain_preserving_kinks(before, after)                             # [2] Point-
    offset = proper_step * after                                                # [] Point-

    event_history = [events]
    direction_history = [before, after]
    for _ in range(n_steps - 1):
        events = step_boost(events) + offset                                    # [2] Point-
        after = step_boost(after)                                               # [] Point-
        event_history.append(events)
        direction_history.append(after)
    return stack(event_history), stack(direction_history)


def coasting_events(events: Point, direction: Point, observer_times: np.ndarray) -> Point:
    """Straight worldlines through `events` along `direction` cross each observer time (`[...] Point-`)."""
    return events + direction * ((observer_times - (mv.t & events)) / (mv.t & direction))  # [...] Point-


def worldline_events(observer_times: np.ndarray, kinks: Point, directions: Point) -> Point:
    """Piecewise-linear endpoint worldlines cross each observer time (`[..., 2] Point-`)."""
    rates = directions / (mv.t & directions)                                    # [...] Point-
    elapsed = observer_times[..., None, None] - (mv.t & kinks)                  # [...] Scalar+
    jumps = rates[1:] - rates[:-1]                                              # [...] Point-
    return kinks[0] + rates[0] * elapsed[..., 0, :] + (jumps[:, None] * elapsed.clip(0.0, np.inf)).sum(axis=-2)


def worldline_polylines(kinks: Point, directions: Point, time_bounds: np.ndarray) -> Point:
    """Endpoint polylines connect the outer observer `time_bounds` through all intermediate `kinks` (`[..., 2] Point-`)."""
    outer_ends = worldline_events(time_bounds, kinks, directions)               # [2, 2] Point-
    return concatenate((outer_ends[:1], kinks, outer_ends[1:]))                 # [..., 2] Point-


def ringing_lengths(observer_times: np.ndarray, damping_ratio: float, natural_frequency: float) -> np.ndarray:
    """An underdamped harmonic mode relaxes from initial compressed length 0.6 toward unit rest length."""
    damping = damping_ratio * natural_frequency
    frequency = natural_frequency * np.sqrt(1.0 - damping_ratio**2)
    return 1.0 - 0.4 * np.exp(-damping * observer_times) * (
        np.cos(frequency * observer_times) + damping / frequency * np.sin(frequency * observer_times)
    )
