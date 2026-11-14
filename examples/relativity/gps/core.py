"""Satellite navigation recovers a spacetime reception event and its uncertainty from arriving signal timestamps.

Projective spacetime geometric algebra models events as intrinsic antivectors and measurement slices as extrinsic vectors. Each transmission event defines its light cone as a quadratic duality map from events to hyperplanes, `emitted | (emitted & Event)`, whose pairing with an event vanishes along null rays. Linearizing those intervals around a reception event builds a positive-definite information duality map whose inverse jointly captures spatial position and receiver-clock uncertainty.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext, stack, where

ga = Algebra("x+y+z+t-w0")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                       # Scalar+
Event = ga.gatype.antivector()                                                    # Antivector-
Slice = ga.gatype.vector()                                                        # Vector+
Readout = Slice.derive.nondegenerate                                              # Vector+
Step = Readout.derive.dual                                                        # Antivector-
Cone = ga.gatype((Slice, Event))                                                  # Slice+ <- Event-
Information = ga.gatype((Readout, Step))                                          # Readout+ <- Step-
Covariance = ga.gatype((Step, Readout))                                           # Step- <- Readout+
Orbit = ga.gatype.bivector()                                                      # Bivector+
# Purely spatial readout planes spanning `x`, `y`, and `z`.
Spatial = Slice.derive.within(Algebra("x+y+z+"))                                  # Vector+

origin = mv.w.dual()                                                              # [] Event-
across = mv.x.dual()                                                              # [] Event-
time = -mv.t.dual()                                                               # [] Event-


# --- math -----------------------------------------------------------------------------
def orbit_positions(planes: Orbit, phases: np.ndarray, radius: float,
                    angular_speed: float, times: Scalar) -> Event:
    """Circular satellite trajectories in `planes` (`[n_satellites] Bivector+`) at `phases`, `radius`, and
    `angular_speed` take coordinate `times` (`[...] Scalar+`) to `[..., n_satellites] Event-`."""
    rotations = (-planes * ((phases + angular_speed * times) / 2)).exp()          # [..., n_satellites] Rotor+
    return origin + (rotations >> across) * radius + time * times                 # [..., n_satellites] Event-


def spatial_distance(first: Event, second: Event) -> Scalar:
    """Rest-frame spatial distance (`[...] Scalar+`) between unit-weight events `first` and `second`
    combines the squared coordinate-time separation with minus the spacetime join's squared norm."""
    return ((mv.t & (first - second)).squared() - (first & second).scalar_norm_squared()).square_root()


def emitted_events(receiver: Event, planes: Orbit, phases: np.ndarray, radius: float,
                   angular_speed: float, n_steps: int) -> Event:
    """Fixed-point iteration over `n_steps` light-time updates locates the retarded satellite
    transmission events (`[n_satellites] Event-`) whose signals arrive together at `receiver`."""
    reception_time = mv.t & receiver                                              # [] Scalar+
    emission_times = reception_time                                               # [] Scalar+
    for _ in range(n_steps):
        satellites = orbit_positions(planes, phases, radius, angular_speed, emission_times)  # [n_satellites] Event-
        emission_times = reception_time - spatial_distance(receiver, satellites)  # [n_satellites] Scalar+
    return orbit_positions(planes, phases, radius, angular_speed, emission_times)  # [n_satellites] Event-


def light_cones(emitted: Event) -> Cone:
    """Transmission vertices `emitted` (`[..., n_satellites] Event-`) define light-cone quadratic duality maps
    `emitted | (emitted & Event)` (`[..., n_satellites] Slice+ <- Event-`)."""
    return emitted | (emitted & Event)                                            # [..., n_satellites] Slice+ <- Event-


def cone_intersections(cones: Cone) -> Event:
    """Pairwise radical planes of four `cones` (`[4] Slice+ <- Event-`) meet in a spacetime line whose
    two intersections (`[2] Event-`) with `cones[0]` solve the four-satellite light-cone system."""
    # Subtracting two light cones cancels the quadratic Lorentz part, leaving the radical hyperplane paired with `w`.
    difference = cones[1:] - cones[0]                                             # [3] Slice+ <- Event-
    at_origin = difference(origin)                                                # [3] Slice+
    radicals = at_origin - mv.w * ((at_origin & origin) / 2)                      # [3] Slice+
    axis = radicals[0] ^ radicals[1] ^ radicals[2]                                # [] Trivector+

    # Intersecting the radical line with two time slices yields unit-weight event points spanning the line.
    base_cut = axis ^ mv.t
    base = base_cut / (mv.w & base_cut)                                           # [] Event-
    tip_cut = axis ^ (mv.t - mv.w)
    tip = tip_cut / (mv.w & tip_cut)                                              # [] Event-
    direction = tip - base                                                        # [] Event-

    # Solving the quadratic interval along `base + parameter * direction` locates the two null intersections.
    quadratic = cones[0](direction) & direction                                   # [] Scalar+
    linear = cones[0](direction) & base                                           # [] Scalar+
    constant = cones[0](base) & base                                              # [] Scalar+
    discriminant_root = (linear.squared() - quadratic * constant).square_root()   # [] Scalar+
    parameters = stack((-linear - discriminant_root, -linear + discriminant_root)) / quadratic  # [2] Scalar+
    return base + direction * parameters                                          # [2] Event-


def causal_reception(candidates: Event, emitted: Event) -> Event:
    """Causal ordering selects from `candidates` (`[2] Event-`) the reception event (`[] Event-`)
    occurring after every transmission in `emitted` (`[n_signals] Event-`)."""
    after = (mv.t & (candidates[:, None] - emitted)) > 0                          # [2, n_signals]
    return where(after[0].all(), candidates[0], candidates[1])                    # [] Event-


def cone_intervals(cones: Cone, emitted: Event, receiver: Event) -> tuple[Scalar, Readout, Scalar]:
    """Pairing `cones` (`[..., n_satellites] Slice+ <- Event-`) with `receiver` yields the squared
    intervals (`[..., n_satellites] Scalar+`), displacement readouts (`[..., n_satellites] Readout+`),
    and timestamp error levers (`[..., n_satellites] Scalar+`)."""
    tangent = cones(receiver[..., None])                                          # [..., n_satellites] Slice+
    # Projecting out `w` retains the nondegenerate spacetime 1-form reading displacements `Step-`.
    readout = 2 * tangent.cast(Readout)                                           # [..., n_satellites] Readout+
    lever = 2 * (mv.t & (receiver[..., None] - emitted))                          # [..., n_satellites] Scalar+
    return tangent & receiver[..., None], readout, lever


def fitted_reception(emitted: Event, initial: Event, sigma: float,
                     n_steps: int) -> tuple[Event, Information, Covariance]:
    """Gauss-Newton iteration over `n_steps` fits the reception event (`[...] Event-`) to noisy
    `emitted` cone vertices with range uncertainty `sigma`, along with `Information` and `Covariance`."""
    cones = light_cones(emitted)                                                  # [..., n_satellites] Slice+ <- Event-
    receiver = initial                                                            # [...] Event-
    for _ in range(n_steps):
        interval, readout, lever = cone_intervals(cones, emitted, receiver)
        weight = 1 / (lever * sigma) ** 2                                         # [..., n_satellites] Scalar+
        information = (weight * readout * (readout & Step)).sum(axis=-1)          # [...] Readout+ <- Step-
        mismatch = (weight * readout * interval).sum(axis=-1)                     # [...] Readout+
        receiver = receiver - information.solve(mismatch)                         # [...] Event-

    _, readout, lever = cone_intervals(cones, emitted, receiver)
    information = (readout * (readout & Step) / (lever * sigma) ** 2).sum(axis=-1)  # [...] Readout+ <- Step-
    covariance = information.inverse()                                            # [...] Step- <- Readout+
    return receiver, information, covariance


def position_semi_axes(covariance: Covariance) -> Event:
    """Principal spatial eigendecomposition of `covariance` (`[...] Step- <- Readout+`) gives the
    one-standard-deviation position ellipsoid semi-axes (`[..., 3] Event-`)."""
    variances, principal = (Spatial & covariance(Spatial)).eigh()                 # [..., 3] Scalar+, Readout+
    return covariance(principal) / variances.square_root()                        # [..., 3] Event-
