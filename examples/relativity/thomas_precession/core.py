"""Successive Lorentz boosts generate spatial Wigner rotations, velocity-space holonomy, and Bargmann–Michel–Telegdi spin precession in spacetime geometric algebra."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import numpy as np

from numga import NumpyContext, concatenate, stack
from numga.algebras import STA as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                         # Scalar+
Bivector = ga.gatype.bivector()                     # Bivector+
Spin = ga.gatype.antibivector()                     # Antibivector-
Antivector = ga.gatype.antivector()                 # Antivector-
Rotor = ga.gatype.rotor()                           # Rotor+

# Unit future-directed four-velocity of the laboratory observer.
time = -mv.t.dual()                                 # [] Antivector-


# --- math -----------------------------------------------------------------------------
def pure_boost(direction: Antivector, rapidity: np.ndarray) -> Rotor:
    """The spacetime plane spanned by `direction` (`[...] Antivector-`) and `time` exponentiates into a pure boost rotor (`[...] Rotor+`) of `rapidity`."""
    plane = (direction & time).dual()                                           # [...] Bivector+
    return (plane * (rapidity / 2)).exp()                                       # [...] Rotor+


def step_boost(tail: Antivector, head: Antivector) -> Rotor:
    """The geometric ratio `head * tail.inverse()` has a square-root rotor (`[...] Rotor+`) that boosts four-velocity `tail` directly to `head`."""
    return (head * tail.inverse()).square_root()                                # [...] Rotor+


def geodesic_velocities(tail: Antivector, head: Antivector, n_samples: int) -> Antivector:
    """Fractional powers of the pure boost from `tail` to `head` sample `n_samples` four-velocities (`[n_samples] Antivector-`) along their hyperbolic geodesic."""
    fractions = np.linspace(0, 1, n_samples, endpoint=False)                    # [n_samples]
    steps = (step_boost(tail, head).log() * fractions).exp()                    # [n_samples] Rotor+
    return steps >> tail                                                        # [n_samples] Antivector-


def wigner_rotor(rapidity: np.ndarray) -> Rotor:
    """Two perpendicular boosts of equal `rapidity` followed by the direct boost back to rest leave a pure spatial Wigner rotor (`[...] Rotor+`)."""
    rocket_boost = pure_boost(mv.x.dual(), rapidity) * pure_boost(mv.y.dual(), rapidity)   # [...] Rotor+
    final_velocity = rocket_boost >> time                                       # [...] Antivector-
    return step_boost(final_velocity, time) * rocket_boost                      # [...] Rotor+


def two_boost_states(rapidities: np.ndarray, n_samples: int) -> Iterator[tuple[Antivector, Antivector, Antivector]]:
    """Yields the geodesic velocity triangle (`[3 * n_samples] Antivector-`), its three corner velocities (`[3] Antivector-`), and the rest-frame spatial axis (`[3] Antivector-`) at each rapidity."""
    x_axis = mv.x.dual()                                                        # [] Antivector-
    y_axis = mv.y.dual()                                                        # [] Antivector-
    for rapidity in rapidities:
        first_boost = pure_boost(x_axis, rapidity)                              # [] Rotor+
        rocket_boost = first_boost * pure_boost(y_axis, rapidity)               # [] Rotor+
        corner_velocity = first_boost >> time                                   # [] Antivector-
        final_velocity = rocket_boost >> time                                   # [] Antivector-
        loop = concatenate([
            geodesic_velocities(time, corner_velocity, n_samples),
            geodesic_velocities(corner_velocity, final_velocity, n_samples),
            geodesic_velocities(final_velocity, time, n_samples),
        ])                                                                      # [3 * n_samples] Antivector-
        corner_unboost = step_boost(corner_velocity, time)                      # [] Rotor+
        final_unboost = step_boost(final_velocity, time)                        # [] Rotor+
        axes = stack([
            x_axis,
            corner_unboost >> (first_boost >> x_axis),
            final_unboost >> (rocket_boost >> x_axis),
        ])                                                                      # [3] Antivector-
        yield loop, stack([time, corner_velocity, final_velocity]), axes


def lab_spin(velocity: Antivector, spin: Spin) -> Spin:
    """The direct boost from `velocity` (`[...] Antivector-`) back to `time` carries `spin` into the laboratory rest slice (`[...] Spin-`)."""
    boost_to_rest = step_boost(velocity, time)                                  # [...] Rotor+
    return boost_to_rest >> spin                                                # [...] Spin-


def carried_spins(velocities: Antivector, spin: Spin) -> Iterator[Spin]:
    """Yields `spin` (`[...] Spin-`) transported without torque along `velocities` (`[n_steps + 1, ...] Antivector-`) by successive pure boosts."""
    yield spin
    for tail, head in zip(velocities[:-1], velocities[1:]):
        step = step_boost(tail, head)                                           # [...] Rotor+
        spin = step >> spin                                                     # [...] Spin-
        yield spin


def orbit_velocities(gammas: np.ndarray, n_steps: int) -> Antivector:
    """Counterclockwise spatial directions boosted to Lorentz factors `gammas` trace circular velocity loops (`[n_steps + 1, n_orbits] Antivector-`)."""
    n_orbits = len(gammas)
    phases = np.linspace(0, 2 * np.pi, n_steps + 1)                             # [n_steps + 1]
    turns = (mv.xy * (-phases[:, None] / 2)).exp()                              # [n_steps + 1, 1] Rotor+
    directions = turns >> mv.y.dual()                                           # [n_steps + 1, 1] Antivector-
    boosts = pure_boost(directions, np.arccosh(gammas))                         # [n_steps + 1, n_orbits] Rotor+
    return boosts >> time                                                       # [n_steps + 1, n_orbits] Antivector-


def orbit_states(velocities: Antivector, spin: Spin, n_stride: int) -> Iterator[tuple[Antivector, Spin]]:
    """Yields every `n_stride`-th orbital four-velocity and laboratory-read spin (`[n_orbits] Antivector-`, `[n_orbits] Spin-`)."""
    for velocity, carried_spin in islice(zip(velocities, carried_spins(velocities, spin)), None, None, n_stride):
        yield velocity, lab_spin(velocity, carried_spin)


def hyperboloid_velocity(spatial: Antivector) -> Antivector:
    """Adding the unit-mass energy component along `time` lifts spatial four-momentum `spatial` (`[...] Antivector-`) onto the unit velocity hyperboloid."""
    return spatial + time * (1 + spatial.squared()).square_root()               # [...] Antivector-


def box_trajectory(half_width: float, momentum: float, n_flight: int, n_bounce: int) -> tuple[Antivector, Antivector]:
    """Straight wall-to-wall flights and wall bounces build one diamond round trip of positions and four-velocities (`[4 * (n_flight + n_bounce)] Antivector-`)."""
    quarter_turns = (mv.xy * (-np.arange(-1, 5) * np.pi / 4)).exp()             # [6] Rotor+
    walls = quarter_turns >> (half_width * mv.x.dual())                         # [6] Antivector-
    corner_momenta = quarter_turns >> (momentum * (mv.x.dual() + mv.y.dual()))  # [6] Antivector-
    wall_starts, wall_ends = walls[:-2], walls[1:-1]                            # [4] Antivector-, Antivector-
    flights = wall_starts + (wall_ends - wall_starts) * np.linspace(0, 1, n_flight)[:, None]   # [n_flight, 4] Antivector-
    incoming_momenta, outgoing_momenta = corner_momenta[1:-1], corner_momenta[2:]              # [4] Antivector-, Antivector-
    bounces = hyperboloid_velocity(
        incoming_momenta + (outgoing_momenta - incoming_momenta) * np.linspace(0, 1, n_bounce)[:, None]
    )                                                                           # [n_bounce, 4] Antivector-
    flight_velocities = hyperboloid_velocity(incoming_momenta).broadcast_to((n_flight, 4))     # [n_flight, 4] Antivector-
    positions = concatenate([
        segment[:, wall] for wall in range(4) for segment in (flights, wall_ends.broadcast_to((n_bounce, 4)))
    ])                                                                          # [4 * (n_flight + n_bounce)] Antivector-
    velocities = concatenate([
        segment[:, wall] for wall in range(4) for segment in (flight_velocities, bounces)
    ])                                                                          # [4 * (n_flight + n_bounce)] Antivector-
    return positions, velocities


def box_states(positions: Antivector, velocities: Antivector, spin: Spin) -> Iterator[tuple[Antivector, Spin]]:
    """Yields each particle position and laboratory-read spin (`[] Antivector-`, `[] Spin-`) along the box trajectory."""
    for position, velocity, carried_spin in zip(positions, velocities, carried_spins(velocities, spin)):
        yield position, lab_spin(velocity, carried_spin)


def bmt_generator(field: Bivector, velocity: Antivector, anomaly: float) -> Bivector:
    """Electromagnetic `field` (`[...] Bivector+`) and magnetic `anomaly` combine at `velocity` into the Bargmann–Michel–Telegdi frame rate (`[...] Bivector+`)."""
    force = field & velocity                                                    # [...] Vector+
    # The particle's instantaneous rest slice pairs to unity with its four-velocity.
    rest_slice = -velocity.dual()                                               # [...] Vector+
    # On `velocity` itself `(rest_slice ^ force)` matches `field`, cancelling `anomaly` from the worldline acceleration.
    return (1 + anomaly) * field - anomaly * (rest_slice ^ force)               # [...] Bivector+


def ring_field(event: Antivector, magnetic: float, electric: np.ndarray) -> Bivector:
    """Vertical `magnetic` bending and radial `electric` focusing form the storage-ring field bivector (`[...] Bivector+`) at `event`."""
    position = event - (mv.t & event) * time                                    # [...] Antivector-
    outward = position / position.antinorm()                                    # [...] Antivector-
    return -magnetic * mv.xy + electric * (outward & time).dual()               # [...] Bivector+


def bmt_step(
    event: Antivector,
    frame: Rotor,
    magnetic: float,
    electric: np.ndarray,
    anomaly: float,
    proper_step: np.ndarray,
) -> tuple[Antivector, Rotor]:
    """Midpoint proper-time integration advances `event` (`[...] Antivector-`) and `frame` (`[...] Rotor+`) by `proper_step` in the storage-ring field."""
    velocity = frame >> time                                                    # [...] Antivector-
    rate = bmt_generator(ring_field(event, magnetic, electric), velocity, anomaly)   # [...] Bivector+
    halfway_frame = (rate * (proper_step / 4)).exp() * frame                    # [...] Rotor+
    halfway_event = event + velocity * (proper_step / 2)                        # [...] Antivector-
    halfway_velocity = halfway_frame >> time                                    # [...] Antivector-
    halfway_rate = bmt_generator(ring_field(halfway_event, magnetic, electric), halfway_velocity, anomaly)   # [...] Bivector+
    next_event = event + halfway_velocity * proper_step                         # [...] Antivector-
    next_frame = (halfway_rate * (proper_step / 2)).exp() * frame               # [...] Rotor+
    return next_event, next_frame


def bmt_trajectory(
    event: Antivector,
    frame: Rotor,
    magnetic: float,
    electric: np.ndarray,
    anomaly: float,
    proper_step: np.ndarray,
) -> Iterator[tuple[Antivector, Rotor]]:
    """Yields successive particle events and Lorentz frames (`[...] Antivector-`, `[...] Rotor+`) integrated by `proper_step`."""
    while True:
        yield event, frame
        event, frame = bmt_step(event, frame, magnetic, electric, anomaly, proper_step)


def ring_radius(gamma: np.ndarray, magnetic: float, electric: np.ndarray) -> np.ndarray:
    """Radial force balance between `magnetic` bending and `electric` focusing sets the circular orbit radius at Lorentz factor `gamma`."""
    speed = np.sqrt(1 - 1 / gamma**2)
    return gamma * speed**2 / (speed * magnetic - electric)


def ring_start(gamma: np.ndarray, magnetic: float, electric: np.ndarray) -> tuple[Antivector, Rotor]:
    """Initial position on the horizontal axis and pure boost along the tangent initialize a circular storage-ring orbit (`[...] Antivector-`, `[...] Rotor+`)."""
    event = mv.x.dual() * ring_radius(gamma, magnetic, electric)                # [...] Antivector-
    frame = pure_boost(mv.y.dual(), np.arccosh(gamma))                          # [...] Rotor+
    return event, frame


def ring_states(
    gamma: float,
    magnetic: float,
    electric: float,
    anomaly: float,
    muon_spin: Spin,
    n_turns: int,
    n_turn_steps: int,
    n_stride: int,
) -> Iterator[tuple[Antivector, Antivector, Spin]]:
    """Yields every `n_stride`-th orbital event, four-velocity, and laboratory-read spin (`[] Antivector-`, `[] Antivector-`, `[] Spin-`) across `n_turns` revolutions."""
    radius = ring_radius(gamma, magnetic, electric)
    proper_period = 2 * np.pi * radius / np.sqrt(gamma**2 - 1)
    event, frame = ring_start(gamma, magnetic, electric)
    states = bmt_trajectory(event, frame, magnetic, electric, anomaly, proper_period / n_turn_steps)
    for event, frame in islice(states, 0, n_turns * n_turn_steps + 1, n_stride):
        velocity = frame >> time                                                # [] Antivector-
        yield event, velocity, lab_spin(velocity, frame >> muon_spin)


def anomalous_angle(velocity: Antivector, spin: Spin) -> Scalar:
    """Logarithmic comparison in the orbital plane measures the laboratory angle (`[...] Scalar+`) by which `spin` leads the forward momentum direction."""
    motion = velocity - (mv.t & velocity) * time                                # [...] Antivector-
    direction = motion / motion.antinorm()                                      # [...] Antivector-
    # Intrinsic spin patch aligned with the orbital tangent in the horizontal plane.
    forward_spin = direction.commutator(mv.xy).regressive(mv.z.dual())          # [...] Spin-
    return (forward_spin.inverse() * lab_spin(velocity, spin)).log() & mv.xy.dual()   # [...] Scalar+


def anomalous_rates(
    gammas: np.ndarray,
    magnetic: float,
    electric: np.ndarray,
    anomaly: float,
    muon_spin: Spin,
    lab_duration: float,
    n_steps: int,
) -> Scalar:
    """Proper-time integration over `lab_duration` returns the normalized anomalous precession rate (`[n_gammas, n_fields] Scalar+`) across Lorentz factors and electric fields."""
    n_gammas = len(gammas)
    n_fields = len(electric)
    event, frame = ring_start(gammas[:, None], magnetic, electric)              # [n_gammas, n_fields] Antivector-, Rotor+
    proper_step = lab_duration / (gammas[:, None] * n_steps)                    # [n_gammas, 1]
    *_, (event, frame) = islice(
        bmt_trajectory(event, frame, magnetic, electric, anomaly, proper_step),
        n_steps + 1,
    )
    angles = anomalous_angle(frame >> time, frame >> muon_spin)                 # [n_gammas, n_fields] Scalar+
    return angles / (anomaly * magnetic * lab_duration)                         # [n_gammas, n_fields] Scalar+
