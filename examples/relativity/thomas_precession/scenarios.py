"""Scenes for Thomas precession: a rocket's two boosts, spins on circular orbits at three speeds, a
gyroscope bouncing in a square box, and a muon in a storage ring with its magic Lorentz factor."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import numpy as np

from numga import concatenate, stack
from examples.relativity.thomas_precession import core

mv = core.mv
duration_ms = 40
# The rocket's two equal boosts, their rapidity swelling and shrinking between two values over the
# loop, and the samples along each side of the triangle of velocities.
n_boost_frames = 120
MIN_RAPIDITY, MAX_RAPIDITY = 0.8, 2.0
n_edge_samples = 32
# Lorentz factors of the circular orbits, their steps around, and the steps between frames.
orbit_gammas = np.array([1.25, 1.5, 1.75])
n_orbit_steps = 450
orbit_stride = 4
# The box's half width, the gyroscope's momentum per unit mass along each axis, the frames of each
# flight and the steps of each bounce, and the round trips shown.
box_half_width = 1.0
box_momentum = 0.8
n_flight_frames = 13
n_bounce_steps = 12
n_loops = 3
# The gyroscope's spin: the patch of the plane it turns in, yz.
gyroscope_spin = mv.yz.dual()                                                   # [] Spin-
# The storage ring: the muon's anomaly exaggerated so the spin laps the momentum in a few turns, the
# magnetic field, the animation's ring purely magnetic, its turns, steps per turn and between frames.
muon_anomaly = 0.1
ring_magnetic = 1.0
ring_electric = 0.0
n_ring_turns = 3
n_ring_turn_steps = 300
ring_stride = 6
n_ring_samples = 128
# The muon's spin, turning in zx, across its first momentum along y.
muon_spin = mv.zx.dual()                                                        # [] Spin-
# The electric fields swept at two Lorentz factors, the magic one last, over a fixed time in the lab.
electric_fields = np.linspace(-0.3, 0.3, 7)
sweep_gammas = np.array([2.0, np.sqrt(1 + 1 / muon_anomaly)])
sweep_lab_duration = 10.0
n_sweep_steps = 200


# --- math -----------------------------------------------------------------------------
def two_boosts(rapidities: np.ndarray, fractions: np.ndarray) -> Iterator[tuple[core.Antivector, core.Antivector, core.Antivector]]:
    """For each rapidity: the triangle of velocities a rocket passes through, boosting along its own x
    and then its own y, and straight back to rest; its corners; and the rocket's x axis at each corner,
    as the observer at rest reads it."""
    for rapidity in rapidities:
        first = core.boost(mv.x.dual(), rapidity)                               # [] Rotor+
        rocket = first * core.boost(mv.y.dual(), rapidity)                      # [] Rotor+
        corner, velocity = first >> core.time, rocket >> core.time              # [] Antivector-, Antivector-
        loop = concatenate([core.geodesic(core.time, corner, fractions), core.geodesic(corner, velocity, fractions),
                            core.geodesic(velocity, core.time, fractions)])     # [3 * samples] Antivector-
        # The rocket's x axis at each corner, after the pure boost back to rest.
        axes = stack([mv.x.dual(), core.rest_boost(corner).reverse() >> (first >> mv.x.dual()),
                      core.rest_boost(velocity).reverse() >> (rocket >> mv.x.dual())])   # [3] Antivector-
        yield loop, stack([core.time, corner, velocity]), axes


def wigner_turn(rapidity: np.ndarray) -> core.Rotor:
    """The turn a rocket's two equal boosts along its own x and y leave behind: their product after
    the straight boost from rest to the same velocity is undone."""
    rocket = core.boost(mv.x.dual(), rapidity) * core.boost(mv.y.dual(), rapidity)   # [...] Rotor+
    return core.rest_boost(rocket >> core.time).reverse() * rocket              # [...] Rotor+


def orbit_velocities(gammas: np.ndarray, n_steps: int) -> core.Antivector:
    """Velocities around circular orbits at each Lorentz factor, counterclockwise from moving along y."""
    phases = np.linspace(0, 2 * np.pi, n_steps + 1)
    directions = (mv.xy * (-phases[:, None] / 2)).exp() >> mv.y.dual()          # [steps + 1, 1] Antivector-
    return core.boost(directions, np.arccosh(gammas)) >> core.time              # [steps + 1, orbits] Antivector-


def orbits(velocities: core.Antivector, spin: core.Spin, stride: int) -> Iterator[tuple[core.Antivector, core.Spin]]:
    """Every `stride`-th velocity along the orbits, with the carried spin as the observer at rest reads it."""
    for velocity, transported in islice(zip(velocities, core.carried(velocities, spin)), None, None, stride):
        yield velocity, core.lab_view(velocity, transported)


def box_path(half_width: float, momentum: float, n_flight: int, n_bounce: int) -> tuple[core.Antivector, core.Antivector]:
    """One round trip of a particle bouncing along a diamond between the midpoints of a square box's
    walls: its positions and velocities. In flight the velocity holds; at a wall the momentum along
    the wall holds and the momentum across it reverses."""
    # Quarter turns from one before the first wall to one past the last.
    quarters = (mv.xy * (-np.arange(-1, 5) * np.pi / 4)).exp()                  # [6] Rotor+
    walls = quarters >> (half_width * mv.x.dual())                              # [6] Antivector-
    corners = quarters >> (momentum * (mv.x.dual() + mv.y.dual()))              # [6] Antivector-
    # Each flight runs from the wall before to its wall; each bounce from its momentum to the next.
    starts, ends = walls[:-2], walls[1:-1]                                      # [4] Antivector-, Antivector-
    flights = starts + (ends - starts) * np.linspace(0, 1, n_flight)[:, None]   # [flight, 4] Antivector-
    before, after = corners[1:-1], corners[2:]                                  # [4] Antivector-, Antivector-
    bounces = core.on_hyperboloid(before + (after - before) * np.linspace(0, 1, n_bounce)[:, None])   # [bounce, 4] Antivector-
    flight_velocities = core.on_hyperboloid(before).broadcast_to((n_flight, 4))   # [flight, 4] Antivector-
    positions = concatenate([part[:, wall] for wall in range(4) for part in (flights, ends.broadcast_to((n_bounce, 4)))])
    velocities = concatenate([part[:, wall] for wall in range(4) for part in (flight_velocities, bounces)])
    return positions, velocities                                                # [frames] Antivector-


def bouncing(positions: core.Antivector, velocities: core.Antivector, spin: core.Spin) -> Iterator[tuple[core.Antivector, core.Spin]]:
    """Each position, with the carried spin as the observer at rest reads it."""
    for position, velocity, transported in zip(positions, velocities, core.carried(velocities, spin)):
        yield position, core.lab_view(velocity, transported)


def ring_start(gamma: np.ndarray, magnetic: float, electric: np.ndarray) -> tuple[core.Antivector, core.Rotor]:
    """A particle on the circular orbit at Lorentz factor `gamma`, at x and moving along y: its event,
    and its frame, the pure boost from rest, so that `muon_spin`, turning in zx, lies along the
    momentum."""
    event = mv.x.dual() * core.ring_radius(gamma, magnetic, electric)           # [...] Antivector-
    frame = core.boost(mv.y.dual(), np.arccosh(gamma))                          # [...] Rotor+
    return event, frame


def ring(gamma: float, magnetic: float, electric: float, anomaly: float, n_turns: int, n_turn_steps: int,
         stride: int) -> Iterator[tuple[core.Antivector, core.Antivector, core.Spin]]:
    """Every `stride`-th step of a particle's turns around the ring: its event, velocity, and spin as
    the observer at rest reads it."""
    radius = core.ring_radius(gamma, magnetic, electric)
    proper_period = 2 * np.pi * radius / np.sqrt(gamma**2 - 1)
    event, frame = ring_start(gamma, magnetic, electric)
    states = core.evolve(event, frame, magnetic, electric, anomaly, proper_period / n_turn_steps)
    for event, frame in islice(states, 0, n_turns * n_turn_steps + 1, stride):
        velocity = frame >> core.time                                           # [] Antivector-
        yield event, velocity, core.lab_view(velocity, frame >> muon_spin)


def anomalous_rates(gammas: np.ndarray, magnetic: float, electric: np.ndarray, anomaly: float,
                    lab_duration: float, n_steps: int) -> core.Scalar:
    """The rate at which the spin turns away from the momentum, over the anomaly times the magnetic
    field, at each Lorentz factor `[gammas, 1]` and electric field `[fields]`: the angle reached after
    `lab_duration` of the observer's time."""
    event, frame = ring_start(gammas, magnetic, electric)
    proper_step = lab_duration / (gammas * n_steps)
    *_, (event, frame) = islice(core.evolve(event, frame, magnetic, electric, anomaly, proper_step), n_steps + 1)
    angle = core.anomalous_angle(frame >> core.time, frame >> muon_spin)   # [gammas, fields] Scalar+
    return angle / (anomaly * magnetic * lab_duration)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.relativity.thomas_precession import render

    # The swell alone; the animation plays it forward and back.
    swell = (1 - np.cos(np.linspace(0, np.pi, n_boost_frames // 2 + 1))) / 2
    rapidities = MIN_RAPIDITY + (MAX_RAPIDITY - MIN_RAPIDITY) * swell
    fractions = np.linspace(0, 1, n_edge_samples)
    save_animation(render.animate_two_boosts(two_boosts(rapidities, fractions)), "thomas_two_boosts", duration_ms,
                   forward_then_back=True)
    velocities = orbit_velocities(orbit_gammas, n_orbit_steps)
    # The gyroscope's spin in each orbit's first frame.
    start_spin = core.rest_boost(velocities[0]) >> gyroscope_spin
    save_animation(render.animate_orbits(velocities, orbits(velocities, start_spin, orbit_stride)), "thomas_orbits", duration_ms)
    positions, path = box_path(box_half_width, box_momentum, n_flight_frames, n_bounce_steps)
    positions, path = concatenate([positions] * n_loops), concatenate([path] * n_loops)
    box_spin = core.rest_boost(path[0]) >> gyroscope_spin
    walls = (mv.xy * (-np.arange(4) * np.pi / 4)).exp() >> (box_half_width * mv.x.dual())
    save_animation(render.animate_box(box_half_width, walls, bouncing(positions, path, box_spin)), "thomas_box", duration_ms)
    magic = np.sqrt(1 + 1 / muon_anomaly)
    radius = core.ring_radius(magic, ring_magnetic, ring_electric)
    outline = (mv.xy * (-np.linspace(0, 2 * np.pi, n_ring_samples) / 2)).exp() >> (radius * mv.x.dual())
    states = ring(magic, ring_magnetic, ring_electric, muon_anomaly, n_ring_turns, n_ring_turn_steps, ring_stride)
    save_animation(render.animate_ring(outline, states), "thomas_storage_ring", duration_ms)
    ratios = anomalous_rates(sweep_gammas[:, None], ring_magnetic, electric_fields, muon_anomaly, sweep_lab_duration, n_sweep_steps)
    save_figure(render.draw_magic(electric_fields / ring_magnetic, ratios), "thomas_magic")

    # checks
    # Two boosts leave a pure turn, by the closed form for equal perpendicular boosts, and by the area
    # of their triangle of velocities: half a turn less twice the angle at rest.
    turn = wigner_turn(rapidities)                                              # [frames] Rotor+
    np.testing.assert_allclose(((turn >> core.time) - core.time).kernel, 0.0, atol=1e-12)
    angle = 2 * turn.log().scalar_product(mv.xy).to_array()
    np.testing.assert_allclose(angle, -2 * np.arctan(np.tanh(rapidities / 2) ** 2), atol=1e-12)
    final = core.boost(mv.x.dual(), rapidities) * core.boost(mv.y.dual(), rapidities) >> core.time
    x, y = (mv.x & final).to_array(), (mv.y & final).to_array()
    np.testing.assert_allclose(angle, -(np.pi / 2 - 2 * np.arctan2(y, x)), atol=1e-12)
    # Around a circular orbit the spin turns back by the cap's area, 2 pi (gamma - 1).
    end = core.lab_view(velocities[-1], tuple(core.carried(velocities, start_spin))[-1])
    expected = (mv.xy * (np.pi * (orbit_gammas - 1))).exp() >> gyroscope_spin
    np.testing.assert_allclose((end - expected).kernel, 0.0, atol=5e-3)
    # In the box, each round trip turns it back by the area the square of momenta encloses on the
    # hyperboloid.
    square_area = 4 * (2 * box_momentum * np.arcsinh(box_momentum / np.sqrt(1 + box_momentum**2))
                       - np.arctan(box_momentum**2 / np.sqrt(1 + 2 * box_momentum**2)))
    *_, (_, box_end) = bouncing(positions, path, box_spin)
    expected = (mv.xy * (n_loops * square_area / 2)).exp() >> gyroscope_spin
    np.testing.assert_allclose((box_end - expected).kernel, 0.0, atol=5e-2)
    # At the magic Lorentz factor the electric field drops out of the anomalous rate; elsewhere it
    # enters by the Bargmann–Michel–Telegdi formula.
    values = ratios.to_array()
    speed = np.sqrt(1 - 1 / sweep_gammas[0] ** 2)
    np.testing.assert_allclose(values[1], 1.0, atol=1e-3)
    np.testing.assert_allclose(values[0], 1 - (1 - 1 / (muon_anomaly * (sweep_gammas[0] ** 2 - 1))) * speed * electric_fields / ring_magnetic, atol=1e-3)


if __name__ == "__main__":
    main()
