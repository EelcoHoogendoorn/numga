"""Scenes for Thomas precession: a rocket's two boosts, spins on circular orbits at three speeds, a
gyroscope bouncing in a square box, and a muon in a storage ring with its magic Lorentz factor."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import numpy as np

from numga import concatenate, stack
from examples.relativity.thomas_precession import core

mv = core.mv
DURATION_MS = 50
# The rocket's two equal boosts, their rapidity swelling and shrinking between two values over the
# loop, and the samples along each side of the triangle of velocities.
N_BOOST_FRAMES = 96
MIN_RAPIDITY, MAX_RAPIDITY = 0.8, 2.0
N_EDGE_SAMPLES = 32
# Lorentz factors of the circular orbits, their steps around, and the steps between frames.
GAMMAS = np.array([1.25, 1.5, 1.75])
N_ORBIT_STEPS = 180
ORBIT_STRIDE = 2
# The box's half width, the gyroscope's momentum per unit mass along each axis, the frames of each
# flight and the steps of each bounce, and the round trips shown.
HALF_WIDTH = 1.0
MOMENTUM = 0.8
N_FLIGHT_FRAMES = 10
N_BOUNCE_STEPS = 10
N_LOOPS = 3
# The storage ring: the muon's anomaly exaggerated so the spin laps the momentum in a few turns, the
# magnetic field, the animation's ring purely magnetic, its turns, steps per turn and between frames.
ANOMALY = 0.1
MAGNETIC = 1.0
RING_ELECTRIC = 0.0
N_TURNS = 3
N_TURN_STEPS = 120
RING_STRIDE = 3
N_RING_SAMPLES = 128
# The electric fields swept at two Lorentz factors, the magic one last, over a fixed time in the lab.
ELECTRIC_FIELDS = np.linspace(-0.3, 0.3, 7)
SWEEP_GAMMAS = np.array([2.0, np.sqrt(1 + 1 / ANOMALY)])
LAB_DURATION = 10.0
N_SWEEP_STEPS = 200


# --- math -----------------------------------------------------------------------------
def two_boosts(rapidities: np.ndarray, fractions: np.ndarray) -> Iterator[tuple[core.Antivector, core.Antivector, core.Antivector]]:
    """For each rapidity: the triangle of velocities a rocket passes through, boosting along its own x
    and then its own y, and straight back to rest; its corners; and the rocket's x axis at each corner,
    as the observer at rest reads it."""
    for rapidity in rapidities:
        first = core.boost(mv.x.dual(), rapidity)                               # [] Rotor
        rocket = first * core.boost(mv.y.dual(), rapidity)                      # [] Rotor
        corner, velocity = first >> core.time, rocket >> core.time              # [] Antivector each
        loop = concatenate([core.geodesic(core.time, corner, fractions), core.geodesic(corner, velocity, fractions),
                            core.geodesic(velocity, core.time, fractions)])     # [3 * samples] Antivector
        # The rocket's x axis at each corner, after the pure boost back to rest.
        axes = stack([mv.x.dual(), core.rest_boost(corner).reverse() >> (first >> mv.x.dual()),
                      core.rest_boost(velocity).reverse() >> (rocket >> mv.x.dual())])   # [3] Antivector
        yield loop, stack([core.time, corner, velocity]), axes


def wigner_turn(rapidity: np.ndarray) -> core.Rotor:
    """The turn a rocket's two equal boosts along its own x and y leave behind: their product after
    the straight boost from rest to the same velocity is undone."""
    rocket = core.boost(mv.x.dual(), rapidity) * core.boost(mv.y.dual(), rapidity)   # [...] Rotor
    return core.rest_boost(rocket >> core.time).reverse() * rocket              # [...] Rotor


def orbit_velocities(gammas: np.ndarray, n_steps: int) -> core.Antivector:
    """Velocities around circular orbits at each Lorentz factor, counterclockwise from moving along y."""
    phases = np.linspace(0, 2 * np.pi, n_steps + 1)
    directions = (mv.yx * (phases[:, None] / 2)).exp() >> mv.y.dual()          # [steps + 1, 1] Antivector
    return core.boost(directions, np.arccosh(gammas)) >> core.time              # [steps + 1, orbits] Antivector


def orbits(velocities: core.Antivector, spin: core.Bivector, stride: int) -> Iterator[tuple[core.Antivector, core.Bivector]]:
    """Every `stride`-th velocity along the orbits, with the carried spin as the observer at rest reads it."""
    for velocity, transported in islice(zip(velocities, core.carried(velocities, spin)), None, None, stride):
        yield velocity, core.lab_view(velocity, transported)


def box_path(half_width: float, momentum: float, n_flight: int, n_bounce: int) -> tuple[core.Antivector, core.Antivector]:
    """One round trip of a particle bouncing along a diamond between the midpoints of a square box's
    walls: its positions and velocities. In flight the velocity holds; at a wall the momentum along
    the wall holds and the momentum across it reverses."""
    # Quarter turns from one before the first wall to one past the last.
    quarters = (mv.yx * (np.arange(-1, 5) * np.pi / 4)).exp()                  # [6] Rotor
    walls = quarters >> (half_width * mv.x.dual())                              # [6] Antivector
    corners = quarters >> (momentum * (mv.x.dual() + mv.y.dual()))              # [6] Antivector
    # Each flight runs from the wall before to its wall; each bounce from its momentum to the next.
    starts, ends = walls[:-2], walls[1:-1]                                      # [4] Antivector each
    flights = starts + (ends - starts) * np.linspace(0, 1, n_flight)[:, None]   # [flight, 4] Antivector
    before, after = corners[1:-1], corners[2:]                                  # [4] Antivector each
    bounces = core.on_hyperboloid(before + (after - before) * np.linspace(0, 1, n_bounce)[:, None])   # [bounce, 4] Antivector
    flight_velocities = core.on_hyperboloid(before).broadcast_to((n_flight, 4))  # [flight, 4] Antivector
    positions = concatenate([part[:, wall] for wall in range(4) for part in (flights, ends.broadcast_to((n_bounce, 4)))])
    velocities = concatenate([part[:, wall] for wall in range(4) for part in (flight_velocities, bounces)])
    return positions, velocities                                                # [frames] Antivector each


def bouncing(positions: core.Antivector, velocities: core.Antivector, spin: core.Bivector) -> Iterator[tuple[core.Antivector, core.Bivector]]:
    """Each position, with the carried spin as the observer at rest reads it."""
    for position, velocity, transported in zip(positions, velocities, core.carried(velocities, spin)):
        yield position, core.lab_view(velocity, transported)


def ring_start(gamma: np.ndarray, magnetic: float, electric: np.ndarray) -> tuple[core.Antivector, core.Rotor]:
    """A particle on the circular orbit at Lorentz factor `gamma`, at x and moving along y: its event,
    and its frame, the pure boost from rest, so that its spin, the frame's plane zx, lies along the
    momentum."""
    event = mv.x.dual() * core.ring_radius(gamma, magnetic, electric)           # [...] Antivector
    frame = core.boost(mv.y.dual(), np.arccosh(gamma))                          # [...] Rotor
    return event, frame


def ring(gamma: float, magnetic: float, electric: float, anomaly: float, n_turns: int, n_turn_steps: int,
         stride: int) -> Iterator[tuple[core.Antivector, core.Antivector, core.Bivector]]:
    """Every `stride`-th step of a particle's turns around the ring: its event, velocity, and spin as
    the observer at rest reads it."""
    radius = core.ring_radius(gamma, magnetic, electric)
    proper_period = 2 * np.pi * radius / np.sqrt(gamma**2 - 1)
    event, frame = ring_start(gamma, magnetic, electric)
    states = core.evolve(event, frame, magnetic, electric, anomaly, proper_period / n_turn_steps)
    for event, frame in islice(states, 0, n_turns * n_turn_steps + 1, stride):
        velocity = frame >> core.time                                           # [] Antivector
        yield event, velocity, core.lab_view(velocity, frame >> mv.xz)


def anomalous_rates(gammas: np.ndarray, magnetic: float, electric: np.ndarray, anomaly: float,
                    lab_duration: float, n_steps: int) -> core.Scalar:
    """The rate at which the spin turns away from the momentum, over the anomaly times the magnetic
    field, at each Lorentz factor `[gammas, 1]` and electric field `[fields]`: the angle reached after
    `lab_duration` of the observer's time."""
    event, frame = ring_start(gammas, magnetic, electric)
    proper_step = lab_duration / (gammas * n_steps)
    *_, (event, frame) = islice(core.evolve(event, frame, magnetic, electric, anomaly, proper_step), n_steps + 1)
    angle = core.anomalous_angle(frame >> core.time, frame >> mv.xz)            # [gammas, fields] Scalar
    return angle / (anomaly * magnetic * lab_duration)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.relativity.thomas_precession import render

    swell = (1 - np.cos(np.linspace(0, 2 * np.pi, N_BOOST_FRAMES, endpoint=False))) / 2
    rapidities = MIN_RAPIDITY + (MAX_RAPIDITY - MIN_RAPIDITY) * swell
    fractions = np.linspace(0, 1, N_EDGE_SAMPLES)
    save_animation(render.animate_two_boosts(two_boosts(rapidities, fractions)), "thomas_two_boosts", DURATION_MS)
    velocities = orbit_velocities(GAMMAS, N_ORBIT_STEPS)
    # A spin along x, the plane yz in each orbit's first frame.
    start_spin = core.rest_boost(velocities[0]) >> mv.zy
    save_animation(render.animate_orbits(velocities, orbits(velocities, start_spin, ORBIT_STRIDE)), "thomas_orbits", DURATION_MS)
    positions, path = box_path(HALF_WIDTH, MOMENTUM, N_FLIGHT_FRAMES, N_BOUNCE_STEPS)
    positions, path = concatenate([positions] * N_LOOPS), concatenate([path] * N_LOOPS)
    box_spin = core.rest_boost(path[0]) >> mv.zy
    walls = (mv.yx * (np.arange(4) * np.pi / 4)).exp() >> (HALF_WIDTH * mv.x.dual())
    save_animation(render.animate_box(HALF_WIDTH, walls, bouncing(positions, path, box_spin)), "thomas_box", DURATION_MS)
    magic = np.sqrt(1 + 1 / ANOMALY)
    radius = core.ring_radius(magic, MAGNETIC, RING_ELECTRIC)
    outline = (mv.yx * (np.linspace(0, 2 * np.pi, N_RING_SAMPLES) / 2)).exp() >> (radius * mv.x.dual())
    states = ring(magic, MAGNETIC, RING_ELECTRIC, ANOMALY, N_TURNS, N_TURN_STEPS, RING_STRIDE)
    save_animation(render.animate_ring(outline, states), "thomas_storage_ring", DURATION_MS)
    ratios = anomalous_rates(SWEEP_GAMMAS[:, None], MAGNETIC, ELECTRIC_FIELDS, ANOMALY, LAB_DURATION, N_SWEEP_STEPS)
    save_figure(render.draw_magic(ELECTRIC_FIELDS / MAGNETIC, ratios), "thomas_magic")

    # checks
    # Two boosts leave a pure turn, by the closed form for equal perpendicular boosts, and by the area
    # of their triangle of velocities: half a turn less twice the angle at rest.
    turn = wigner_turn(rapidities)                                              # [frames] Rotor
    np.testing.assert_allclose(((turn >> core.time) - core.time).kernel, 0.0, atol=1e-12)
    angle = 2 * turn.log().scalar_product(mv.xy).kernel[..., 0]
    np.testing.assert_allclose(angle, -2 * np.arctan(np.tanh(rapidities / 2) ** 2), atol=1e-12)
    final = core.boost(mv.x.dual(), rapidities) * core.boost(mv.y.dual(), rapidities) >> core.time
    x, y = (mv.x & final).to_array(), (mv.y & final).to_array()
    np.testing.assert_allclose(angle, -(np.pi / 2 - 2 * np.arctan2(y, x)), atol=1e-12)
    # Around a circular orbit the spin turns back by the cap's area, 2 pi (gamma - 1).
    end = core.lab_view(velocities[-1], tuple(core.carried(velocities, start_spin))[-1])
    expected = (mv.yx * (-np.pi * (GAMMAS - 1))).exp() >> mv.zy
    np.testing.assert_allclose((end - expected).kernel, 0.0, atol=5e-3)
    # In the box, each round trip turns it back by the area the square of momenta encloses on the
    # hyperboloid.
    square_area = 4 * (2 * MOMENTUM * np.arcsinh(MOMENTUM / np.sqrt(1 + MOMENTUM**2))
                       - np.arctan(MOMENTUM**2 / np.sqrt(1 + 2 * MOMENTUM**2)))
    *_, (_, box_end) = bouncing(positions, path, box_spin)
    expected = (mv.yx * (-N_LOOPS * square_area / 2)).exp() >> mv.zy
    np.testing.assert_allclose((box_end - expected).kernel, 0.0, atol=5e-2)
    # At the magic Lorentz factor the electric field drops out of the anomalous rate; elsewhere it
    # enters by the Bargmann–Michel–Telegdi formula.
    values = ratios.kernel[..., 0]
    speed = np.sqrt(1 - 1 / SWEEP_GAMMAS[0] ** 2)
    np.testing.assert_allclose(values[1], 1.0, atol=1e-3)
    np.testing.assert_allclose(values[0], 1 - (1 - 1 / (ANOMALY * (SWEEP_GAMMAS[0] ** 2 - 1))) * speed * ELECTRIC_FIELDS / MAGNETIC, atol=1e-3)


if __name__ == "__main__":
    main()
