"""Streams the four Thomas-precession animations, saves the magic-momentum chart, and verifies analytical spin-precession identities."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import numpy as np

from numga import concatenate
from examples.relativity.thomas_precession import core

mv = core.mv
frame_duration_ms = 40

n_boost_frames = 120
min_rapidity = 0.8
max_rapidity = 2.0
n_edge_samples = 31

orbit_gammas = np.array([1.25, 1.5, 1.75])
n_orbit_steps = 448
n_orbit_stride = 4

box_half_width = 1.0
box_momentum = 0.8
n_flight_frames = 13
n_bounce_steps = 12
n_loops = 3
gyroscope_spin = mv.yz.dual()                                                   # [] Spin-

muon_anomaly = 0.1
ring_magnetic = 1.0
ring_electric = 0.0
n_ring_turns = 3
n_ring_turn_steps = 300
n_ring_stride = 6
n_ring_samples = 127
muon_spin = mv.zx.dual()                                                        # [] Spin-

electric_fields = np.linspace(-0.3, 0.3, 7)
sweep_gammas = np.array([2.0, np.sqrt(1 + 1 / muon_anomaly)])
sweep_lab_duration = 10.0
n_sweep_steps = 200


# --- plumbing -------------------------------------------------------------------------
def record_states(
    states: Iterable,
    animate: Callable[..., Iterable[np.ndarray]],
    *arguments,
    name: str,
    duration_ms: int,
) -> list:
    """Streams `states` once through `animate(*arguments, ...)` into `save_animation` and returns the visited states."""
    from examples.util.animation import save_animation

    visited = []

    def stream() -> Iterator:
        for state in states:
            visited.append(state)
            yield state

    save_animation(animate(*arguments, stream()), name, duration_ms)
    return visited


def main() -> None:
    """Runs the four animations and magic-momentum sweep, saving outputs and checking precession identities."""
    from examples.relativity.thomas_precession import render
    from examples.util.animation import save_figure

    swell = (1 - np.cos(np.linspace(0, 2 * np.pi, n_boost_frames, endpoint=False))) / 2
    rapidities = min_rapidity + (max_rapidity - min_rapidity) * swell           # [n_boost_frames]
    wigner_rotors = core.wigner_rotor(rapidities)                               # [n_boost_frames] Rotor+
    boost_history = record_states(
        core.two_boost_states(rapidities, n_edge_samples),
        render.two_boosts_animation,
        name="thomas_two_boosts",
        duration_ms=frame_duration_ms,
    )

    velocities = core.orbit_velocities(orbit_gammas, n_orbit_steps)             # [n_orbit_steps + 1, 3] Antivector-
    initial_orbit_spins = core.step_boost(core.time, velocities[0]) >> gyroscope_spin   # [3] Spin-
    orbit_history = record_states(
        core.orbit_states(velocities, initial_orbit_spins, n_orbit_stride),
        render.orbits_animation,
        velocities,
        name="thomas_orbits",
        duration_ms=frame_duration_ms,
    )

    single_positions, single_velocities = core.box_trajectory(
        box_half_width, box_momentum, n_flight_frames, n_bounce_steps
    )
    positions = concatenate([single_positions] * n_loops)
    box_velocities = concatenate([single_velocities] * n_loops)
    initial_box_spin = core.step_boost(core.time, box_velocities[0]) >> gyroscope_spin   # [] Spin-
    wall_turns = (mv.xy * (-np.arange(4) * np.pi / 4)).exp()                    # [4] Rotor+
    wall_midpoints = wall_turns >> (box_half_width * mv.x.dual())               # [4] Antivector-
    box_history = record_states(
        core.box_states(positions, box_velocities, initial_box_spin),
        render.box_animation,
        box_half_width,
        wall_midpoints,
        name="thomas_box",
        duration_ms=frame_duration_ms,
    )

    magic_gamma = float(np.sqrt(1 + 1 / muon_anomaly))
    radius = core.ring_radius(magic_gamma, ring_magnetic, ring_electric)
    ring_turns = (mv.xy * (-np.linspace(0, 2 * np.pi, n_ring_samples, endpoint=False) / 2)).exp()   # [n_ring_samples] Rotor+
    ring_points = ring_turns >> (radius * mv.x.dual())                          # [n_ring_samples] Antivector-
    ring_history = record_states(
        core.ring_states(
            magic_gamma,
            ring_magnetic,
            ring_electric,
            muon_anomaly,
            muon_spin,
            n_ring_turns,
            n_ring_turn_steps,
            n_ring_stride,
        ),
        render.ring_animation,
        ring_points,
        name="thomas_storage_ring",
        duration_ms=frame_duration_ms,
    )

    rate_ratios = core.anomalous_rates(
        sweep_gammas,
        ring_magnetic,
        electric_fields,
        muon_anomaly,
        muon_spin,
        sweep_lab_duration,
        n_sweep_steps,
    )                                                                           # [2, 7] Scalar+
    save_figure(render.magic_momentum_figure(electric_fields / ring_magnetic, rate_ratios), "thomas_magic")

    # checks
    # Two perpendicular boosts leave a pure spatial rotation matching the closed-form Wigner angle.
    assert len(boost_history) == n_boost_frames
    assert ((mv.t & ((wigner_rotors >> core.time) - core.time)).abs() < 1e-6).all()
    wigner_angles = 2 * wigner_rotors.log().scalar_product(mv.xy)
    assert ((wigner_angles + 2 * np.arctan(np.tanh(rapidities / 2) ** 2)).abs() < 1e-6).all()

    # One circular velocity orbit turns each spin back by the enclosed hyperbolic cap area 2 * pi * (gamma - 1).
    _, final_orbit_spins = orbit_history[-1]
    orbit_turns = (mv.xy * (np.pi * (orbit_gammas - 1))).exp()
    expected_orbit_spins = orbit_turns >> gyroscope_spin
    assert ((final_orbit_spins - expected_orbit_spins).antinorm() < 5e-3).all()

    # Three diamond round trips in the box turn the spin back by three velocity-square hyperbolic areas.
    _, final_box_spin = box_history[-1]
    square_area = 4 * (
        2 * box_momentum * np.arcsinh(box_momentum / np.sqrt(1 + box_momentum**2))
        - np.arctan(box_momentum**2 / np.sqrt(1 + 2 * box_momentum**2))
    )
    box_turn = (mv.xy * (n_loops * square_area / 2)).exp()
    expected_box_spin = box_turn >> gyroscope_spin
    assert ((final_box_spin - expected_box_spin).antinorm() < 5e-2).all()

    # In the pure magnetic ring, three revolutions advance the laboratory spin by 6 * pi * anomaly * gamma.
    _, _, initial_ring_spin = ring_history[0]
    _, _, final_ring_spin = ring_history[-1]
    ring_turn = (mv.xy * (-np.pi * n_ring_turns * muon_anomaly * magic_gamma)).exp()
    expected_ring_spin = ring_turn >> initial_ring_spin
    assert ((final_ring_spin - expected_ring_spin).antinorm() < 5e-3).all()

    # Radial electric fields drop out at the magic Lorentz factor and follow the BMT slope off it.
    low_speed = np.sqrt(1 - 1 / sweep_gammas[0] ** 2)
    expected_low_rates = (
        1
        - (1 - 1 / (muon_anomaly * (sweep_gammas[0] ** 2 - 1)))
        * low_speed
        * electric_fields
        / ring_magnetic
    )
    assert ((rate_ratios[1] - 1).abs() < 1e-3).all()
    assert ((rate_ratios[0] - expected_low_rates).abs() < 1e-3).all()


if __name__ == "__main__":
    main()
