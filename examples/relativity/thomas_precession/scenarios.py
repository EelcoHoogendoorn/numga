"""Scenes for Thomas precession and Bargmann–Michel–Telegdi spin dynamics in the spacetime algebra:
discrete Fermi–Walker transport around circular orbits at three Lorentz factors, and continuous
storage-ring evolution comparing a torque-free gyroscope (`g = 0`), a Dirac particle (`g = 2`), and
an anomalous magnetic moment (`g > 2`)."""

from __future__ import annotations

import numpy as np

from examples.relativity.thomas_precession import core

mv = core.mv
# Lorentz factors and step count for discrete Fermi–Walker transport around a circular orbit.
GAMMAS = np.array([1.2, 1.5, 2.0])
LOOP_STEPS = 180
# Storage-ring Lorentz factor, gyromagnetic ratios, cyclotron frequency, and step count.
RING_GAMMA = 1.6
G_FACTORS = np.array([0.0, 2.0, 2.5])
CYCLOTRON_RATE = 1.0
RING_STEPS = 180
EVERY = 6


# --- math -----------------------------------------------------------------------------
def thomas_loop():
    """Discrete Fermi–Walker transport around circular orbits at `GAMMAS` `[orbits]`: returns unit-circle
    positions `[LOOP_STEPS + 1, 1] Vector`, four-velocities `[LOOP_STEPS + 1, orbits] Vector`, four-spins
    `[LOOP_STEPS + 1, orbits] Vector`, and laboratory rest-frame spins `[LOOP_STEPS + 1, orbits] Vector`."""
    rapidities = np.arccosh(GAMMAS)                                             # [orbits]
    positions, four_velocities, start_spins = core.circular_orbit_velocities(rapidities, LOOP_STEPS)
    four_spins, lab_spins = core.fermi_walker_loop(four_velocities, start_spins)

    # --- checks
    # Four-velocity stays unit timelike, four-spin stays unit spacelike and orthogonal to four-velocity,
    # and the laboratory rest-frame spin has zero time component.
    np.testing.assert_allclose((four_velocities | four_velocities).kernel, 1.0, atol=1e-12)
    np.testing.assert_allclose((four_spins | four_spins).kernel, -1.0, atol=1e-12)
    np.testing.assert_allclose((four_velocities | four_spins).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((lab_spins | core.TIME).kernel, 0.0, atol=1e-12)
    # After one full counterclockwise orbit, the laboratory spin has rotated clockwise by 2*pi*(gamma - 1).
    expected_turn = (mv.xy * (-np.pi * (GAMMAS - 1.0))).exp()                   # [orbits] Rotor
    expected_end = expected_turn >> start_spins                                 # [orbits] Vector
    np.testing.assert_allclose((lab_spins[-1] - expected_end).kernel, 0.0, atol=5e-3)
    return positions, four_velocities, four_spins, lab_spins


def storage_ring():
    """Continuous BMT evolution over one cyclotron orbit in a uniform magnetic field `mv.xy` for three
    gyromagnetic ratios `G_FACTORS`: returns unit-circle positions `[RING_STEPS + 1, 1] Vector`,
    four-velocities, four-spins, and laboratory spins `[RING_STEPS + 1, cases] Vector`."""
    rapidities = np.full_like(G_FACTORS, np.arccosh(RING_GAMMA))                # [cases]
    positions, orbit_velocities, _ = core.circular_orbit_velocities(rapidities, RING_STEPS)
    start_frame = core.rest_boost(orbit_velocities[0])                          # [cases] Rotor
    field = CYCLOTRON_RATE * mv.xy                                              # [] Bivector
    dt = 2 * np.pi / (CYCLOTRON_RATE * RING_STEPS)
    rest_spin = mv.y                                                            # [] Vector
    four_velocities, four_spins, lab_spins = core.evolve_bmt(
        field, start_frame, rest_spin, G_FACTORS, dt, RING_STEPS
    )

    # --- checks
    # BMT rate map agrees with `omega | s` and preserves `u | s == 0` because `s | (omega | u) + u | (omega | s) == 0`.
    rate_map = core.bmt_map(field, four_velocities[0], G_FACTORS)               # [cases] Vector <- Vector
    np.testing.assert_allclose((rate_map(four_spins[0]) - (core.bmt_generator(field, four_velocities[0], G_FACTORS) | four_spins[0])).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((four_velocities | four_spins).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((lab_spins | core.TIME).kernel, 0.0, atol=1e-12)
    # After one cyclotron period in the xy plane, each laboratory spin has rotated by
    # `2 * pi * (1 + RING_GAMMA * (g / 2 - 1))` around the xy plane.
    net_angles = 2 * np.pi * (1.0 + RING_GAMMA * (G_FACTORS / 2.0 - 1.0))
    expected_end = (mv.xy * (net_angles / 2.0)).exp() >> rest_spin              # [cases] Vector
    np.testing.assert_allclose((lab_spins[-1] - expected_end).kernel, 0.0, atol=1e-2)
    return positions, four_velocities, four_spins, lab_spins


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.relativity.thomas_precession import render

    loop_pos, loop_u, loop_s, loop_lab = thomas_loop()
    ring_pos, ring_u, ring_s, ring_lab = storage_ring()
    save_figure(render.draw_thomas_summary(loop_pos, loop_u, loop_lab, ring_pos, ring_lab), "thomas_precession")
    save_animation(render.animate_storage_ring(ring_pos, ring_lab, EVERY), "thomas_precession", 50)
