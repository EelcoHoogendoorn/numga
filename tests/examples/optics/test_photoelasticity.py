"""The hole satisfies mechanical equilibrium; its optical response obeys the stress-optic law."""

import numpy as np

from numga import stack
from examples.optics.photoelasticity import core


ROUND_OFF = 1e-10


def uniform_stress(along_x: float, along_y: float, shear: float) -> core.Stress:
    """A uniform symmetric stress from its normal stresses along x and y and its shear."""
    return (along_x * core.mv.x * (core.mv.x | core.Planar)
            + along_y * core.mv.y * (core.mv.y | core.Planar)
            + shear * (core.mv.x * (core.mv.y | core.Planar) + core.mv.y * (core.mv.x | core.Planar)))


def test_free_rim_and_uniaxial_stress_concentration():
    radius = 1.4
    tension = 2.3
    samples = 73
    angles = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    normal = (core.mv.xy * (-angles / 2)).exp() >> core.mv.x
    remote = uniform_stress(1.7, -0.4, 0.6)
    axial = tension * (core.Planar + (core.mv.x >> core.Planar)) / 2
    ends = stack([core.mv.x, core.mv.y])
    tangent = core.mv.xy | ends

    traction = core.kirsch(radius * normal, radius, remote)(normal)
    rim = core.kirsch(radius * ends, radius, axial)
    hoop = tangent | rim(tangent)
    expected = tension * np.array([-1., 3.])

    # checks
    np.testing.assert_allclose(traction.kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(hoop.kernel[..., 0], expected, atol=ROUND_OFF, rtol=0)


def test_stress_rotates_with_load_and_positions_and_recovers_remote_field():
    radius = 0.8
    far_distance = 1e7 * radius
    turn = (core.mv.xy * -0.37).exp()
    positions = core.mv(core.Planar, [[1.2, 0.4], [-1.6, 0.7], [0.3, -2.1]])
    remote = uniform_stress(2.1, 0.3, -0.5)
    far_positions = positions.normalized() * far_distance
    moved_remote = turn >> remote(turn << core.Planar)

    stress = core.kirsch(positions, radius, remote)
    moved = core.kirsch(turn >> positions, radius, moved_remote)
    rotated = turn >> stress(turn << core.Planar)
    far_stress = core.kirsch(far_positions, radius, remote)

    # checks
    np.testing.assert_allclose((moved - rotated).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((stress - stress.adjoint()).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((far_stress - remote).kernel, 0, atol=ROUND_OFF, rtol=0)


def test_stress_has_zero_divergence_away_from_the_hole():
    radius = 0.7
    step = 1e-4 * radius
    positions = core.mv(core.Planar, [[1.1, 0.3], [-1.3, 0.8], [0.4, -1.7]])
    directions = stack([core.mv.x, core.mv.y])
    remote = uniform_stress(1.8, -0.2, 0.4)
    plus = core.kirsch(positions[:, None] + step * directions, radius, remote)
    minus = core.kirsch(positions[:, None] - step * directions, radius, remote)
    derivative = (plus - minus) / (2 * step)
    divergence = derivative(directions).sum(axis=-1)

    # checks: this tolerance tests the centred derivative's truncation error.
    np.testing.assert_allclose(divergence.kernel, 0, atol=1e-7, rtol=0)


def test_retarder_turns_the_sphere_and_ignores_mean_stress():
    stress_phase = 83 * np.pi + 0.41
    mean = 3.7
    base = uniform_stress(2.4, -0.8, 0.5)
    isotropic = core.mv.scalar([mean]) * core.Planar
    turns = core.retarder(stack([base, base + isotropic, isotropic]), stress_phase)
    states = stack([core.mv.x, core.mv.z, (0.6 * core.mv.x - 0.48 * core.mv.y + 0.64 * core.mv.z)])
    exits = turns[:, None] >> states[None, :]

    # checks: states stay on the sphere, the mean stress turns nothing, and alone leaves light as it was.
    np.testing.assert_allclose(exits.scalar_norm_squared().kernel, 1, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((exits[0] - exits[1]).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((exits[2] - states).kernel, 0, atol=ROUND_OFF, rtol=0)


def test_linear_and_circular_dark_ports_match_the_retardance_law():
    orientations = 19
    angles = np.linspace(0, np.pi, orientations, endpoint=False)
    retardances = np.array([0., 0.01, 0.83, np.pi, 2 * np.pi, 7.1, 80 * np.pi + 0.4])
    turn = (core.mv.xy * (-angles[:, None] / 2)).exp()
    base = (core.Planar + (core.mv.x >> core.Planar)) / 2
    stress = (turn >> base(turn << core.Planar)) * retardances[None, :]

    turns = core.retarder(stress, 1.0)
    linear_power = core.transmitted(turns >> core.polarization(core.mv.x), core.polarization(core.mv.y))
    circular_power = core.transmitted(turns >> core.mv.z, -core.mv.z)
    phase_intensity = np.sin(retardances / 2)**2
    expected_linear = np.sin(2 * angles[:, None])**2 * phase_intensity
    expected_circular = np.broadcast_to(phase_intensity, (orientations, len(retardances)))

    # checks: the rotor exponential over forty turns of the sphere is good to about 4e-7.
    np.testing.assert_allclose(linear_power.kernel[..., 0], expected_linear, atol=1e-4, rtol=0)
    np.testing.assert_allclose(circular_power.kernel[..., 0], expected_circular, atol=1e-4, rtol=0)


def test_loading_cycle_returns_to_dark_and_matches_the_loaded_plate():
    radius = 0.8
    stress_phase = 13.7
    loads = np.array([0., 0.5, 1., 0.5, 0.])
    positions = core.mv(core.Planar, [[[1.2, 0.4], [-1.6, 0.7], [0.3, -2.1]]])
    remote = uniform_stress(2.1, 0.3, -0.5)
    incident = stack([core.polarization(core.mv.x), core.mv.z])
    analyser = -incident

    stress = core.kirsch(positions, radius, remote)
    frames = core.polariscope(core.half_turn(stress, stress_phase) * loads[:, None, None, None],
                              incident[:, None, None], analyser[:, None, None])
    # Recompute the stress at every remote load, rather than scaling the reference one.
    scaled_stress = core.kirsch(positions[None], radius, remote * loads[:, None, None])
    turns = core.retarder(scaled_stress, stress_phase)
    expected = core.transmitted(turns[:, None] >> incident[None, :, None, None], analyser[None, :, None, None])

    # checks
    np.testing.assert_allclose((frames - expected).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((frames - frames[::-1]).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(frames[0].kernel, 0, atol=ROUND_OFF, rtol=0)
