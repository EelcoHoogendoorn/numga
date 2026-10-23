"""The hole satisfies mechanical equilibrium; its optical response obeys the stress-optic law."""

import numpy as np

from numga import stack
from examples.optics.photoelasticity import core


ROUND_OFF = 1e-10


def uniform_stress(along_x: float, along_y: float, shear: float) -> core.Stress:
    """A uniform symmetric stress from its normal stresses along x and y and its shear."""
    x, y = core.mv.x.dual(), core.mv.y.dual()
    return (along_x * x * (core.Cut & x)
            + along_y * y * (core.Cut & y)
            + shear * (x * (core.Cut & y) + y * (core.Cut & x)))


def planar(coordinates: list) -> core.Planar:
    """Directions in the plate from their x and y coordinates."""
    coordinates = np.array(coordinates)
    return core.mv.x.dual() * coordinates[..., 0] + core.mv.y.dual() * coordinates[..., 1]


def test_free_rim_and_uniaxial_stress_concentration():
    radius = 1.4
    tension = 2.3
    samples = 73
    angles = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    # The cuts along the rim, facing out, and the directions across them.
    cuts = (core.mv.xy * (-angles / 2)).exp() >> core.mv.x
    normal = cuts.dual()
    remote = uniform_stress(1.7, -0.4, 0.6)
    axial = uniform_stress(tension, 0.0, 0.0)
    end_cuts = stack([core.mv.x, core.mv.y])
    tangent_cuts = core.quarter_turn >> end_cuts

    traction = core.kirsch(radius * normal, radius, remote)(cuts)
    rim = core.kirsch(radius * end_cuts.dual(), radius, axial)
    hoop = tangent_cuts & rim(tangent_cuts)
    expected = tension * np.array([-1., 3.])

    # checks
    np.testing.assert_allclose(traction.kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(hoop.kernel[..., 0], expected, atol=ROUND_OFF, rtol=0)


def test_stress_rotates_with_load_and_positions_and_recovers_remote_field():
    radius = 0.8
    far_distance = 1e7 * radius
    turn = (core.mv.xy * -0.37).exp()
    positions = planar([[1.2, 0.4], [-1.6, 0.7], [0.3, -2.1]])
    remote = uniform_stress(2.1, 0.3, -0.5)
    far_positions = positions / positions.squared().square_root() * far_distance
    moved_remote = turn >> remote(turn << core.Cut)
    cuts = stack([core.mv.x + 0.3 * core.mv.y, core.mv.y - 1.2 * core.mv.x])

    stress = core.kirsch(positions, radius, remote)
    moved = core.kirsch(turn >> positions, radius, moved_remote)
    rotated = turn >> stress(turn << core.Cut)
    far_stress = core.kirsch(far_positions, radius, remote)
    # Symmetric: the stress form reads two cuts the same either way round.
    form = core.Cut & stress(core.Cut)

    # checks
    np.testing.assert_allclose((moved - rotated).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((form(cuts[0], cuts[1]) - form(cuts[1], cuts[0])).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((far_stress - remote).kernel, 0, atol=ROUND_OFF, rtol=0)


def test_stress_has_zero_divergence_away_from_the_hole():
    radius = 0.7
    step = 1e-4 * radius
    positions = planar([[1.1, 0.3], [-1.3, 0.8], [0.4, -1.7]])
    # Steps along x and y, and the cuts that read them, paired with no metric.
    cuts = stack([core.mv.x, core.mv.y])
    directions = cuts.dual()
    remote = uniform_stress(1.8, -0.2, 0.4)
    plus = core.kirsch(positions[:, None] + step * directions, radius, remote)
    minus = core.kirsch(positions[:, None] - step * directions, radius, remote)
    derivative = (plus - minus) / (2 * step)
    divergence = derivative(cuts).sum(axis=-1)

    # checks: this tolerance tests the centred derivative's truncation error.
    np.testing.assert_allclose(divergence.kernel, 0, atol=1e-7, rtol=0)


def test_mean_stress_delays_every_direction_alike():
    stress_phase = 83 * np.pi + 0.41
    mean = 3.7
    base = uniform_stress(2.4, -0.8, 0.5)
    # An isotropic stress: the mean times the metric, the traction straight across every cut.
    isotropic = core.mv.scalar([mean]) * core.Cut.dual()
    plates = core.retarder(stack([base, base + isotropic, isotropic]), stress_phase)
    delay = np.exp(1j * stress_phase * mean)

    # checks: the mean stress only delays the whole field, and alone leaves its polarization as it was.
    np.testing.assert_allclose((plates[1] - delay * plates[0]).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((plates[2] - delay * (core.polarizer(core.mv.x.dual()) + core.polarizer(core.mv.y.dual()))).kernel,
                               0, atol=ROUND_OFF, rtol=0)


def test_linear_and_circular_dark_fields_match_the_retardance_law():
    orientations = 19
    angles = np.linspace(0, np.pi, orientations, endpoint=False)
    retardances = np.array([0., 0.01, 0.83, np.pi, 2 * np.pi, 7.1, 80 * np.pi + 0.4])
    turn = (core.mv.xy * (-angles[:, None] / 2)).exp()
    base = uniform_stress(1.0, 0.0, 0.0)
    stress = (turn >> base(turn << core.Cut)) * retardances[None, :]

    plates = core.retarder(stress, 1.0)
    linear_power = core.linear_polariscope(plates, core.mv.x.dual())
    circular_power = core.circular_polariscope(plates, core.mv.x.dual())
    phase_intensity = np.sin(retardances / 2)**2
    expected_linear = np.sin(2 * angles[:, None])**2 * phase_intensity
    expected_circular = np.broadcast_to(phase_intensity, (orientations, len(retardances)))

    # checks
    np.testing.assert_allclose(linear_power.kernel[..., 0], expected_linear, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(circular_power.kernel[..., 0], expected_circular, atol=ROUND_OFF, rtol=0)


def test_loading_cycle_returns_to_dark_and_matches_the_loaded_plate():
    radius = 0.8
    stress_phase = 13.7
    loads = np.array([0., 0.5, 1., 0.5, 0.])
    positions = planar([[[1.2, 0.4], [-1.6, 0.7], [0.3, -2.1]]])
    remote = uniform_stress(2.1, 0.3, -0.5)

    stress = core.kirsch(positions, radius, remote)
    frames = core.circular_polariscope(core.retarder(stress * loads[:, None, None], stress_phase), core.mv.x.dual())
    # Recompute the stress at every remote load, rather than scaling the reference one.
    scaled_stress = core.kirsch(positions[None], radius, remote * loads[:, None, None])
    expected = core.circular_polariscope(core.retarder(scaled_stress, stress_phase), core.mv.x.dual())

    # checks
    np.testing.assert_allclose((frames - expected).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((frames - frames[::-1]).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(frames[0].kernel, 0, atol=ROUND_OFF, rtol=0)
