"""Geometric and physical checks of the plane-wave curvature example."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from examples.relativity.curvature import render, scenarios
from examples.relativity.curvature.core import (
    Antibivector, Antivector, curvature_of_strain, integrate_acceleration, lower, mv, polarizations, polarized,
    strain_patterns, tidal_map, time_direction, travel, wave_packet, wave_vector,
)

POLARIZATIONS = scenarios.POLARIZATIONS


def ring(count: int) -> Antivector:
    """Unit rest separations in the plane across the wave."""
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return (mv.xy * (angles / 2)).exp() >> mv.x.dual()


def areas(seed: int) -> Antibivector:
    """Random areas, joins of random directions."""
    first, second = mv.vector(np.random.default_rng(seed).normal(size=(2, 5, 4))).dual()
    return first & second


@pytest.mark.parametrize("amplitudes", [(1, 0), (0, 1), (.6, -.8)])
def test_every_plane_the_curvature_gives_contains_the_wave_and_a_quarter_turn_reverses_it(amplitudes):
    plus, cross = polarizations()
    curvature = amplitudes[0] * plus + amplitudes[1] * cross
    assert np.linalg.matrix_rank(curvature.kernel) == 2
    np.testing.assert_allclose((wave_vector ^ curvature(areas(3))).kernel, 0, atol=1e-14)
    quarter_turn = (mv.xy * (np.pi / 4)).exp()
    np.testing.assert_allclose(((quarter_turn >> curvature(quarter_turn << Antibivector)) + curvature).kernel, 0, atol=1e-14)


def test_pair_symmetry_bianchi_identity_and_vacuum():
    plus, cross = polarizations()
    curvature = 1.3 * plus - .7 * cross
    first, second = areas(7), areas(8)
    np.testing.assert_allclose(((first & curvature(second)) - (second & curvature(first))).kernel, 0, atol=1e-13)
    a, b, c = mv.vector(np.random.default_rng(9).normal(size=(3, 4))).dual()
    cyclic = (c & curvature(a & b)) + (a & curvature(b & c)) + (b & curvature(c & a))
    np.testing.assert_allclose(cyclic.kernel, 0, atol=1e-13)
    # Vacuum: every observer's tides stretch as much as they squeeze.
    for rapidity, direction in ((0.0, mv.x.dual()), (0.4, mv.y.dual()), (-0.8, mv.z.dual())):
        observer = ((lower(direction) ^ mv.t) * (rapidity / 2)).exp() >> time_direction
        accelerations = lower.solve(tidal_map(curvature, observer))
        np.testing.assert_allclose(accelerations.eigvals().to_array().real.sum(), 0, atol=1e-12)


def test_observer_sees_opposite_transverse_tides_and_doppler_scaling():
    plus, _ = polarizations()
    for rapidity in (-.7, 0, .5):
        observer = ((lower(mv.z.dual()) ^ mv.t) * (rapidity / 2)).exp() >> time_direction
        response = tidal_map(plus, observer)
        # A chasing observer sees the wave's frequency redshifted; the curvature takes the observer twice,
        # so the tidal amplitude scales with the frequency squared.
        amplitude = np.exp(-2 * rapidity)
        np.testing.assert_allclose(np.sort(lower.solve(response).eigvals().to_array().real),
                                   [-amplitude, 0, 0, amplitude], atol=1e-13)
        np.testing.assert_allclose(response(observer).kernel, 0, atol=1e-13)
        np.testing.assert_allclose(response(travel).kernel, 0, atol=1e-13)


def test_curvature_and_observer_binding_are_lorentz_covariant():
    plus, _ = polarizations()
    rotor = (mv.tx * .31).exp() * (mv.yz * -.23).exp()
    plane_wave = (rotor >> wave_vector) ^ (rotor >> mv.x), (rotor >> wave_vector) ^ (rotor >> mv.y)
    transformed = plane_wave[0] * (plane_wave[0] & Antibivector) - plane_wave[1] * (plane_wave[1] & Antibivector)
    np.testing.assert_allclose((transformed - (rotor >> plus(rotor << Antibivector))).kernel, 0, atol=1e-13)
    np.testing.assert_allclose((tidal_map(transformed, rotor >> time_direction)
                                - (rotor >> tidal_map(plus, time_direction)(rotor << Antivector))).kernel, 0, atol=1e-13)


def test_packet_second_derivative_matches_finite_differences():
    time = np.linspace(-.5, 6.5, 2801)
    strain, second = wave_packet(time, duration=6.0, cycles=3, amplitude=1e-4)
    dt = time[1] - time[0]
    numerical = (-strain[:-4] + 16 * strain[1:-3] - 30 * strain[2:-2] + 16 * strain[3:-1] - strain[4:]) / (12 * dt**2)
    interior = (time[2:-2] > .1) & (time[2:-2] < 5.9)
    np.testing.assert_allclose(numerical[interior], second[2:-2][interior], atol=1e-11)
    # Outside the window only the Gaussian's tails remain.
    outside = (time <= 0) | (time >= 6)
    np.testing.assert_allclose(strain[outside], 0, atol=1e-11)
    np.testing.assert_allclose(second[outside], 0, atol=1e-9)
    np.testing.assert_allclose(strain[len(time) // 2], [1e-4, 0], atol=1e-16)


def test_integrated_detector_response_converges_to_weak_wave_displacements():
    plus, cross = polarizations()
    reference = ring(12)
    plus_matrix = np.diag([1, -1])
    cross_matrix = np.array([[0, 1], [1, 0]])
    errors = []
    for count in (321, 641):
        time = np.linspace(-1, 7, count)
        _, second = wave_packet(time, duration=6.0, cycles=3, amplitude=1e-4)
        waves = polarized(plus, cross, -0.5 * second[:, None] * POLARIZATIONS)
        acceleration = lower.solve(tidal_map(waves, time_direction)[:, :, None](reference))
        displacement = render.transverse(integrate_acceleration(time, acceleration))

        # Independent prediction: the displacement is half the strain applied to the separation, for all
        # three polarizations and every bead, with the stated Gaussian packet.
        envelope = 1e-4 * np.exp(-(time - 3)**2 / (2 * 0.5**2))
        hp = (envelope * np.cos(np.pi * (time - 3)))[:, None, None]
        hc = (envelope * np.sin(np.pi * (time - 3)))[:, None, None]
        strain = np.stack((hp * plus_matrix, hc * cross_matrix, hp * plus_matrix + hc * cross_matrix), axis=1)
        expected = .5 * np.einsum("tcij,nj->tcni", strain, render.transverse(reference))
        errors.append(np.max(np.abs(displacement - expected)))

    assert errors[1] < errors[0] / 10  # Fourth-order integration convergence.
    assert errors[1] < 2e-9
    # At rest before the packet, to within the Gaussian's tails.
    np.testing.assert_allclose(displacement[time <= 0], 0, atol=1e-11)
    tail = displacement[time >= 6]
    np.testing.assert_allclose(tail, 0, atol=2e-10)
    np.testing.assert_allclose(np.diff(tail, axis=0) / (time[1] - time[0]), 0, atol=1e-10)


def test_strain_map_predicts_the_ring_and_its_second_derivative_is_the_curvature():
    plus_strain, cross_strain = strain_patterns()
    # Stretch along x, squeeze along y, nothing along time or the wave.
    for separation, image in ((mv.x.dual(), mv.x.dual()), (mv.y.dual(), -mv.y.dual()),
                              (time_direction, 0 * time_direction), (mv.z.dual(), 0 * time_direction)):
        np.testing.assert_allclose((plus_strain(separation) - image).kernel, 0, atol=1e-15)

    time = np.linspace(-1, 7, 641)
    strain, second = wave_packet(time, duration=6.0, cycles=3, amplitude=1e-4)
    plus, cross = polarizations()
    waves = polarized(plus, cross, -0.5 * second[:, None] * POLARIZATIONS)
    reference = ring(12)
    response = tidal_map(waves, time_direction)
    displacement = integrate_acceleration(time, lower.solve(response[:, :, None](reference)))
    predicted = polarized(plus_strain, cross_strain, 0.5 * strain[:, None] * POLARIZATIONS)[:, :, None](reference)
    np.testing.assert_allclose((displacement - predicted).kernel, 0, atol=2e-9)
    assert np.abs(predicted.kernel).max() > 4e-5

    # The tidal acceleration is the strain's second time derivative as a map on separations.
    acceleration_map = polarized(plus_strain, cross_strain, 0.5 * second[:, None] * POLARIZATIONS)
    np.testing.assert_allclose((lower.solve(response) - acceleration_map).kernel, 0, atol=1e-12)

    # Wedged with the wave vector, that second derivative is the curvature on pairs of directions.
    two_form = curvature_of_strain(wave_vector, acceleration_map)
    edge = mv.vector(np.random.default_rng(5).normal(size=4)).dual()
    np.testing.assert_allclose((two_form(edge) - waves(edge & Antivector)).kernel, 0, atol=1e-12)


def test_figures_and_animation_draw():
    time, profiles, reference, displacement, acceleration = scenarios.detector()
    figures = [
        render.draw_curvature_map(*scenarios.curvature_map()),
        render.draw_doppler(*scenarios.doppler()),
        render.draw_detector(time, reference, displacement, acceleration, scenarios.AMPLIFICATION),
        render.draw_packet(time, *wave_packet(time, scenarios.DURATION, scenarios.N_CYCLES, scenarios.AMPLITUDE)),
    ]
    assert all(isinstance(figure, plt.Figure) for figure in figures)
    frames = render.animate_detector(time[::100], reference, displacement[::100], acceleration[::100],
                                     scenarios.AMPLIFICATION)
    assert frames and frames[0].ndim == 3 and all(f.shape == frames[0].shape for f in frames)
