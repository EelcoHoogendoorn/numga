"""Geometric and physical checks of the plane-wave curvature example."""


import matplotlib.pyplot as plt
import numpy as np
import pytest

from numga import stack

from examples.relativity.curvature import render, scenarios
from examples.relativity.curvature.core import (
    Bivector, Curvature, Tidal, Vector, curvature_of_strain, detector_ring, integrate_acceleration, mv,
    polarized_strain, polarized_waves, strain_patterns, t, wave_packet, x, y, z,
)


def plane_wave(k, x, y) -> Curvature:
    """The tutorial's plus curvature: two null dyads with opposite weights."""
    nx, ny = k.wedge(x), k.wedge(y)
    return nx * (nx | Bivector) - ny * (ny | Bivector)


def tidal(curvature, observer) -> Tidal:
    """The tutorial's tidal map: the observer bound twice, the separation open."""
    return curvature(observer.wedge(Vector)).commutator(observer)


def polarizations():
    plus = plane_wave(t + z, x, y)
    rotation = (mv.xy * (np.pi / 8)).exp()
    cross = rotation >> plus(rotation << Bivector)
    return plus, cross


def polarized_wave(plus, cross, profile):
    return stack((plus * profile[:, 0], cross * profile[:, 0], plus * profile[:, 0] + cross * profile[:, 1]), axis=1)


@pytest.mark.parametrize("amplitudes", [(1, 0), (0, 1), (.6, -.8)])
def test_nonzero_curvature_has_rank_two_and_annihilates_its_image(amplitudes):
    plus, cross = polarizations()
    curvature = amplitudes[0] * plus + amplitudes[1] * cross
    assert np.linalg.matrix_rank(curvature.kernel) == 2
    np.testing.assert_allclose(curvature(curvature).kernel, 0, atol=1e-14)
    # Null output planes are nonzero even though their Lorentz norms vanish.
    image = curvature(t.wedge(x))
    assert np.linalg.norm(image.kernel) > .5
    np.testing.assert_allclose((image | image).kernel, 0, atol=1e-14)


def test_riemann_pair_symmetry_bianchi_identity_and_vacuum_ricci():
    plus, cross = polarizations()
    curvature = 1.3 * plus - .7 * cross
    a, b, c, d = mv.vector(np.random.default_rng(7).normal(size=(4, 4)))
    ab, cd = a.wedge(b), c.wedge(d)
    np.testing.assert_allclose(
        (ab | curvature(cd)).kernel, (curvature(ab) | cd).kernel, atol=1e-13,
    )
    cyclic = (curvature(ab).commutator(c)
              + curvature(b.wedge(c)).commutator(a)
              + curvature(c.wedge(a)).commutator(b))
    np.testing.assert_allclose(cyclic.kernel, 0, atol=1e-13)
    # Contract one curvature slot with the reciprocal Lorentz basis.
    basis = mv.vector(np.eye(4))
    reciprocal = mv.vector(np.diag([1, -1, -1, -1]))
    ricci = curvature(basis.wedge(Vector)).commutator(reciprocal).sum(axis=0)
    np.testing.assert_allclose(ricci.kernel, 0, atol=1e-13)
    # Frame-free: the Ricci form is the trace of the curvature against its wedge slot,
    # with the inner-product slot and the separation left open.
    ricci_form = Vector.commutator(curvature(Vector.wedge(Vector))).trace(slot=1)
    np.testing.assert_allclose(ricci_form.kernel, 0, atol=1e-13)
    # The same trace on a non-vacuum curvature reproduces the frame contraction as a form.
    dyad = curvature + 0.7 * (t ^ x) * ((t ^ x) | Bivector)
    frame_map = dyad(basis.wedge(Vector)).commutator(reciprocal).sum(axis=0)
    a, b = mv.vector(np.random.default_rng(11).normal(size=(2, 4)))
    np.testing.assert_allclose(
        Vector.commutator(dyad(Vector.wedge(Vector))).trace(slot=1)(a, b).kernel,
        (a | frame_map(b)).kernel, atol=1e-13,
    )


def test_observer_sees_opposite_transverse_tides_and_doppler_scaling():
    curvature = plane_wave(t + z, x, y)
    for rapidity in (-.7, 0, .5):
        observer = t * np.cosh(rapidity) + z * np.sinh(rapidity)
        response = tidal(curvature, observer)
        # A chasing observer sees the wave frequency redshift; curvature has
        # two time slots, so the tidal amplitude scales with frequency squared.
        amplitude = np.exp(-2 * rapidity)
        np.testing.assert_allclose(
            np.sort(np.linalg.eigvals(response.kernel).real),
            [-amplitude, 0, 0, amplitude], atol=1e-13,
        )
        np.testing.assert_allclose(response(observer).kernel, 0, atol=1e-13)
        np.testing.assert_allclose(response(t + z).kernel, 0, atol=1e-13)


def test_curvature_and_observer_binding_are_lorentz_covariant():
    plus, _ = polarizations()
    rotor = (mv.tx * .31).exp() * (mv.yz * -.23).exp()
    transformed = plane_wave(rotor >> (t + z), rotor >> x, rotor >> y)
    conjugated = rotor >> plus(rotor << Bivector)
    np.testing.assert_allclose(transformed.kernel, conjugated.kernel, atol=1e-13)
    observer_response = tidal(transformed, rotor >> t)
    transformed_response = rotor >> tidal(plus, t)(rotor << Vector)
    np.testing.assert_allclose(
        observer_response.kernel, transformed_response.kernel, atol=1e-13,
    )


def test_packet_acceleration_matches_the_strain_second_derivative():
    time = np.linspace(-.5, 6.5, 2801)
    strain, second = wave_packet(time, duration=6.0, cycles=3, amplitude=1e-4)
    h, acceleration = strain.kernel[..., 0], second.kernel[..., 0]
    dt = time[1] - time[0]
    numerical = (-h[:-4] + 16 * h[1:-3] - 30 * h[2:-2]
                 + 16 * h[3:-1] - h[4:]) / (12 * dt**2)
    interior = (time[2:-2] > .1) & (time[2:-2] < 5.9)
    np.testing.assert_allclose(
        numerical[interior], acceleration[2:-2][interior], atol=1e-11,
    )
    outside = (time <= 0) | (time >= 6)
    np.testing.assert_array_equal(h[outside], 0)
    np.testing.assert_array_equal(acceleration[outside], 0)
    np.testing.assert_allclose(h[len(time) // 2], [1e-4, 0], atol=1e-16)


def test_integrated_detector_response_converges_to_weak_wave_displacements():
    plus, cross = polarizations()
    reference = detector_ring(12)
    plus_matrix = np.diag([1, -1])
    cross_matrix = np.array([[0, 1], [1, 0]])
    errors = []
    for count in (321, 641):
        time = np.linspace(-1, 7, count)
        _, second = wave_packet(time, duration=6.0, cycles=3, amplitude=1e-4)
        acceleration = tidal(polarized_wave(plus, cross, second) * -.5, t)[:, :, None](reference)
        displacement = integrate_acceleration(time, acceleration).kernel[..., 1:3]

        # Independent prediction: the displacement is half the strain applied to the separation, for all
        # three polarizations and every bead, with the stated sin^4 packet.
        envelope = 1e-4 * np.sin(np.pi * np.clip(time / 6, 0, 1))**4
        hp = (envelope * np.cos(np.pi * (time - 3)))[:, None, None]
        hc = (envelope * np.sin(np.pi * (time - 3)))[:, None, None]
        strain = np.stack((hp * plus_matrix, hp * cross_matrix,
                           hp * plus_matrix + hc * cross_matrix), axis=1)
        expected = .5 * np.einsum("tcij,nj->tcni", strain, reference.kernel[..., 1:3])
        errors.append(np.max(np.abs(displacement - expected)))

    assert errors[1] < errors[0] / 10  # Fourth-order integration convergence.
    assert errors[1] < 2e-9
    np.testing.assert_array_equal(displacement[time <= 0], 0)
    tail = displacement[time >= 6]
    np.testing.assert_allclose(tail, 0, atol=2e-10)
    np.testing.assert_allclose(np.diff(tail, axis=0) / (time[1] - time[0]), 0, atol=1e-12)

    # Stopping the calculation halfway through the pulse must retain the
    # nonzero displacement; the integrator must not reset its final frame.
    middle = len(time) // 2
    prefix = integrate_acceleration(time[:middle + 1], acceleration[:middle + 1])
    np.testing.assert_allclose(prefix.kernel[-1, ..., 1:3], displacement[middle], atol=1e-15)
    assert np.max(np.abs(prefix.kernel[-1])) > 4e-5


def test_strain_map_predicts_the_ring_and_its_second_derivative_is_the_curvature():
    plus_strain, cross_strain = strain_patterns()
    # Stretch along x, squeeze along y, nothing along time or the wave.
    for separation, image in ((x, x), (y, -y), (t, t * 0), (z, z * 0)):
        np.testing.assert_allclose(plus_strain(separation).kernel, image.kernel, atol=1e-15)

    time = np.linspace(-1, 7, 641)
    strain, second = wave_packet(time, duration=6.0, cycles=3, amplitude=1e-4)
    waves = polarized_waves(*polarizations(), second)
    reference = detector_ring(12)
    displacement = integrate_acceleration(time, tidal(waves, t)[:, :, None](reference))
    predicted = polarized_strain(plus_strain, cross_strain, strain)[:, :, None](reference)
    np.testing.assert_allclose(displacement.kernel, predicted.kernel, atol=2e-9)
    assert np.abs(predicted.kernel).max() > 4e-5

    # The tidal map is the strain's second time derivative as a map on separations.
    acceleration_map = polarized_strain(plus_strain, cross_strain, second)
    np.testing.assert_allclose(tidal(waves, t).kernel, acceleration_map.kernel, atol=1e-15)

    # Wedged with the wave vector, that second derivative is the curvature on pairs of vectors.
    two_form = curvature_of_strain(t + z, acceleration_map)
    edge = mv.vector(np.random.default_rng(5).normal(size=4))
    np.testing.assert_allclose(two_form.bind(edge).kernel, waves(edge.wedge(Vector)).kernel, atol=1e-15)


def test_figures_and_animation_draw():
    """Each scenario runs its checks; the figures and a short animation draw."""
    detector = scenarios.detector_scenario()
    figures = [
        render.draw_curvature_map(*scenarios.curvature_map_scenario()),
        render.draw_doppler(*scenarios.doppler_scenario()),
        render.draw_detector(*detector),
    ]
    assert all(isinstance(figure, plt.Figure) for figure in figures)
    time, reference, displacement, acceleration, amplification = detector
    frames = render.animate_detector(time[::100], reference, displacement[::100], acceleration[::100], amplification)
    assert frames and frames[0].ndim == 3 and all(f.shape == frames[0].shape for f in frames)
