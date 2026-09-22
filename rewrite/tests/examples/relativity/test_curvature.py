"""Geometric and physical checks of the plane-wave curvature example."""

import numpy as np
import pytest

from numga import stack

from examples.relativity.curvature import Curvature, Tidal, main
from examples.relativity.curvature_plumbing import (
    Bivector, Vector, detector_ring, integrate_acceleration, mv, t, wave_packet, x, y, z,
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
    strain, second = wave_packet(time)
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
        _, second = wave_packet(time)
        acceleration = tidal(polarized_wave(plus, cross, second) * -.5, t)[:, :, None](reference)
        displacement = integrate_acceleration(time, acceleration).kernel[..., 1:3]

        # Independent TT-gauge prediction: delta x = h_TT x / 2, for all
        # three polarizations and every bead, with the stated sin^4 packet.
        envelope = 1e-4 * np.sin(np.pi * np.clip(time / 6, 0, 1))**4
        hp = (envelope * np.cos(np.pi * (time - 3)))[:, None, None]
        hc = (envelope * np.sin(np.pi * (time - 3)))[:, None, None]
        strain = np.stack((hp * plus_matrix, hp * cross_matrix,
                           hp * plus_matrix + hc * cross_matrix), axis=1)
        expected = .5 * np.einsum("tcij,nj->tcni", strain, reference.kernel[..., 1:3])
        errors.append(np.max(np.abs(displacement - expected)))

    assert errors[1] < errors[0] / 10  # Fourth-order integration convergence.
    assert errors[1] < 2e-11
    np.testing.assert_array_equal(displacement[time <= 0], 0)
    tail = displacement[time >= 6]
    np.testing.assert_allclose(tail, 0, atol=2e-12)
    np.testing.assert_allclose(np.diff(tail, axis=0) / (time[1] - time[0]), 0, atol=1e-12)

    # Stopping the calculation halfway through the pulse must retain the
    # nonzero displacement; the integrator must not reset its final frame.
    middle = len(time) // 2
    prefix = integrate_acceleration(time[:middle + 1], acceleration[:middle + 1])
    np.testing.assert_allclose(prefix.kernel[-1, ..., 1:3], displacement[middle], atol=1e-15)
    assert np.max(np.abs(prefix.kernel[-1])) > 4e-5


def test_tutorial_runs_and_saves(tmp_path):
    out = tmp_path / "curvature.png"
    main(plot_path=str(out))
    assert out.exists()
