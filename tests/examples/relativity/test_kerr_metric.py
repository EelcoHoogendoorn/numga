"""Metric and frame descriptions of the same Kerr rays, observers and parallel transport."""

import numpy as np

from numga import stack
from examples.relativity.kerr import core, metric

mv = core.mv
T, X, Y, Z = core.time, mv.x.dual(), mv.y.dual(), mv.z.dual()
DIRECTIONS = stack([T, X, Y, Z])
SLICES = stack([mv.t, mv.x, mv.y, mv.z])
ROUND_OFF = 1e-11


def events(coordinates):
    return (DIRECTIONS * np.asarray(coordinates)).sum(axis=-1)


def test_metric_and_hamiltonian_agree_with_the_frame_on_general_momenta():
    positions = events([[0., 4., 2., 1.], [3., -3., 1., 2.], [-1., 2., -4., -1.]])
    momenta = (SLICES * np.array([[1., .8, -.3, .2], [2., .1, .5, -.7], [-1., .4, .2, .9]])).sum(axis=-1)
    spin, mass = .7 * mv.xy + .2 * mv.yz, 1.
    geometry = metric.field(positions, spin, mass)
    frame_geometry = core.field(positions, spin, mass)
    tangent = metric.inverse_metric(geometry)(momenta)
    np.testing.assert_allclose((metric.metric(geometry)(tangent) - momenta).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((metric.hamiltonian(geometry, momenta) - core.hamiltonian(frame_geometry, momenta)).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((tangent - core.motion(frame_geometry, momenta)[0]).kernel, 0, atol=ROUND_OFF, rtol=0)
    flat = core.field(4 * X, spin, 0.)
    # Non-null momenta distinguish the Hamiltonian's sign, which a null check cannot.
    np.testing.assert_allclose(core.hamiltonian(flat, mv.t).kernel, .5, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(core.hamiltonian(flat, mv.x).kernel, -.5, atol=ROUND_OFF, rtol=0)


def test_christoffel_is_symmetric_and_matches_the_changing_local_frame():
    position = events([1., 4., 2., 1.])
    spin, mass = .7 * mv.xy + .2 * mv.yz, 1.
    along, direction = events([1., .4, -.2, .3]), events([.2, -.3, .8, -.1])
    geometry = metric.field(position, spin, mass)
    frame_geometry = core.field(position, spin, mass)
    coordinate_direction = core.gauge(frame_geometry)(direction)
    increment = 1e-4
    frame_change = (core.gauge(core.field(position + increment * along, spin, mass))(direction)
                    - core.gauge(core.field(position - increment * along, spin, mass))(direction)) / (2 * increment)
    local_along = core.gauge_inverse(frame_geometry)(along)
    local_rate = -core.rotation(frame_geometry, local_along).commutator(direction)
    coordinate_rate = frame_change + core.gauge(frame_geometry)(local_rate)
    np.testing.assert_allclose((metric.christoffel(geometry, along, coordinate_direction) + coordinate_rate).kernel,
                               0, atol=2e-9, rtol=0)
    np.testing.assert_allclose((metric.christoffel(geometry, along, direction)
                               - metric.christoffel(geometry, direction, along)).kernel, 0, atol=ROUND_OFF, rtol=0)
    # The covariant derivative of the metric vanishes.
    variation = metric.metric_change(geometry, along, direction) & direction
    connection_reading = metric.metric(geometry)(metric.christoffel(geometry, along, direction)) & direction
    np.testing.assert_allclose((variation - 2 * connection_reading).kernel, 0, atol=ROUND_OFF, rtol=0)


def test_both_cameras_use_the_coordinate_static_observer_for_screen_and_frequency():
    spin, mass = .8 * mv.xy, 1.
    orientation = (mv.zx * (-1.2 / 2)).exp()
    eye = (orientation >> Z) * 28
    position, momentum = core.camera_rays(eye, orientation, 5, 3, .35, spin, mass)
    metric_position, metric_momentum = metric.camera_rays(eye, orientation, 5, 3, .35, spin, mass)
    local_screen = core.camera_screen(position, momentum, orientation, spin, mass)
    coordinate_screen = metric.camera_screen(metric_position, metric_momentum, orientation, spin, mass)
    geometry = metric.field(position, spin, mass)
    frame_geometry = core.field(position, spin, mass)
    observer = T / metric.proper_squared(geometry, T).square_root()
    tangent = metric.inverse_metric(geometry)(momentum)
    np.testing.assert_allclose((metric_momentum - momentum).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((coordinate_screen - core.gauge(frame_geometry)(local_screen)).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((metric.metric(geometry)(observer) & coordinate_screen).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((momentum & coordinate_screen).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((momentum & tangent).kernel, 0, atol=ROUND_OFF, rtol=0)
    # The camera boost maps its rest-time direction to precisely the static observer.
    carried_observer = core.gauge(frame_geometry)(core.camera_boost(frame_geometry) >> T)
    np.testing.assert_allclose((carried_observer - observer).kernel, 0, atol=ROUND_OFF, rtol=0)


def test_disk_images_and_polarization_agree_after_coordinate_transport():
    mass, spin = 1., .8 * mv.xy
    orientation = (mv.zx * (-1.2 / 2)).exp()
    eye = (orientation >> Z) * 28
    position, momentum = core.camera_rays(eye, orientation, 9, 7, .35, spin, mass)
    screen = core.camera_screen(position, momentum, orientation, spin, mass)
    coordinate_screen = core.gauge(core.field(position, spin, mass))(screen)
    inner = core.innermost_stable_orbit(mv.xy, spin, mass)
    outer, capture, escape = 12., 1.68, 60.

    def opacity(event, momentum, radius):
        depth = core.disk_depth(radius, inner, outer, 4.)
        return core.disk_opacity(event, momentum, depth, mv.z, mv.xy, spin, mass)

    first, _ = core.resolve((position, momentum, screen), spin, mass, .08, 720,
                            core.polarized_rates, mv.z, inner, outer, opacity, escape, capture)
    second, _ = metric.resolve((position, momentum, coordinate_screen), spin, mass, .08, 720,
                               metric.polarized_rates, mv.z, inner, outer, opacity, escape, capture)
    np.testing.assert_array_equal(first.disk_pixels, second.disk_pixels)
    np.testing.assert_array_equal(first.sky_pixels, second.sky_pixels)
    np.testing.assert_allclose((first.disk_weights - second.disk_weights).kernel, 0, atol=1e-9, rtol=0)
    first_covariance = core.disk_coherency(*first.disk, mv.z, spin, mass, .8)
    second_covariance = metric.disk_coherency(*second.disk, mv.z, spin, mass, .8)
    # Linear interpolation to a crossing and RK4 in different variables leave a finite-step error.
    np.testing.assert_allclose((first_covariance - second_covariance).kernel, 0, atol=2e-3, rtol=0)
    first_shift = core.frequency_ratio(first.disk[0], first.disk[1], eye, mv.xy, spin, mass)
    second_shift = metric.frequency_ratio(second.disk[0], second.disk[1], eye, mv.xy, spin, mass)
    np.testing.assert_allclose((first_shift - second_shift).kernel, 0, atol=1e-9, rtol=0)
