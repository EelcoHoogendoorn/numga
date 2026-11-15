"""Metric and frame descriptions of the same Kerr rays, observers, and parallel transport."""

import numpy as np

from numga import stack
from examples.relativity.kerr import core, metric

mv = core.mv
time_axis, x_axis, y_axis, z_axis = core.time, mv.x.dual(), mv.y.dual(), mv.z.dual()
directions = stack([time_axis, x_axis, y_axis, z_axis])
slices = stack([mv.t, mv.x, mv.y, mv.z])
round_off = 1e-11


def events(coordinates):
    return (directions * np.asarray(coordinates)).sum(axis=-1)


def test_metric_and_hamiltonian_agree_with_the_frame_on_general_momenta():
    positions = events([[0., 4., 2., 1.], [3., -3., 1., 2.], [-1., 2., -4., -1.]])
    momenta = (slices * np.array([[1., .8, -.3, .2], [2., .1, .5, -.7], [-1., .4, .2, .9]])).sum(axis=-1)
    spin, mass = (.7 * mv.xy + .2 * mv.yz).dual(), 1.
    _, profile, null, profile_grad, null_grad = metric.field_components(positions, spin, mass)
    _, frame_profile, frame_null, frame_profile_grad, frame_null_grad, _ = core.field_components(positions, spin, mass)
    del profile_grad, null_grad
    tangent = metric.inverse_metric(profile, null)(momenta)
    assert ((metric.metric(profile, null)(tangent) - momenta).scalar_norm_squared() < round_off**2).all()
    assert ((metric.hamiltonian(profile, null, momenta)
             - core.hamiltonian(frame_profile, frame_null, momenta)).abs() < round_off).all()
    frame_tangent = core.motion(frame_profile, frame_null, frame_profile_grad, frame_null_grad, momenta)[0]
    assert ((tangent - frame_tangent).scalar_norm_squared() < round_off**2).all()
    _, flat_profile, flat_null, _, _, _ = core.field_components(4 * x_axis, spin, 0.)
    # Non-null momenta distinguish the Hamiltonian's sign, which a null check cannot.
    assert ((core.hamiltonian(flat_profile, flat_null, mv.t) - 0.5).abs() < round_off).all()
    assert ((core.hamiltonian(flat_profile, flat_null, mv.x) + 0.5).abs() < round_off).all()


def test_christoffel_is_symmetric_and_matches_the_changing_local_frame():
    position = events([1., 4., 2., 1.])
    spin, mass = (.7 * mv.xy + .2 * mv.yz).dual(), 1.
    step_direction, direction = events([1., .4, -.2, .3]), events([.2, -.3, .8, -.1])
    _, profile, null, profile_grad, null_grad = metric.field_components(position, spin, mass)
    _, frame_profile, frame_null, frame_profile_grad, frame_null_grad, frame_curl = core.field_components(position, spin, mass)
    coordinate_direction = core.gauge(frame_profile, frame_null)(direction)
    increment = 1e-4
    _, ahead_profile, ahead_null, _, _, _ = core.field_components(position + increment * step_direction, spin, mass)
    _, behind_profile, behind_null, _, _, _ = core.field_components(position - increment * step_direction, spin, mass)
    frame_change = (core.gauge(ahead_profile, ahead_null)(direction)
                    - core.gauge(behind_profile, behind_null)(direction)) / (2 * increment)
    local_step = core.gauge_inverse(frame_profile, frame_null)(step_direction)
    connection = core.rotation(frame_profile, frame_null, frame_profile_grad, frame_null_grad, frame_curl, local_step)
    local_rate = -connection.commutator(direction)
    coordinate_rate = frame_change + core.gauge(frame_profile, frame_null)(local_rate)
    christoffel_step = metric.christoffel(profile, null, profile_grad, null_grad, step_direction, coordinate_direction)
    assert ((christoffel_step + coordinate_rate).scalar_norm_squared() < 4e-18).all()
    sym_a = metric.christoffel(profile, null, profile_grad, null_grad, step_direction, direction)
    sym_b = metric.christoffel(profile, null, profile_grad, null_grad, direction, step_direction)
    assert ((sym_a - sym_b).scalar_norm_squared() < round_off**2).all()
    # The covariant derivative of the metric vanishes.
    variation = metric.metric_change(profile, null, profile_grad, null_grad, step_direction, direction) & direction
    connection_reading = metric.metric(profile, null)(sym_a) & direction
    assert ((variation - 2 * connection_reading).abs() < round_off).all()


def test_both_cameras_use_the_coordinate_static_observer_for_screen_and_frequency():
    spin, mass = .8 * mv.xy.dual(), 1.
    orientation = (mv.zx * (-1.2 / 2)).exp()
    eye = (orientation >> z_axis) * 28
    position, momentum = core.camera_rays(eye, orientation, 5, 3, .35, spin, mass)
    metric_position, metric_momentum = metric.camera_rays(eye, orientation, 5, 3, .35, spin, mass)
    local_screen = core.camera_screen(position, momentum, orientation, spin, mass)
    coordinate_screen = metric.camera_screen(metric_position, metric_momentum, orientation, spin, mass)
    _, profile, null, _, _ = metric.field_components(position, spin, mass)
    _, frame_profile, frame_null, _, _, _ = core.field_components(position, spin, mass)
    observer = time_axis / metric.proper_squared(profile, null, time_axis).square_root()
    tangent = metric.inverse_metric(profile, null)(momentum)
    screen_axes = stack([x_axis, y_axis])
    carried_screen = core.gauge(frame_profile, frame_null)[..., None](local_screen[..., None](screen_axes))
    screen_vectors = coordinate_screen[..., None](screen_axes)
    assert ((metric_momentum - momentum).scalar_norm_squared() < round_off**2).all()
    assert ((screen_vectors - carried_screen).scalar_norm_squared() < round_off**2).all()
    assert ((metric.metric(profile, null)(observer)[..., None] & screen_vectors).abs() < round_off).all()
    assert ((momentum[..., None] & screen_vectors).abs() < round_off).all()
    assert ((momentum & tangent).abs() < round_off).all()
    # The camera boost maps its rest-time direction to precisely the static observer.
    boost = core.camera_boost(frame_profile, frame_null)
    carried_observer = core.gauge(frame_profile, frame_null)(boost >> time_axis)
    assert ((carried_observer - observer).scalar_norm_squared() < round_off**2).all()


def test_the_metric_camera_rays_resolve():
    mass, spin = 1., .8 * mv.xy.dual()
    orientation = (mv.zx * (-1.2 / 2)).exp()
    eye = (orientation >> z_axis) * 28
    position, momentum = metric.camera_rays(eye, orientation, 2, 1, .35, spin, mass)
    screen = metric.camera_screen(position, momentum, orientation, spin, mass)
    inner = metric.innermost_stable_orbit(mv.xy, spin, mass)
    outer, capture, escape = 12., 1.68, 60.

    def opacity(event, ray_momentum, radius):
        depth = metric.disk_depth(radius, inner, outer, 4.)
        return metric.disk_opacity(event, ray_momentum, depth, mv.z, mv.xy, spin, mass)

    metric.resolve((position, momentum, screen), spin, mass, .2, 8, metric.polarized_rates, mv.z, inner, outer,
                   opacity, escape, capture)
