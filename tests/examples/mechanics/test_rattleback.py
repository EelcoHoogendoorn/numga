"""Rolling constraints, work balance and spin reversal of a rattleback."""

import numpy as np

from examples import instantiate
from examples.mechanics.rattleback import scenarios
from numga import stack
from numga.algebras import PGA3D


def model():
    core = instantiate("examples.mechanics.rattleback.core", PGA3D)
    mv = core.mv
    directions = stack((mv.x, mv.y, mv.z)).dual()
    semiaxes = np.array([5, 17 / 16, 1])
    mass, gravity = 4, 9.8
    skew = np.pi / 20
    shape = (directions * (core.Plane & directions) * semiaxes ** 2).sum(axis=0)
    mass_axes = (-mv.xy * skew / 2).exp() >> directions
    offsets = (mass_axes * semiaxes)[:, None] * np.array([-1, 1]) * np.sqrt(3 / 5)
    points = core.origin + offsets.reshape(-1)
    inertia = (points & points.commutator(core.Rate) * (mass / points.shape[0])).sum(axis=0)
    return core, shape, inertia, mass, gravity


def test_rolling_annihilates_contact_motion_and_contact_work():
    core, shape, inertia, mass, gravity = model()
    mv = core.mv
    normal = (mv.x + 2 * mv.y + 8 * mv.z).normalized()
    height = (normal & shape(normal)).square_root()
    contact = -shape(normal) / height
    rolling = (contact * core.origin.inverse() / 2).exp() >> core.Spin
    point = core.origin + contact
    reaction = point & (mv.x + 2 * mv.y + 3 * mv.z).dual()
    np.testing.assert_allclose(point.commutator(rolling).kernel, 0, atol=1e-11)
    np.testing.assert_allclose(rolling.adjugate()(reaction).kernel, 0, atol=1e-11)


def test_reduced_balance_matches_newton_euler_and_has_zero_energy_derivative():
    core, shape, inertia, mass, gravity_strength = model()
    mv = core.mv
    up = mv.z.dual()
    gravity = -up * gravity_strength
    motor = (mv.xz * 0.21 + mv.yz * 0.12 + mv.xy * 0.07).exp()
    spin = mv.yz * 0.3 - mv.xz * -0.7 + mv.xy * 1.2
    rate, acceleration = core.dynamics(motor, spin, shape, inertia, mass, gravity, up)
    normal, contact, height, rolling = core.contact_geometry(motor, shape, up)
    normal_rate = spin.commutator(normal)
    height_rate = (normal & shape(normal_rate)) / height
    contact_rate = -shape(normal_rate) / height - contact * height_rate / height
    rate_change = rolling(acceleration) + (contact_rate * core.origin.inverse()).commutator(spin)
    power = rate_change & inertia(rate)
    np.testing.assert_allclose((power + mass * gravity_strength * height_rate).kernel, 0, atol=1e-10)

    # Independently eliminate the contact reaction from Newton's two balances.
    normal_array = normal.cast(PGA3D.subspace("x y z")).kernel
    omega = np.array([0.3, -0.7, 1.2])
    lengths_squared = np.array([25, (17 / 16) ** 2, 1])
    angle = np.pi / 20
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    second_moment = rotation @ np.diag(mass * lengths_squared / 5) @ rotation.T
    moment = np.trace(second_moment) * np.eye(3) - second_moment
    height_array = np.sqrt(normal_array @ (lengths_squared * normal_array))
    lever = -lengths_squared * normal_array / height_array
    normal_change = np.cross(normal_array, omega)
    lever_change = (-lengths_squared * normal_change / height_array
                    - lever * (normal_array @ (lengths_squared * normal_change)) / height_array ** 2)
    velocity = np.cross(lever, omega)
    contact_moment = moment + mass * ((lever @ lever) * np.eye(3) - np.outer(lever, lever))
    torque = (np.cross(moment @ omega, omega)
              + mass * np.cross(lever, np.cross(lever_change, omega))
              - mass * np.cross(lever, np.cross(velocity, omega))
              + mass * gravity_strength * np.cross(lever, normal_array))
    expected = np.linalg.solve(contact_moment, torque)
    angular_axes = stack((mv.yz, -mv.xz, mv.xy))
    actual = -angular_axes.scalar_product(acceleration).kernel[:, 0]
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-11)


def test_mirroring_the_body_mirrors_its_rolling_dynamics():
    core, shape, inertia, mass, gravity_strength = model()
    mv = core.mv
    up = mv.z.dual()
    gravity = -up * gravity_strength
    motor = (mv.xz * 0.13 + mv.yz * 0.07 + mv.xy * 0.2).exp()
    spin = mv.xy * -1.5 + mv.yz * 0.2
    rate, acceleration = core.dynamics(motor, spin, shape, inertia, mass, gravity, up)
    mirror = (mv.x + mv.y / 3 + mv.z / 5).normalized()
    mirrored_rate, mirrored_acceleration = core.dynamics(
        mirror >> motor, mirror >> spin,
        mirror >> shape(mirror << core.Plane),
        mirror >> inertia(mirror << core.Rate),
        mass, mirror >> gravity, mirror >> up,
    )
    np.testing.assert_allclose(mirrored_rate.kernel, (mirror >> rate).kernel, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(mirrored_acceleration.kernel, (mirror >> acceleration).kernel,
                               atol=1e-10, rtol=1e-10)


def test_spin_reverses_and_energy_converges_with_the_time_step():
    coarse_dt, fine_dt = 0.05, 0.025
    duration = 10
    deck_fraction = 0.8
    _, _, _, _, coarse_energy = scenarios.launches(coarse_dt, round(duration / coarse_dt))
    outline, mass_points, placements, spins, fine_energy = scenarios.launches(fine_dt, round(duration / fine_dt))
    coarse_error = np.max(np.abs(coarse_energy.kernel / coarse_energy[0].kernel - 1))
    fine_error = np.max(np.abs(fine_energy.kernel / fine_energy[0].kernel - 1))
    assert fine_error < 0.001
    assert fine_error < coarse_error / 4

    # A tipped body spins up from rest; the spinning launch reverses direction.
    spin_about_vertical = -(placements >> spins).scalar_product(scenarios.mv.xy).kernel[..., 0]
    assert abs(spin_about_vertical[0, 0]) < 1e-11
    assert spin_about_vertical[-1, 0] > 1.4
    assert spin_about_vertical[0, 1] < -1.4
    assert spin_about_vertical[-1, 1] > 1.4

    # The massless top cut contains the masses and never touches the table.
    core = scenarios.core
    _, shape, _, _ = scenarios.body()
    _, contact, _, _ = core.contact_geometry(placements, shape, core.mv.z.dual())
    deck_height = deck_fraction * (core.mv.z & outline(core.mv.z.dual()))
    assert np.max(((core.mv.z & mass_points) - deck_height).kernel) < 0
    assert np.max(((core.mv.z & contact) - deck_height).kernel) < 0
