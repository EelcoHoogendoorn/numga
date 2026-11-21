"""Rolling constraints, instantaneous power balance, mirror equivariance, and rendering checks cover the rattleback example."""

from itertools import islice

import matplotlib.pyplot as plt
import numpy as np

from examples.mechanics.rattleback import core, render, scenarios
from numga import dim, stack


def test_rolling_annihilates_contact_motion_and_contact_work():
    shape, centre, _, _, _, _ = scenarios.rattleback_body()
    mv = core.mv
    normal = (mv.x + 2 * mv.y + 8 * mv.z).normalized()
    height = (normal & shape(normal)).square_root()
    contact = centre - shape(normal) / height
    contact_shift = (contact * core.origin.inverse() / 2).exp()
    rolling = contact_shift >> core.Spin
    point = core.origin + contact
    reaction = point & (mv.x + 2 * mv.y + 3 * mv.z).dual()
    np.testing.assert_allclose(point.commutator(rolling).kernel, 0, atol=1e-11)
    np.testing.assert_allclose(rolling.adjugate()(reaction).kernel, 0, atol=1e-11)


def test_reduced_balance_matches_newton_euler_and_conserves_instantaneous_energy():
    shape, centre, inertia, _, _, _ = scenarios.rattleback_body()
    mv = core.mv
    up = core.up
    gravity = -up * scenarios.gravity_strength
    motor = (mv.xz * 0.21 + mv.yz * 0.12 + mv.xy * 0.07).exp()
    spin = (mv.yz * 0.3 - mv.xz * -0.7 + mv.xy * 1.2) / dim.time
    rate, acceleration = core.dynamics(motor, spin, shape, centre, inertia, scenarios.mass, gravity, up)
    normal, support, height, rolling = core.contact_geometry(motor, shape, centre, up)
    normal_rate = spin.commutator(normal)
    height_rate = (normal & shape(normal_rate)) / height
    contact_rate = -shape(normal_rate) / height - support * height_rate / height
    rate_change = rolling(acceleration) + (contact_rate * core.origin.inverse()).commutator(spin)
    power = rate_change & inertia(rate)
    rise = height_rate - (normal_rate & centre)
    assert ((power + scenarios.mass * scenarios.gravity_strength * rise) / (dim.mass * dim.length**2 / dim.time**3)).abs() < 1e-9

    angular_axes = stack((mv.yz, -mv.xz, mv.xy))
    moment = ((angular_axes[:, None] & inertia(angular_axes[None, :])) / (dim.mass * dim.length**2)).to_array()
    offset = ((core.mirrors & centre) / dim.length).to_array()
    normal_array = (core.mirrors | normal).to_array()
    omega = np.array([0.3, -0.7, 1.2])
    lengths_squared = (scenarios.semi_axes / dim.length).to_array() ** 2
    mass_value = float((scenarios.mass / dim.mass).to_array())
    gravity_value = float((scenarios.gravity_strength / (dim.length / dim.time**2)).to_array())
    height_array = np.sqrt(normal_array @ (lengths_squared * normal_array))
    lever = offset - lengths_squared * normal_array / height_array
    normal_change = np.cross(normal_array, omega)
    support_array = -lengths_squared * normal_array / height_array
    lever_change = (
        -lengths_squared * normal_change / height_array
        - support_array * (normal_array @ (lengths_squared * normal_change)) / height_array**2
    )
    velocity = np.cross(lever, omega)
    contact_moment = moment + mass_value * ((lever @ lever) * np.eye(3) - np.outer(lever, lever))
    torque = (
        np.cross(moment @ omega, omega)
        + mass_value * np.cross(lever, np.cross(lever_change, omega))
        - mass_value * np.cross(lever, np.cross(velocity, omega))
        + mass_value * gravity_value * np.cross(lever, normal_array)
    )
    expected = np.linalg.solve(contact_moment, torque)
    actual = (-angular_axes.scalar_product(acceleration) * dim.time**2).to_array()
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_mirroring_the_body_mirrors_its_rolling_dynamics():
    shape, centre, inertia, _, _, _ = scenarios.rattleback_body()
    mv = core.mv
    up = core.up
    gravity = -up * scenarios.gravity_strength
    motor = (mv.xz * 0.13 + mv.yz * 0.07 + mv.xy * 0.2).exp()
    spin = (mv.xy * -1.5 + mv.yz * 0.2) / dim.time
    rate, acceleration = core.dynamics(motor, spin, shape, centre, inertia, scenarios.mass, gravity, up)
    mirror = (mv.x + mv.y / 3 + mv.z / 5).normalized()
    mirrored_rate, mirrored_acceleration = core.dynamics(
        mirror >> motor, mirror >> spin,
        mirror >> shape(mirror << core.Plane), mirror >> centre,
        mirror >> inertia(mirror << core.Rate),
        scenarios.mass, mirror >> gravity, mirror >> up,
    )
    np.testing.assert_allclose(mirrored_rate.kernel, (mirror >> rate).kernel, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(mirrored_acceleration.kernel, (mirror >> acceleration).kernel, atol=1e-10, rtol=1e-10)


def test_single_step_and_render():
    shape, centre, inertia, wood, balls, samples = scenarios.rattleback_body()
    still = core.mv.scalar(0.0, unit=1 / dim.time)
    states = list(islice(scenarios.launch_trajectory(scenarios.rocking_tilt, still, still, scenarios.dt, shape, centre, inertia), 2))
    assert len(states) == 2

    setup_fig = render.setup_figure(wood, balls, samples, states[0][0])
    plt.close(setup_fig)

    frame = next(render.sized_frames(wood, balls, samples, states, 64))
    assert frame.ndim == 3 and frame.shape[1] == 64 and frame.shape[2] == 3
    assert frame.dtype == np.uint8
