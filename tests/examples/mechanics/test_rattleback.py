"""Rolling constraints, work balance and spin reversal of a rattleback."""

import numpy as np

from examples.mechanics.rattleback import scenarios
from numga import stack
from numga.algebras import PGA3D


def model():
    shape, centre, inertia, _, _ = scenarios.body()
    return scenarios.core, shape, centre, inertia, scenarios.MASS, scenarios.GRAVITY


def test_rolling_annihilates_contact_motion_and_contact_work():
    core, shape, centre, inertia, mass, gravity = model()
    mv = core.mv
    normal = (mv.x + 2 * mv.y + 8 * mv.z).normalized()
    height = (normal & shape(normal)).square_root()
    contact = centre - shape(normal) / height
    rolling = (contact * core.origin.inverse() / 2).exp() >> core.Spin
    point = core.origin + contact
    reaction = point & (mv.x + 2 * mv.y + 3 * mv.z).dual()
    np.testing.assert_allclose(point.commutator(rolling).kernel, 0, atol=1e-11)
    np.testing.assert_allclose(rolling.adjugate()(reaction).kernel, 0, atol=1e-11)


def test_reduced_balance_matches_newton_euler_and_has_zero_energy_derivative():
    core, shape, centre, inertia, mass, gravity_strength = model()
    mv = core.mv
    up = mv.z.dual()
    gravity = -up * gravity_strength
    motor = (mv.xz * 0.21 + mv.yz * 0.12 + mv.xy * 0.07).exp()
    spin = mv.yz * 0.3 - mv.xz * -0.7 + mv.xy * 1.2
    rate, acceleration = core.dynamics(motor, spin, shape, centre, inertia, mass, gravity, up)
    normal, support, height, rolling = core.contact_geometry(motor, shape, centre, up)
    normal_rate = spin.commutator(normal)
    height_rate = (normal & shape(normal_rate)) / height
    contact_rate = -shape(normal_rate) / height - support * height_rate / height
    rate_change = rolling(acceleration) + (contact_rate * core.origin.inverse()).commutator(spin)
    power = rate_change & inertia(rate)
    # The centre of mass rises as the shell's centre does, less as the shell's centre turns about it.
    rise = height_rate - (normal_rate & centre)
    np.testing.assert_allclose((power + mass * gravity_strength * rise).kernel, 0, atol=1e-10)

    # Independently eliminate the contact reaction from Newton's two balances, with the moment of inertia read off
    # as the pairing of unit spins with their momenta, and the shell's centre off the centre of mass.
    angular_axes = stack((mv.yz, -mv.xz, mv.xy))
    moment = (angular_axes[:, None] & inertia(angular_axes[None, :])).kernel[..., 0]
    offset = (stack((mv.x, mv.y, mv.z)) & centre).kernel[:, 0]
    normal_array = normal.cast(PGA3D.subspace("x y z")).kernel
    omega = np.array([0.3, -0.7, 1.2])
    lengths_squared = scenarios.SEMI_AXES**2
    height_array = np.sqrt(normal_array @ (lengths_squared * normal_array))
    lever = offset - lengths_squared * normal_array / height_array
    normal_change = np.cross(normal_array, omega)
    support = -lengths_squared * normal_array / height_array
    lever_change = (-lengths_squared * normal_change / height_array
                    - support * (normal_array @ (lengths_squared * normal_change)) / height_array ** 2)
    velocity = np.cross(lever, omega)
    contact_moment = moment + mass * ((lever @ lever) * np.eye(3) - np.outer(lever, lever))
    torque = (np.cross(moment @ omega, omega)
              + mass * np.cross(lever, np.cross(lever_change, omega))
              - mass * np.cross(lever, np.cross(velocity, omega))
              + mass * gravity_strength * np.cross(lever, normal_array))
    expected = np.linalg.solve(contact_moment, torque)
    actual = -angular_axes.scalar_product(acceleration).kernel[:, 0]
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-11)


def test_mirroring_the_body_mirrors_its_rolling_dynamics():
    core, shape, centre, inertia, mass, gravity_strength = model()
    mv = core.mv
    up = mv.z.dual()
    gravity = -up * gravity_strength
    motor = (mv.xz * 0.13 + mv.yz * 0.07 + mv.xy * 0.2).exp()
    spin = mv.xy * -1.5 + mv.yz * 0.2
    rate, acceleration = core.dynamics(motor, spin, shape, centre, inertia, mass, gravity, up)
    mirror = (mv.x + mv.y / 3 + mv.z / 5).normalized()
    mirrored_rate, mirrored_acceleration = core.dynamics(
        mirror >> motor, mirror >> spin,
        mirror >> shape(mirror << core.Plane), mirror >> centre,
        mirror >> inertia(mirror << core.Rate),
        mass, mirror >> gravity, mirror >> up,
    )
    np.testing.assert_allclose(mirrored_rate.kernel, (mirror >> rate).kernel, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(mirrored_acceleration.kernel, (mirror >> acceleration).kernel,
                               atol=1e-10, rtol=1e-10)


def test_spin_reverses_and_energy_converges_with_the_time_step():
    coarse_dt, fine_dt = 0.05, 0.025
    duration = 10
    _, _, _, _, coarse_energy = scenarios.launches(coarse_dt, round(duration / coarse_dt))
    built, _, placements, spins, fine_energy = scenarios.launches(fine_dt, round(duration / fine_dt))
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

    # The table only ever touches the painted underside: the touching point lies inside the deck and the cuts and
    # outside the balls, every other quadric of the wood.
    core = scenarios.core
    shape, centre, _, _, _ = scenarios.body()
    _, support, _, _ = core.contact_geometry(placements, shape, centre, core.mv.z.dual())
    contact = core.origin + centre + support                                  # [steps + 1, cases] Point
    others = built.wood()[1:]                                                 # [quadrics] Quadric
    assert np.max((others[:, None, None](contact) & contact).kernel) < 0
