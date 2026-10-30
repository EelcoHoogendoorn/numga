"""The hollow solid's support, dissipative contact, and reflection symmetry."""
from __future__ import annotations

import numpy as np
import pytest

from numga import stack
from examples.mechanics.tippe_top import core, scenarios


@pytest.fixture(scope="module")
def body():
    return scenarios.body()


def test_contacts_lie_on_the_parts_and_have_horizontal_tangents(body):
    n_latitudes, n_longitudes = 16, 32
    latitudes = (np.arange(n_latitudes) + 0.5) * np.pi / n_latitudes
    longitudes = np.arange(n_longitudes) * 2 * np.pi / n_longitudes
    turns = (core.mv.xy * (-longitudes[None, :] / 2)).exp() * (core.mv.xz * (-latitudes[:, None] / 2)).exp()
    normals = (turns >> core.up).reshape(-1)
    parts = body.contacts[:, None]
    points = core.contact.lowest_point(parts, normals)
    np.testing.assert_allclose((parts(points) & points).to_array(), 0, atol=1e-10)
    tangent = parts(points)
    direction = tangent.dual() - core.origin * (core.mv.w & tangent.dual())
    np.testing.assert_allclose((direction & normals).kernel, 0, atol=1e-10)

    # The cavity never removes the shell's exposed support, and the stem is behind the contacts.
    heights = normals.dual_inverse() & points
    cut = (body.cavity(points[0]) & points[0]) < 0
    assert not np.any(cut & (heights[0] < heights[1]))
    stem = core.contact.lowest_point(body.stem, normals)
    assert np.all((normals.dual_inverse() & stem) >= heights.min(axis=0) - 1e-10)


def test_gravity_and_friction_transfer_and_dissipate_energy(body):
    motor, rate = core.launch(body, scenarios.spins, scenarios.lean)
    initial_energy = (body.inertia(rate) & rate) / 2 + scenarios.gravity * (core.mv.z & (motor >> body.mass))
    end_motor, end_rate = core.advance(body, motor, rate, scenarios.gravity,
                                      scenarios.static_friction, scenarios.dynamic_friction,
                                      scenarios.step, scenarios.n_steps_per_frame)
    final_energy = (body.inertia(end_rate) & end_rate) / 2 + scenarios.gravity * (core.mv.z & (end_motor >> body.mass))
    assert np.all(final_energy < initial_energy)
    assert np.all(final_energy > initial_energy * 0.9)

    # With the table far below, the mass centre follows its gravitational parabola.
    height, duration, n_steps = 10, 0.1, 25
    lift = (core.up * core.origin.inverse() * height / 2).exp()
    airborne = lift * motor
    final_motor, _ = core.advance(body, airborne, rate * 0, scenarios.gravity, 0., 0., duration / n_steps, n_steps)
    fall = core.mv.z & ((final_motor >> body.mass.normalized()) - (airborne >> body.mass.normalized()))
    np.testing.assert_allclose(fall.to_array(), -scenarios.gravity * duration**2 / 2, atol=3e-4)


def test_mirroring_the_cavity_and_launch_mirrors_the_motion(body):
    mirror = core.mv.y
    reflected_inertia = mirror >> body.inertia(mirror << core.Rate)
    reflected = core.Body(*(mirror >> part(mirror << core.Point) for part in (body.shell, body.cavity, body.stem, body.tip)),
                          mirror >> body.mass, reflected_inertia, reflected_inertia.inverse())
    pair = core.Body(*(stack((original, mirrored)) for original, mirrored in zip(body, reflected)))
    motor, rate = core.launch(body, scenarios.spins[:1], scenarios.lean)
    # This lean lies in the mirror plane, so both placements coincide at release.
    motors = motor.broadcast_to((2,))
    rates = stack((rate[0], mirror >> rate[0])).retyped(core.Rate)
    following, velocities = core.advance(pair, motors, rates, scenarios.gravity,
                                         scenarios.static_friction, scenarios.dynamic_friction,
                                         scenarios.step, scenarios.n_steps_per_frame)
    np.testing.assert_allclose(((mirror >> (following[0] >> core.origin)) - (following[1] >> core.origin)).kernel, 0, atol=1e-8)
    np.testing.assert_allclose(((mirror >> (following[0] >> core.up)) - (following[1] >> core.up)).kernel, 0, atol=1e-8)
    np.testing.assert_allclose(((mirror >> velocities[0]) - velocities[1]).kernel, 0, atol=1e-8)

    # Independent launches have identical motion whether run separately or in a batch.
    alone, alone_rate = core.advance(body, motor, rate, scenarios.gravity,
                                     scenarios.static_friction, scenarios.dynamic_friction,
                                     scenarios.step, scenarios.n_steps_per_frame)
    np.testing.assert_allclose((alone[0] - following[0]).kernel, 0, atol=1e-8)
    np.testing.assert_allclose((alone_rate[0] - velocities[0]).kernel, 0, atol=1e-8)
