"""A soft charge moving near the speed of light, and a point charge circling and shedding waves."""

import numpy as np

from examples.electromagnetism.moving_charge import core, scenarios


def test_maxwell_with_and_without_the_metric():
    moving = scenarios.charge(scenarios.TOP_RAPIDITY)
    events = core.grid(scenarios.HALF_WIDTH, scenarios.HALF_HEIGHT, 15, 10)  # [rows, columns] Vector

    # The potential's derivative is the field alone: the Lorenz-gauge scalar vanishes.
    potentials = core.potential_gradient(moving, events)
    np.testing.assert_allclose((core.potential_derivative(potentials) - core.field(potentials)).kernel, 0.0, atol=1e-12)
    # The metric-free trace agrees with the metric contraction, and gives no magnetic charge.
    gradients = core.field_gradient(moving, events)
    current, magnetic_source = core.sources(gradients)
    np.testing.assert_allclose((core.field_derivative(gradients) - current).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(magnetic_source.kernel, 0.0, atol=1e-12)
    # The current is the Plummer density carried along the charge's time direction.
    at_rest = moving.boost << events
    separations = at_rest - (at_rest | core.mv.t) * core.mv.t
    softened = scenarios.CORE_RADIUS**2 - (separations | separations)
    density = 3 * scenarios.CHARGE * scenarios.CORE_RADIUS**2 / (4 * np.pi) / (softened * softened * softened.square_root())
    np.testing.assert_allclose((current - density * (moving.boost >> core.mv.t)).kernel, 0.0, atol=1e-9)


def test_the_circling_charge_is_seen_along_light_rays():
    orbit = core.Orbit(scenarios.ORBIT_CHARGE, scenarios.ORBIT_RADIUS, scenarios.ORBIT_SPEEDS["fast"])
    events = core.grid(scenarios.ORBIT_HALF_WIDTH, scenarios.ORBIT_HALF_WIDTH, 12, 12) + core.mv.t * 3.0

    # The light from the charge at the retarded time reaches each event.
    position, _, _ = core.worldline(orbit, core.retarded(orbit, events))
    rays = events - position
    np.testing.assert_allclose((rays | rays).kernel, 0.0, atol=1e-10)
    # The radiating potential keeps the Lorenz gauge.
    potentials = core.orbit_potential_gradient(orbit, events)
    np.testing.assert_allclose((core.potential_derivative(potentials) - core.field(potentials)).kernel, 0.0, atol=1e-12)
