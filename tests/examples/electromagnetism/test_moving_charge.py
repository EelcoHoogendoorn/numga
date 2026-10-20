"""A soft charge moving near the speed of light, and a point charge circling and shedding waves."""

import numpy as np

from examples.electromagnetism.moving_charge import core, scenarios


def test_maxwell_by_incidence_with_the_metric_in_the_vacuum():
    moving = scenarios.charge(scenarios.TOP_RAPIDITY)
    events = core.grid(scenarios.HALF_WIDTH, scenarios.HALF_HEIGHT, 15, 10)  # [rows, columns] Antivector

    # The field's derivative vanishes.
    gradients = core.field_gradient(moving, events)
    np.testing.assert_allclose((core.Vector ^ gradients(core.Antivector)).trace(1, 2).kernel, 0.0, atol=1e-12)
    # The field's gradient is the derivative of the potential's field along each step.
    potentials = core.potential_gradient(moving, events)
    step, size = core.mv.x.dual() + 0.4 * core.mv.y.dual() - 0.7 * core.time, 1e-6
    ahead = core.field(core.potential_gradient(moving, events + size * step))
    behind = core.field(core.potential_gradient(moving, events - size * step))
    np.testing.assert_allclose((gradients(step) - (ahead - behind) / (2 * size)).kernel, 0.0, atol=1e-5)
    # The potential keeps the Lorenz gauge: the derivative of its dual vanishes.
    np.testing.assert_allclose((core.Vector ^ potentials(core.Antivector).dual()).trace(1, 2).kernel, 0.0, atol=1e-12)
    # The current is the Plummer density carried along the charge's time direction, and positive.
    at_rest = moving.boost << events
    separations = at_rest - (core.mv.t & at_rest) * core.time
    softened = scenarios.CORE_RADIUS**2 - (separations | separations)
    density = 3 * scenarios.CHARGE * scenarios.CORE_RADIUS**2 / (4 * np.pi) / (softened * softened * softened.square_root())
    current = core.current(gradients)
    np.testing.assert_allclose((current - density * (moving.boost >> core.time)).kernel, 0.0, atol=1e-9)
    assert np.all((core.mv.t & current).to_array() > 0)


def test_the_circling_charge_is_seen_along_light_rays():
    orbit = core.Orbit(scenarios.ORBIT_CHARGE, scenarios.ORBIT_RADIUS, scenarios.ORBIT_SPEEDS[-1])
    events = core.grid(scenarios.ORBIT_HALF_WIDTH, scenarios.ORBIT_HALF_WIDTH, 12, 12) + core.time * 3.0

    # The light from the charge at the retarded time reaches each event.
    position, _, _ = core.worldline(orbit, core.retarded(orbit, events))
    rays = events - position
    np.testing.assert_allclose((rays | rays).kernel, 0.0, atol=1e-10)
    # The radiating potential keeps the Lorenz gauge.
    potentials = core.orbit_potential_gradient(orbit, events)
    np.testing.assert_allclose((core.Vector ^ potentials(core.Antivector).dual()).trace(1, 2).kernel, 0.0, atol=1e-12)
