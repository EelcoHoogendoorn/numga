"""Potential flow past a Joukowski wing."""

import numpy as np

from examples.mechanics.wing import core, scenarios


def test_the_potential_has_no_derivative_and_the_wing_lifts_by_its_circulation():
    pitched = core.pitched(scenarios.wing(), scenarios.HIGHEST_ATTACK)
    stream = scenarios.stream()
    plane = core.rings(pitched, 20, scenarios.ANGLES, scenarios.REACH)
    edge = scenarios.trailing_edge(pitched)
    flow = core.flow(pitched, stream, plane, edge)

    # The potential's derivative vanishes, the Cauchy–Riemann condition.
    np.testing.assert_allclose(core.derivative(flow.potential_gradient).kernel, 0.0, atol=1e-9)
    # The stream function changes across a small step by the velocity's flux through it.
    step = core.mv.vector([0.3, -0.2]) * 1e-6
    forward, backward = core.flow(pitched, stream, plane + step, edge), core.flow(pitched, stream, plane - step, edge)
    crossing = (forward.points - backward.points) / 2
    np.testing.assert_allclose(((forward.stream - backward.stream) / 2 - (flow.velocity ^ crossing)).kernel, 0.0, atol=1e-12)
    # Far away the flow is the stream.
    far = core.flow(pitched, stream, core.rings(pitched, 2, 8, 1e4)[-1], edge)
    np.testing.assert_allclose((far.velocity - stream).kernel, 0.0, atol=1e-3)
    # The pressure around the surface is the Kutta–Joukowski lift, with no drag.
    surface = core.flow(pitched, stream, core.rings(pitched, 1, scenarios.ANGLES, 1.0)[0], edge)
    next_idx = (np.arange(scenarios.ANGLES) + 1) % scenarios.ANGLES
    steps = surface.points[next_idx] - surface.points
    speed_squared = surface.velocity | surface.velocity
    force = (0.5 * scenarios.DENSITY * (speed_squared + speed_squared[next_idx]) / 2 * -steps.dual()).sum(axis=-1)
    np.testing.assert_allclose((force - core.lift(pitched, stream, scenarios.DENSITY)).kernel, 0.0, atol=1e-3)
