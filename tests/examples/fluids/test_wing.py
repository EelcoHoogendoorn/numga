"""Potential flow past a Joukowski wing."""

import numpy as np

from examples.fluids.wing import core, scenarios


def test_the_velocity_has_no_derivative_and_the_wing_lifts_by_its_circulation():
    pitched = scenarios.wing(scenarios.HIGHEST_ATTACK)
    stream = scenarios.stream
    plane = core.rings(pitched, 20, scenarios.N_ANGLES, scenarios.OUTER_DISTANCE)
    edge = scenarios.trailing_edge(pitched)
    _, velocity, _, _ = core.flow(pitched, stream, plane, edge)

    # Past the cylinder the velocity has neither divergence nor vorticity.
    _, gradient, _ = core.cylinder(pitched, stream, plane)
    np.testing.assert_allclose(core.derivative(gradient).kernel, 0.0, atol=1e-12)
    # A step changes the stream function by the velocity's flux through it, at the wing as at the cylinder.
    step = core.mv.vector([0.3, -0.2]).dual() * 1e-6
    forward_points, _, _, forward_stream = core.flow(pitched, stream, plane + step, edge)
    backward_points, _, _, backward_stream = core.flow(pitched, stream, plane - step, edge)
    crossing = (forward_points - backward_points) / 2
    np.testing.assert_allclose(((forward_stream - backward_stream) / 2 - (velocity ^ crossing)).kernel, 0.0, atol=1e-12)
    # Far away the flow is the stream.
    _, far_velocity, _, _ = core.flow(pitched, stream, core.rings(pitched, 2, 8, 1e4)[-1], edge)
    np.testing.assert_allclose((far_velocity - stream).kernel, 0.0, atol=1e-3)
    # The pressure around the surface, each step's swept area in the sense the surface winds, is the
    # Kutta–Joukowski lift, with no drag.
    surface, surface_velocity, _, _ = core.flow(pitched, stream, core.rings(pitched, 1, scenarios.N_ANGLES, 1.0)[0], edge)
    next_idx = (np.arange(scenarios.N_ANGLES) + 1) % scenarios.N_ANGLES
    steps = surface[next_idx] - surface
    speed_squared = surface_velocity.dot(surface_velocity)
    pressure = -0.5 * scenarios.DENSITY * (speed_squared + speed_squared[next_idx]) / 2
    winding = (surface & surface[next_idx]).sum(axis=-1)
    force = (pressure * (steps & core.Point)).sum(axis=-1) * (winding / winding.norm())
    np.testing.assert_allclose((force - (core.lift(pitched, stream, scenarios.DENSITY) & core.Point)).kernel, 0.0, atol=1e-3)
