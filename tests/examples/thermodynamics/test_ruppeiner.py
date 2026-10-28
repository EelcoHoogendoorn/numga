"""The physical fluctuation metric, its equal-cost contours and curvature."""
import itertools

import numpy as np

from numga.backend.jax import derivative
from examples.thermodynamics.ruppeiner import core, render, scenarios


def test_metric_matches_heat_capacity_and_pressure_response():
    temps = np.array([1.02, 1.2, 1.8])
    vols = np.array([1, 0.8, 1.5])
    states = core.mv.t.dual() * temps + core.mv.v.dual() * vols
    form = core.metric(states, 9/8, 1/3, 1.5)
    pressure_change = derivative(lambda state: core.pressure(state, 9/8, 1/3))(states)
    thermal = form(core.mv.t.dual()) & core.mv.t.dual()
    mechanical = form(core.mv.v.dual()) & core.mv.v.dual()
    np.testing.assert_allclose(thermal.to_array(), 1.5 / temps**2, atol=1e-11)
    np.testing.assert_allclose(mechanical.to_array(), (-pressure_change(core.mv.v.dual()) / core.temperature(states)).to_array(), atol=1e-11)
    assert np.all(mechanical > 0)
    np.testing.assert_allclose((form(core.mv.t.dual()) & core.mv.v.dual()).to_array(), 0, atol=1e-11)


def test_curvature_matches_independent_closed_expression_and_flat_limit():
    temps, vols = np.array([1.002, 1.2, 1.8]), np.array([1, 0.8, 1.5])
    states = core.mv.t.dual() * temps + core.mv.v.dual() * vols
    a, b, capacity = 9/8, 1/3, 1.5
    bracket = temps * vols**3 - 2*a*(vols-b)**2
    expected = 2*a*(vols-b)**2*(temps*vols**3-a*(vols-b)**2) / (capacity*bracket**2)
    result = core.thermodynamic_curvature(states, a, b, capacity).to_array()
    np.testing.assert_allclose(result, expected, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(core.thermodynamic_curvature(states, 0, 0, capacity).to_array(), 0, atol=1e-10)
    assert abs(result[0] * 3 * (temps[0]-1)**2 - 1) < 0.01


def test_spinodal_is_where_the_mechanical_cost_vanishes_and_frames_are_drawn():
    volumes = np.array([0.8, 1, 1.5])
    # The spinodal: where the pressure's slope along the volume vanishes.
    spinodal = core.mv.t.dual() * (2 * 9 / 8 * (volumes - 1 / 3) ** 2 / volumes ** 3) + core.mv.v.dual() * volumes
    mechanical = core.metric(spinodal, 9/8, 1/3, 1.5)(core.mv.v.dual()) & core.mv.v.dual()
    np.testing.assert_allclose(mechanical.to_array(), 0, atol=1e-11)
    frames = render.fluctuations(scenarios.metric_of, scenarios.curvature_of, scenarios.pressure_of,
                                 scenarios.approach(np.linspace(1.8, 1.02, 4)))
    frame = next(itertools.islice(frames, 2, None))
    assert frame.ndim == 3 and frame.dtype == np.uint8
