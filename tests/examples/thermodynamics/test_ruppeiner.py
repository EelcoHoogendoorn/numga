"""Smoke test stepping the cooling path near the critical point and rendering one fluctuation-metric frame."""
import numpy as np

from examples.thermodynamics.ruppeiner import render, scenarios


def test_fluctuations_run_and_draw():
    next(render.fluctuations(scenarios.metric_of, scenarios.curvature_of, scenarios.pressure_of,
                             scenarios.cooling_path(np.array([1.8]))))
