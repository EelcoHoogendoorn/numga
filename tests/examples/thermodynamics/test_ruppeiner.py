"""The van der Waals fluid's fluctuation metric: one state on the approach to the critical point, its first frame
drawn."""
import numpy as np

from examples.thermodynamics.ruppeiner import render, scenarios


def test_fluctuations_run_and_draw():
    next(render.fluctuations(scenarios.metric_of, scenarios.curvature_of, scenarios.pressure_of,
                             scenarios.approach(np.array([1.8]))))
