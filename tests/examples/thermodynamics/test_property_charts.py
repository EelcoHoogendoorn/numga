"""A cooling coil and a reheater: the air's path through them at two steps."""
from examples.thermodynamics.property_charts import scenarios


def test_conditioning_runs():
    path, *_ = scenarios.conditioning(2)
    next(scenarios.passing(path))
