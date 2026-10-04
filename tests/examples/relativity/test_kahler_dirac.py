"""The field over the whole algebra of the plane and time: the step is antisymmetric, and the scenes
pass their checks and draw."""

from __future__ import annotations

import numpy as np

from examples.relativity.kahler_dirac import core, render, scenarios


def test_the_step_from_time_is_minus_the_reverse_of_the_step_from_space():
    """In the reverse's scalar product, negative on the time blades, a time field paired with the step
    of a space field is the space field paired with the step of the time field: each step is minus the
    other's transpose, so the leapfrog keeps its energy."""
    grid = core.Grid(12)
    rng = np.random.default_rng(0)
    mass = core.mv.scalar(rng.uniform(0.0, 1.0, (len(grid.here), 1)))
    space = core.mv(core.Space, rng.normal(size=(len(grid.here), 4)))
    time = core.mv(core.Time, rng.normal(size=(len(grid.here), 4)))
    forward = time.reverse().scalar_product(core.step(grid, mass, core.Space)(space)).sum(axis=0)
    backward = space.reverse().scalar_product(core.step(grid, mass, core.Time)(time)).sum(axis=0)
    np.testing.assert_allclose(forward.to_array(), backward.to_array(), rtol=1e-12)


def test_scenes_pass_their_checks_and_draw():
    grid, space, time = scenarios.spreading()
    assert len(render.animate(grid, space[:, :1], time[:, :1], 2, 1)) == 1
    scenarios.barrier()
    grid, space, time = scenarios.standing()
    assert len(render.animate(grid, space, time, 6, 1)) == scenarios.PHASES
