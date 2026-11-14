"""Tests the E8 root orbits, invariant rotation planes, and planar renderers."""

import numpy as np

from examples.math.e8 import core, render, scenarios


def test_e8_root_system_planes_and_renderers() -> None:
    """Checks the 240 roots, winding rates, fractional rotor step, and both planar renderers."""
    roots, plane_x, plane_y, planes, rates, coxeter, seeds = scenarios.root_system()
    assert ((roots.scalar_norm_squared() - 2).abs() < 1e-6).all()
    assert (((coxeter >> roots[-1]) - seeds).scalar_norm_squared() < 1e-6).all()
    assert ((rates - np.array([1, 7, 11, 13])).abs() < 1e-6).all()

    rotor = core.step_rotor(planes, rates, core.coxeter_order, 2)             # [] Rotor+
    advance = rotor >> core.Vector                                            # [] Vector+ <- Vector+
    two_substeps = core.orbit(advance, seeds, 3)
    _ = next(two_substeps)
    _ = next(two_substeps)
    step_one = next(two_substeps)                                             # [8] Vector+
    assert ((step_one - roots[1]).scalar_norm_squared() < 1e-6).all()

    render.draw_roots(roots, plane_x[0], plane_y[0])
    next(render.planes_animation(roots, plane_x, plane_y, 2, core.orbit(advance, seeds, 1)))
