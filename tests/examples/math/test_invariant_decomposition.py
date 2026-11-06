"""A six-dimensional rotation split into its three planes by its spectrum and by its wedge powers, a rigid motion
split into its placed planes, and the turned point's orbit drawn, at the fewest times."""

from __future__ import annotations

from examples.math.invariant_decomposition import core, render, scenarios


def test_the_rotation_and_the_motion_split_and_the_orbit_draws():
    bivector, start, motion = scenarios.samples(scenarios.SEED)
    parts, points, circles = scenarios.tangle(bivector, start, 2)
    core.from_wedge_powers(bivector)
    core.placed(motion)
    next(render.animate(points, circles, parts, 1))
