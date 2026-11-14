"""The transversals from the Klein form meet all four lines and pass through the points where the
fourth line crosses the quadric of the other three, whether real or complex."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.math.klein_quadric import core, render, scenarios


def test_both_transversals_exist_whether_the_fourth_line_crosses_the_hyperboloid_or_not():
    """A steep line outside the waist crosses the hyperboloid and has two real transversals, whereas
    one through the waist misses the surface and yields a complex-conjugate pair of transversals."""
    three_lines = scenarios.waist_lines(np.array([0.3, 2.3, 4.4]))
    lines, transversals, _ = scenarios.configuration(three_lines, scenarios.fourth_lines(np.array([1.4, 0.0])))
    basis_axes = stack((core.mv.xy, core.mv.yz, core.mv.zx, core.mv.xw, core.mv.yw, core.mv.zw)).dual()
    components = transversals[..., None] & basis_axes
    is_real = (components - components.real()).abs().max(axis=-1) <= 1e-6 * components.abs().max(axis=-1)
    assert is_real[0].all() and not is_real[1].any()
    # Conjugating one complex transversal produces a scalar multiple of the other.
    conjugate_first = -(transversals[1, 0] - 2 * transversals[1, 0].real())
    ratio = (conjugate_first & basis_axes[0]) / (transversals[1, 1] & basis_axes[0])
    assert (((conjugate_first - ratio * transversals[1, 1])[..., None] & basis_axes).abs() < 1e-11).all()
    assert (((lines[1, :, None] ^ transversals[1, None, :]).dual()).abs() < 1e-11).all()


def test_the_transversals_meet_all_four_lines_and_pass_through_the_crossings():
    three_lines = scenarios.waist_lines(np.array([0.5, 2.0, 4.0]))
    lines, transversals, crossings = scenarios.configuration(
        three_lines, scenarios.fourth_lines(np.array([1.2, -0.9]))
    )
    surface = ((core.Point & three_lines[0]) ^ three_lines[2]) & three_lines[1]
    basis_points = stack((core.mv.x, core.mv.y, core.mv.z, core.mv.w)).dual()
    assert ((transversals ^ transversals).dual().abs() < 1e-11).all()
    assert (((lines[..., :, None] ^ transversals[..., None, :]).dual()).abs() < 1e-11).all()
    assert ((surface(crossings) & crossings).abs() < 1e-9).all()
    incidence_residuals = (
        ((crossings[..., :, None] & transversals[..., None, :])[..., None] & basis_points).abs().max(axis=-1)
    )
    assert (incidence_residuals.min(axis=-1) < 1e-9).all()


def test_one_frame_of_the_swing_draws():
    scenes = list(scenarios.swing_scenes(1, 3))
    figure = render.scene_figure(*scenes[0])
    plt.close(figure)
    next(render.swing_frames(1, iter(scenes)))
