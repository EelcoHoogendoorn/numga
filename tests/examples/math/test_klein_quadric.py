"""The four-line problem: the transversals from the Klein form meet all four lines, through the
points where the fourth line crosses the quadric of the other three, real or complex; the scenes
pass their checks and draw."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.math.klein_quadric import render, scenarios


def test_both_transversals_exist_whether_the_fourth_line_crosses_the_hyperboloid_or_not():
    """A steep line outside the waist crosses the hyperboloid and has two real transversals; one
    through the waist misses it, and its transversals are a complex-conjugate pair, still lines
    meeting all four."""
    three = scenarios.waist_lines(np.array([0.3, 2.3, 4.4]))
    lines, _, across, _ = scenarios.scene(three, scenarios.steep(np.array([1.4, 0.0])), 12)
    assert render.real(across[0].kernel).all() and not render.real(across[1].kernel).any()
    # The complex conjugate of one transversal is a multiple of the other.
    pair = np.stack([across[1, 0].kernel.conj(), across[1, 1].kernel])
    assert np.linalg.svd(pair, compute_uv=False)[-1] < 1e-12
    np.testing.assert_allclose((lines[1, :, None] ^ across[1, None, :]).kernel, 0.0, atol=1e-12)


def test_scenes_pass_their_checks_and_draw():
    scenes = list(scenarios.swing(4, 12))
    figure = render.draw(*scenes[0], azimuth=-60.0)
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
    assert len(render.animate(scenes)) == 4
