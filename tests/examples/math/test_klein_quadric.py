"""The four-line problem: the transversals from the Klein form meet all four lines, through the
points where the fourth line crosses the quadric of the other three, real or complex."""

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


def test_the_transversals_meet_all_four_lines_and_pass_through_the_crossings():
    three = scenarios.waist_lines(np.array([0.5, 2.0, 4.0]))
    lines, rulings, across, crossing = scenarios.scene(three, scenarios.steep(np.array([1.2, -0.9])), 12)
    surface = scenarios.core.ruled_quadric(three[0], three[1], three[2])
    np.testing.assert_allclose((across ^ across).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((lines[..., :, None] ^ across[..., None, :]).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((surface(crossing) & crossing).kernel, 0.0, atol=1e-10)
    off = np.abs((crossing[..., :, None] & across[..., None, :]).kernel).max(axis=-1)
    np.testing.assert_allclose(off.min(axis=-1), 0.0, atol=1e-10)


def test_scenes_draw():
    scenes = list(scenarios.swing(4, 12))
    figure = render.draw(*scenes[0], azimuth=render.AZIMUTH)
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
    assert len(render.animate(scenes)) == 4
