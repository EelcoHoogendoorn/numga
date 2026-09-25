"""The Hopf fibration: the fibre over a direction is the top eigenspace of the direction paired with
the Hopf form, and the scenarios pass their checks and draw."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.math.hopf import core, render, scenarios


def test_the_fibre_is_the_top_eigenspace_of_the_paired_form():
    """(direction | HOPF) has eigenvalues -1, -1, 1, 1, and both spinors of the top pair point along
    the direction; turning a spinor on the right in the xy plane keeps it on its fibre."""
    direction = core.mv.vector(np.array([0.3, -0.5, 0.8])).normalized()
    values, spinors = (direction | core.HOPF).eigh()
    np.testing.assert_allclose(values.to_array(), [-1.0, -1.0, 1.0, 1.0], atol=1e-12)
    top = spinors[-2:]
    np.testing.assert_allclose((core.HOPF(top, top) - direction).kernel, 0.0, atol=1e-12)
    turned = core.fibre(spinors[-1], np.array([0.4, 2.1]))
    np.testing.assert_allclose((core.HOPF(turned, turned) - direction).kernel, 0.0, atol=1e-9)


def test_scenarios_pass_their_checks_and_draw():
    figures = (render.draw_tori(*scenarios.tori(np.pi * np.array([0.8, 0.5]), 5, 120)),
               render.draw_lift(*scenarios.lift(2.2, 200)))
    for figure in figures:
        assert isinstance(figure, plt.Figure)
        plt.close(figure)
    assert len(render.animate_sweep(scenarios.sweep(3, 60))) == 3
