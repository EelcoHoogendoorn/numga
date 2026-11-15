"""Coverage tests for the five-dimensional commutator exponential and tennis-racket trajectories."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from numga import NumpyContext
from numga.algebra import Algebra

from examples.mechanics.tennis_racket import render, scenarios


def test_adjoint_is_the_open_commutator_in_five_dimensions():
    """Sandwiching bivectors by `turn.exp()` matches the matrix exponential of `2 * turn.commutator(Bivector)` in 5D."""
    context = NumpyContext(Algebra.from_pqr(5, 0, 0), dtype=np.float64)
    mv = context.multivector
    Bivector = context.algebra.gatype.bivector()                                # Bivector+
    n_planes = len(Bivector.output_subspace)
    rng = np.random.default_rng(0)
    turn = mv.bivector(rng.normal(scale=0.3, size=n_planes))                    # Bivector+
    adjoint = turn.commutator(Bivector) * 2                                     # Bivector+ <- Bivector+
    assert adjoint.kernel.shape == (10, 10)
    np.testing.assert_allclose(expm(adjoint.kernel), (turn.exp() >> Bivector).kernel, atol=1e-10)


def test_a_step_of_each_integrator_and_the_figures_draw():
    """One integration step in 3D runs every integrator and renders both trajectory and drift figures."""
    rates, energies = scenarios.spinning_racket(3, scenarios.step, scenarios.step, scenarios.seed)
    assert ((energies / energies[0] - 1).abs() < 1e-3).all()
    fig_trajectories = render.trajectories_figure(rates, energies)
    curves = list(scenarios.integrator_comparison((3,), scenarios.step, scenarios.step, scenarios.seed))
    fig_comparison = render.integrator_comparison_figure(curves, scenarios.step)
    plt.close(fig_trajectories)
    plt.close(fig_comparison)
