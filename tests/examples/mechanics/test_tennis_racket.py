"""The sandwich by a rotor is the exponential of its commutator, in five dimensions; and a racket takes one step
of each integrator, and its figures draw."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from numga import NumpyContext
from numga.algebra import Algebra

from examples.mechanics.tennis_racket import render, scenarios


def test_adjoint_is_the_open_commutator_in_five_dimensions():
    """The sandwich by h.exp() on bivectors equals the exponential of the commutator with h, on the ten-dimensional
    bivectors of five dimensions."""
    context = NumpyContext(Algebra.from_pqr(5, 0, 0), dtype=np.float64)
    mv = context.multivector
    Bivector = context.algebra.gatype.bivector()
    rng = np.random.default_rng(0)
    H = mv.bivector(rng.normal(scale=0.3, size=len(Bivector.output_subspace)))
    ad = H.commutator(Bivector) * 2.0          # the library commutator is half the bracket
    assert ad.kernel.shape == (10, 10)
    np.testing.assert_allclose(expm(ad.kernel), (H.exp() >> Bivector).kernel, atol=1e-10)


def test_a_step_of_each_integrator_and_the_figures_draw():
    render.draw_trajectories(*scenarios.spinning_racket(3, scenarios.STEP, scenarios.STEP, scenarios.SEED))
    curves = list(scenarios.integrator_comparison((3,), scenarios.STEP, scenarios.STEP, scenarios.SEED))
    render.draw_integrator_comparison(curves, scenarios.STEP)
