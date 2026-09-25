"""Unit tests for the tennis racket integrator stress test and its Lie-group steppers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

from numga import NumpyContext
from numga.algebra import Algebra

from examples import instantiate
from examples.mechanics.tennis_racket import render, scenarios


def racket(p: int):
    return instantiate("examples.mechanics.tennis_racket.core", Algebra.from_pqr(p, 0, 0))


def momentum_drift(p: int, integrator: str, dt: float) -> float:
    """Worst relative drift of the world-frame angular momentum over all bodies and steps."""
    core = racket(p)
    body = core.racket(42)
    motors, rates = core.simulate(body, getattr(core.lie, integrator), dt, int(20.0 / dt))
    return float(core.momentum_drift(motors, rates, body.inertia).to_array().max())


def test_adjoint_is_the_open_commutator_in_five_dimensions():
    """The sandwich by h.exp() on bivectors equals the exponential of the commutator with h, on the ten-dimensional
    bivectors of five dimensions."""
    context = NumpyContext(Algebra.from_pqr(5, 0, 0), dtype=np.float64)
    mv = context.multivector
    bivector = context.algebra.subspace.bivector()
    rng = np.random.default_rng(0)
    H = mv.bivector(rng.normal(scale=0.3, size=len(bivector)))
    ad = H.commutator(bivector) * 2.0          # the library commutator is half the bracket
    assert ad.kernel.shape == (10, 10)
    np.testing.assert_allclose(expm(ad.kernel), (H.exp() >> bivector).kernel, atol=1e-10)


def test_energy_is_blind_to_the_motor_step():
    """All steppers run the same RK4 on the autonomous body-frame rate, so energies agree."""
    core = racket(4)
    body = core.racket(42)
    histories = []
    for step in (core.lie.explicit_verlet, core.lie.explicit_rk4, core.lie.explicit_rkmk4):
        _, rates = core.simulate(body, step, 0.25, 40)
        histories.append(core.lie.kinetic_energy(rates, body.inertia).to_array())
    for h in histories[1:]:
        np.testing.assert_allclose(h, histories[0], rtol=1e-11)


def test_rkmk4_conserves_world_momentum_to_fourth_order():
    """Halving dt cuts RKMK4's momentum drift about sixteenfold; Verlet and RK4 only halve it."""
    for p in (3, 4):
        coarse = momentum_drift(p, "explicit_rkmk4", 0.25)
        fine = momentum_drift(p, "explicit_rkmk4", 0.125)
        assert coarse < 0.0001
        assert 10.0 < coarse / fine < 24.0
        for first_order in ("explicit_verlet", "explicit_rk4"):
            ratio = momentum_drift(p, first_order, 0.25) / momentum_drift(p, first_order, 0.125)
            assert 1.7 < ratio < 2.4


def test_rkmk4_in_five_dimensions():
    """The same stepper integrates Spin(5) rotors with fourth-order momentum conservation."""
    coarse = momentum_drift(5, "explicit_rkmk4", 0.25)
    fine = momentum_drift(5, "explicit_rkmk4", 0.125)
    assert coarse < 0.01
    assert 10.0 < coarse / fine < 24.0
    assert momentum_drift(5, "explicit_verlet", 0.25) > 1e-2


def test_intermediate_axis_tumbles_and_others_do_not():
    """In 3D the spin about the medial axis flips sign; the major and minor axes stay put."""
    core = racket(3)
    body = core.racket(42)
    _, rates = core.simulate(body, core.lie.explicit_rkmk4, 0.25, 800)
    trajectory = rates.kernel
    flips = [np.any(trajectory[:, i, i] * trajectory[0, i, i] < 0) for i in range(3)]
    assert sum(flips) == 1


def test_figures_draw():
    figures = [
        render.draw_trajectories(*scenarios.spinning_racket(4, 0.25, 10.0, 42)),
        render.draw_integrator_comparison(scenarios.integrator_comparison((3,), 0.25, 10.0, 42), 0.25),
    ]
    assert all(isinstance(figure, plt.Figure) for figure in figures)
