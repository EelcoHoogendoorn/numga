"""Unit tests for the tennis racket integrator stress test and its Lie-group steppers."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from numga import NumpyContext
from numga.algebra import Algebra

from examples.mechanics.tennis_racket import INTEGRATORS, setup_tennis_racket_bodies, simulate_tennis_racket


def momentum_drift(p: int, integrator: str, dt: float, runtime: float = 20.0) -> float:
    """Worst relative drift of the world-frame angular momentum over all bodies and steps."""
    context = NumpyContext(Algebra.from_pqr(p, 0, 0), dtype=np.float64)
    _, _, _, momenta = simulate_tennis_racket(context, dt=dt, runtime=runtime, integrator=integrator)
    return float(np.max(np.linalg.norm(momenta - momenta[0], axis=-1) / np.linalg.norm(momenta[0], axis=-1)))


def test_adjoint_is_the_open_commutator_in_five_dimensions():
    """Ad(exp H) = exp(ad_H) on the ten-dimensional bivector algebra of Spin(5)."""
    context = NumpyContext(Algebra.from_pqr(5, 0, 0), dtype=np.float64)
    mv = context.multivector
    bivector = context.algebra.subspace.bivector()
    rng = np.random.default_rng(0)
    H = mv.bivector(rng.normal(scale=0.3, size=len(bivector)))
    ad = H.commutator(bivector) * 2.0          # the library commutator is half the bracket
    assert ad.kernel.shape == (10, 10)
    np.testing.assert_allclose(expm(ad.kernel), (H.exp() >> bivector).kernel, atol=1e-12)


def test_energy_is_blind_to_the_motor_step():
    """All steppers run the same RK4 on the autonomous body-frame rate, so energies agree."""
    context = NumpyContext(Algebra.from_pqr(4, 0, 0), dtype=np.float64)
    histories = [simulate_tennis_racket(context, dt=0.25, runtime=10.0, integrator=name)[2] for name in INTEGRATORS]
    for h in histories[1:]:
        np.testing.assert_allclose(h, histories[0], rtol=1e-12)


def test_rkmk4_conserves_world_momentum_to_fourth_order():
    """Halving dt cuts RKMK4's momentum drift about sixteenfold; Verlet and RK4 only halve it."""
    for p in (3, 4):
        coarse = momentum_drift(p, "rkmk4", 0.25)
        fine = momentum_drift(p, "rkmk4", 0.125)
        assert coarse < 1e-6
        assert 10.0 < coarse / fine < 24.0
        for first_order in ("verlet", "rk4"):
            ratio = momentum_drift(p, first_order, 0.25) / momentum_drift(p, first_order, 0.125)
            assert 1.7 < ratio < 2.4


def test_rkmk4_in_five_dimensions():
    """The same stepper integrates Spin(5) rotors with fourth-order momentum conservation."""
    coarse = momentum_drift(5, "rkmk4", 0.25)
    fine = momentum_drift(5, "rkmk4", 0.125)
    assert coarse < 1e-4
    assert 10.0 < coarse / fine < 24.0
    assert momentum_drift(5, "verlet", 0.25) > 1e-2


def test_intermediate_axis_tumbles_and_others_do_not():
    """In 3D the spin about the medial axis flips sign; the major and minor axes stay put."""
    context = NumpyContext(Algebra.from_pqr(3, 0, 0), dtype=np.float64)
    trajectory, _, _, _ = simulate_tennis_racket(context, dt=0.25, runtime=200.0, integrator="rkmk4")
    body, _ = setup_tennis_racket_bodies(context)
    flips = [np.any(trajectory[:, i, i] * trajectory[0, i, i] < 0) for i in range(3)]
    assert sum(flips) == 1


def test_comparison_figure(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from examples.mechanics.tennis_racket import run_integrator_comparison
    out = tmp_path / "integrators.png"
    run_integrator_comparison(dims=(3,), runtime=10.0, save_path=str(out))
    assert out.exists()
