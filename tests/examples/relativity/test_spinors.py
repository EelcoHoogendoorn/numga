"""Spinor projections and observable currents."""

import numpy as np

from examples.relativity.spinors import core


TOLERANCE = 1e-11
SAMPLES = 24


def test_spinor_observables_transform_with_lorentz_rotors_and_ignore_phase():
    rng = np.random.default_rng(931)
    mv = core.mv
    spinors = mv(core.Spinor, rng.normal(size=(SAMPLES, 8)) * 0.3)
    rotors = (
        (mv.tx * rng.uniform(-0.4, 0.4, SAMPLES)).exp()
        * (mv.yz * rng.uniform(-0.6, 0.6, SAMPLES)).exp()
        * (mv.xy * rng.uniform(-0.6, 0.6, SAMPLES)).exp()
    )
    phases = (mv.yx * rng.uniform(-1.0, 1.0, SAMPLES)).exp()
    moved = rotors * spinors
    phased = spinors * phases

    # The spinor transforms on one side; its vector observables follow the
    # spacetime rotation and boost. Sign and right phase leave them unchanged.
    for observable in (core.current, core.spin):
        np.testing.assert_allclose(
            (observable(moved) - (rotors >> observable(spinors))).kernel,
            0, atol=TOLERANCE,
        )
        np.testing.assert_allclose(
            (observable(phased) - observable(spinors)).kernel,
            0, atol=TOLERANCE,
        )
        np.testing.assert_allclose(
            (observable(-spinors) - observable(spinors)).kernel,
            0, atol=TOLERANCE,
        )


def test_weyl_projections_split_the_current_into_two_covariant_null_currents():
    rng = np.random.default_rng(932)
    mv = core.mv
    spinors = mv(core.Spinor, rng.normal(size=(SAMPLES, 8)) * 0.3)
    plus, minus = core.PLUS(spinors), core.MINUS(spinors)
    rotor = (mv.tx * 0.27).exp() * (mv.yz * -0.38).exp()
    plus_current, minus_current = core.current(plus), core.current(minus)

    # Complementary projections commute with every left Lorentz action.
    np.testing.assert_allclose((plus + minus - spinors).kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose((core.PLUS(plus) - plus).kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose((core.MINUS(minus) - minus).kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose(core.PLUS(minus).kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose(core.MINUS(plus).kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose(
        (core.PLUS(rotor * spinors) - rotor * plus).kernel, 0, atol=TOLERANCE,
    )
    np.testing.assert_allclose(
        (core.MINUS(rotor * spinors) - rotor * minus).kernel, 0, atol=TOLERANCE,
    )

    # Each current is future null, while their sum is the Dirac current and
    # their difference is the spin current.
    np.testing.assert_allclose(plus_current.squared().kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose(minus_current.squared().kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose(
        (plus_current + minus_current - core.current(spinors)).kernel,
        0, atol=TOLERANCE,
    )
    np.testing.assert_allclose(
        (plus_current - minus_current - core.spin(spinors)).kernel,
        0, atol=TOLERANCE,
    )
    assert np.all((plus_current | mv.t).kernel > 0)
    assert np.all((minus_current | mv.t).kernel > 0)
    assert np.all(core.current(spinors).squared().kernel > 0)


def test_charge_conjugation_exchanges_chirality_and_reverses_the_phase():
    rng = np.random.default_rng(933)
    mv = core.mv
    spinors = mv(core.Spinor, rng.normal(size=(SAMPLES, 8)) * 0.3)
    conjugate = core.CHARGE_CONJUGATION
    phase = (mv.yx * 0.43).exp()
    rotor = (mv.ty * -0.32).exp() * (mv.xz * 0.24).exp()
    majorana = core.MAJORANA(spinors)

    # Charge conjugation is an involution, commutes with Lorentz action, and
    # reverses the chosen phase rather than being linear over that phase.
    np.testing.assert_allclose((conjugate(conjugate(spinors)) - spinors).kernel,
                               0, atol=TOLERANCE)
    np.testing.assert_allclose(
        (conjugate(rotor * spinors) - rotor * conjugate(spinors)).kernel,
        0, atol=TOLERANCE,
    )
    np.testing.assert_allclose(
        (conjugate(core.PHASE(spinors)) + core.PHASE(conjugate(spinors))).kernel,
        0, atol=TOLERANCE,
    )
    np.testing.assert_allclose(
        (conjugate(spinors * phase) - conjugate(spinors) * phase.reverse()).kernel,
        0, atol=TOLERANCE,
    )
    np.testing.assert_allclose(
        (conjugate(core.PLUS(spinors)) - core.MINUS(conjugate(spinors))).kernel,
        0, atol=TOLERANCE,
    )

    # These commuting Majorana states are fixed by conjugation and carry a
    # null current. Real coefficients alone do not impose this constraint;
    # a generic phase also takes a fixed state out of the fixed subspace.
    np.testing.assert_allclose((conjugate(majorana) - majorana).kernel,
                               0, atol=TOLERANCE)
    np.testing.assert_allclose(core.current(majorana).squared().kernel,
                               0, atol=TOLERANCE)
    np.testing.assert_allclose(core.spin(majorana).kernel, 0, atol=TOLERANCE)
    assert np.max(np.abs((conjugate(spinors) - spinors).kernel)) > 0.1
    assert np.max(np.abs((conjugate(majorana * phase) - majorana * phase).kernel)) > 0.1
