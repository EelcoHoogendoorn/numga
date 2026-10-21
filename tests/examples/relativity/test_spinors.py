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

    # checks: the spinor transforms on one side; its vector observables follow the
    # spacetime rotation and boost. A right phase leaves them unchanged.
    for observable in (core.current, core.spin):
        np.testing.assert_allclose(
            (observable(moved) - (rotors >> observable(spinors))).kernel,
            0, atol=TOLERANCE,
        )
        np.testing.assert_allclose(
            (observable(phased) - observable(spinors)).kernel,
            0, atol=TOLERANCE,
        )


def test_weyl_projections_split_the_current_into_two_covariant_null_currents():
    rng = np.random.default_rng(932)
    mv = core.mv
    spinors = mv(core.Spinor, rng.normal(size=(SAMPLES, 8)) * 0.3)
    right, left = core.right_projector(spinors), core.left_projector(spinors)
    rotor = (mv.tx * 0.27).exp() * (mv.yz * -0.38).exp()
    right_current, left_current = core.current(right), core.current(left)

    # checks: complementary projections commute with every left Lorentz action.
    np.testing.assert_allclose((right + left - spinors).kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose((core.right_projector(right) - right).kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose((core.left_projector(left) - left).kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose(core.right_projector(left).kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose(core.left_projector(right).kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose(
        (core.right_projector(rotor * spinors) - rotor * right).kernel, 0, atol=TOLERANCE,
    )
    np.testing.assert_allclose(
        (core.left_projector(rotor * spinors) - rotor * left).kernel, 0, atol=TOLERANCE,
    )
    # Each current is future null, while their sum is the Dirac current and
    # their difference is the spin current.
    np.testing.assert_allclose(right_current.squared().kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose(left_current.squared().kernel, 0, atol=TOLERANCE)
    np.testing.assert_allclose(
        (right_current + left_current - core.current(spinors)).kernel,
        0, atol=TOLERANCE,
    )
    np.testing.assert_allclose(
        (right_current - left_current - core.spin(spinors)).kernel,
        0, atol=TOLERANCE,
    )
    assert np.all(core.density(right).kernel > 0)
    assert np.all(core.density(left).kernel > 0)
    assert np.all(core.current(spinors).squared().kernel < 0)


def test_charge_conjugation_exchanges_chirality_and_reverses_the_phase():
    rng = np.random.default_rng(933)
    mv = core.mv
    spinors = mv(core.Spinor, rng.normal(size=(SAMPLES, 8)) * 0.3)
    conjugate = core.charge_conjugation
    phase = (mv.yx * 0.43).exp()
    rotor = (mv.ty * -0.32).exp() * (mv.xz * 0.24).exp()
    majorana = core.majorana_projector(spinors)

    # checks: charge conjugation is an involution, commutes with Lorentz action, and
    # reverses the chosen phase rather than being linear over that phase.
    np.testing.assert_allclose((conjugate(conjugate(spinors)) - spinors).kernel,
                               0, atol=TOLERANCE)
    np.testing.assert_allclose(
        (conjugate(rotor * spinors) - rotor * conjugate(spinors)).kernel,
        0, atol=TOLERANCE,
    )
    np.testing.assert_allclose(
        (conjugate(core.phase_unit(spinors)) + core.phase_unit(conjugate(spinors))).kernel,
        0, atol=TOLERANCE,
    )
    np.testing.assert_allclose(
        (conjugate(spinors * phase) - conjugate(spinors) * phase.reverse()).kernel,
        0, atol=TOLERANCE,
    )
    np.testing.assert_allclose(
        (conjugate(core.right_projector(spinors)) - core.left_projector(conjugate(spinors))).kernel,
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
