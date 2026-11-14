"""Tests cover Lorentz covariance of spinor currents, chiral Weyl projections, charge conjugation, and spinor figure rendering."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.relativity.spinors import core, render

tolerance = 1e-9
n_samples = 8
spacetime_axes = stack((core.mv.t, core.mv.x, core.mv.y, core.mv.z))            # [4] Vector+


def spinor_error(difference: core.Spinor) -> core.Scalar:
    """Observer density measures the positive-definite squared norm of `difference` across all eight real spinor components."""
    return core.density(difference).max()


def current_error(difference: core.Antivector) -> core.Scalar:
    """Pairing `difference` against the four spacetime coordinate planes measures its largest component magnitude."""
    return (spacetime_axes & difference[..., None]).abs().max()


def test_spinor_observables_transform_with_lorentz_rotors_and_ignore_phase() -> None:
    """Left Lorentz rotors carry the Dirac and spin currents by sandwiching while right phase turns leave both unchanged."""
    rng = np.random.default_rng(931)
    mv = core.mv
    spinors = mv(core.Spinor, rng.normal(size=(n_samples, 8)) * 0.3)            # [n_samples] Spinor+
    rotors = (
        (mv.tx * rng.uniform(-0.4, 0.4, n_samples)).exp()
        * (mv.yz * rng.uniform(-0.6, 0.6, n_samples)).exp()
        * (mv.xy * rng.uniform(-0.6, 0.6, n_samples)).exp()
    )                                                                           # [n_samples] Spinor+
    phase_rotors = (mv.yx * rng.uniform(-1.0, 1.0, n_samples)).exp()            # [n_samples] Spinor+
    lorentz_spinors = rotors * spinors                                          # [n_samples] Spinor+
    phased_spinors = spinors * phase_rotors                                     # [n_samples] Spinor+

    for observable in (core.current, core.spin):
        expected_currents = rotors >> observable(spinors)                       # [n_samples] Antivector-
        assert current_error(observable(lorentz_spinors) - expected_currents) < tolerance
        assert current_error(observable(phased_spinors) - observable(spinors)) < tolerance


def test_weyl_projections_split_the_current_into_two_covariant_null_currents() -> None:
    """Complementary Weyl projectors commute with left Lorentz rotors and split the Dirac current into two future-null currents."""
    rng = np.random.default_rng(932)
    mv = core.mv
    spinors = mv(core.Spinor, rng.normal(size=(n_samples, 8)) * 0.3)            # [n_samples] Spinor+
    right_spinors = core.right_projector(spinors)                               # [n_samples] Spinor+
    left_spinors = core.left_projector(spinors)                                 # [n_samples] Spinor+
    rotor = (mv.tx * 0.27).exp() * (mv.yz * -0.38).exp()                        # [] Spinor+
    right_currents = core.current(right_spinors)                                # [n_samples] Antivector-
    left_currents = core.current(left_spinors)                                  # [n_samples] Antivector-

    assert spinor_error(right_spinors + left_spinors - spinors) < tolerance
    assert spinor_error(core.right_projector(right_spinors) - right_spinors) < tolerance
    assert spinor_error(core.left_projector(left_spinors) - left_spinors) < tolerance
    assert spinor_error(core.right_projector(left_spinors)) < tolerance
    assert spinor_error(core.left_projector(right_spinors)) < tolerance
    assert spinor_error(core.right_projector(rotor * spinors) - rotor * right_spinors) < tolerance
    assert spinor_error(core.left_projector(rotor * spinors) - rotor * left_spinors) < tolerance

    assert right_currents.squared().abs().max() < tolerance
    assert left_currents.squared().abs().max() < tolerance
    assert current_error(right_currents + left_currents - core.current(spinors)) < tolerance
    assert current_error(right_currents - left_currents - core.spin(spinors)) < tolerance
    assert (core.density(right_spinors) > 0).all()
    assert (core.density(left_spinors) > 0).all()
    assert (core.current(spinors).squared() < 0).all()


def test_charge_conjugation_exchanges_chirality_and_reverses_the_phase() -> None:
    """Charge conjugation commutes with Lorentz rotors, exchanges Weyl subspaces, and reverses right phase turns."""
    rng = np.random.default_rng(933)
    mv = core.mv
    spinors = mv(core.Spinor, rng.normal(size=(n_samples, 8)) * 0.3)            # [n_samples] Spinor+
    conjugate = core.charge_conjugation
    phase_rotor = (mv.yx * 0.43).exp()                                          # [] Spinor+
    rotor = (mv.ty * -0.32).exp() * (mv.xz * 0.24).exp()                        # [] Spinor+
    majorana_spinors = core.majorana_projector(spinors)                         # [n_samples] Spinor+

    assert spinor_error(conjugate(conjugate(spinors)) - spinors) < tolerance
    assert spinor_error(conjugate(rotor * spinors) - rotor * conjugate(spinors)) < tolerance
    assert spinor_error(conjugate(core.phase_unit(spinors)) + core.phase_unit(conjugate(spinors))) < tolerance
    assert spinor_error(conjugate(spinors * phase_rotor) - conjugate(spinors) * phase_rotor.reverse()) < tolerance
    assert spinor_error(conjugate(core.right_projector(spinors)) - core.left_projector(conjugate(spinors))) < tolerance

    assert spinor_error(conjugate(majorana_spinors) - majorana_spinors) < tolerance
    assert core.current(majorana_spinors).squared().abs().max() < tolerance
    assert current_error(core.spin(majorana_spinors)) < tolerance
    assert spinor_error(conjugate(spinors) - spinors) > 1e-2
    assert spinor_error(conjugate(majorana_spinors * phase_rotor) - majorana_spinors * phase_rotor) > 1e-2


def test_render_builds_rotation_frame_and_static_figures() -> None:
    """The renderer produces an RGB frame for the rotation stream and matplotlib figures for chirality and Majorana sweeps."""
    frames = list(render.rotation_animation(core.rotation_states(np.array([0.0]))))
    assert len(frames) == 1
    assert frames[0].ndim == 3 and frames[0].shape[2] == 3

    left_current, right_current, dirac_current = core.chiral_currents(0.5)
    chirality_fig = render.chirality_figure(left_current, right_current, dirac_current)
    assert len(chirality_fig.axes) == 1
    plt.close(chirality_fig)

    angles = np.linspace(0.0, 2 * np.pi, 5)
    fixed_density, negated_density = core.majorana_densities(angles)
    majorana_fig = render.majorana_figure(angles, fixed_density, negated_density)
    assert len(majorana_fig.axes) == 1
    plt.close(majorana_fig)
