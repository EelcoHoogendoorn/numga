"""Spinor rotations, chiral currents and Majorana reality."""

import numpy as np

from examples.relativity.spinors import core

n_rotation_frames = 192
rotation_angles = np.linspace(0, 4 * np.pi, n_rotation_frames, endpoint=False)
frame_duration = 40
rapidity = 0.7
n_phase_samples = 129
phase_angles = np.linspace(0, 2 * np.pi, n_phase_samples)


# --- math -----------------------------------------------------------------------------
def rotation() -> tuple[core.Antivector, core.Scalar, core.Scalar]:
    """A spinor turned through two full turns about y: its spin, and the intensities at the two outputs
    where it is recombined with the unturned spinor, their sum and their difference."""
    reference = core.mv.scalar()
    psi = (core.mv.xz * (rotation_angles / 2)).exp()
    return core.spin(psi), core.interference(psi, reference), core.interference(psi, -reference)


def chirality() -> tuple[core.Antivector, core.Antivector, core.Antivector]:
    """A boosted spinor split into its left- and right-handed parts: their currents, and the spinor's."""
    psi = (core.mv.tz * (-rapidity / 2)).exp()
    right, left = core.right_projector(psi), core.left_projector(psi)
    return core.current(left), core.current(right), core.current(psi)


def majorana() -> tuple[core.Scalar, core.Scalar]:
    """A Majorana spinor turned by a phase: the densities of the parts charge conjugation keeps and
    negates, over the phase angles."""
    psi = core.majorana_projector(core.mv.scalar())
    psi = psi / core.density(psi).square_root()
    phase = (core.mv.yx * phase_angles).exp()
    phased = psi * phase
    conjugated = core.charge_conjugation(phased)
    return core.density((phased + conjugated) / 2), core.density((phased - conjugated) / 2)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.relativity.spinors import render

    spin, bright, dark = rotation()
    left_current, right_current, current = chirality()
    fixed_density, negated_density = majorana()
    save_animation(render.animate_rotation(spin, bright, dark), "spinor_rotation", frame_duration)
    save_figure(render.draw_chirality(left_current, right_current, current), "spinor_chirality")
    save_figure(render.draw_majorana(phase_angles, fixed_density, negated_density), "spinor_majorana")

    # checks
    # One full turn restores the spin but darkens the sum port: the spinor has changed sign.
    half = n_rotation_frames // 2
    np.testing.assert_allclose((spin[half] - spin[0]).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(bright[half].to_array(), 0.0, atol=1e-12)
    np.testing.assert_allclose(dark[half].to_array(), 1.0, atol=1e-12)
    # Each Weyl current is null, and the two add to the Dirac current.
    np.testing.assert_allclose([left_current.squared().to_array(), right_current.squared().to_array()], 0.0, atol=1e-12)
    np.testing.assert_allclose((left_current + right_current - current).kernel, 0.0, atol=1e-12)
    # The phase moves the density between the conjugation eigenspaces, never losing any.
    np.testing.assert_allclose((fixed_density + negated_density).to_array(), 1.0, atol=1e-12)


if __name__ == "__main__":
    main()
