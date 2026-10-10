"""Spinor rotations, chiral currents and Majorana reality."""

import numpy as np

from examples.relativity.spinors import core

ROTATION_FRAMES = 96
ROTATION_ANGLES = np.linspace(0, 4 * np.pi, ROTATION_FRAMES, endpoint=False)
DURATION_MS = 80
RAPIDITY = 0.7
PHASE_SAMPLES = 129
PHASE_ANGLES = np.linspace(0, 2 * np.pi, PHASE_SAMPLES)


# --- math -----------------------------------------------------------------------------
def rotation() -> tuple[core.Vector, core.Scalar, core.Scalar]:
    reference = core.mv.scalar()
    # Compose two equal turns to cover the full four-pi spinor cycle.
    turn = (core.mv.xz * (ROTATION_ANGLES / 4)).exp().squared()
    psi = turn * reference
    return core.spin(psi), core.interference(psi, reference), core.interference(psi, -reference)


def chirality() -> tuple[core.Vector, core.Vector, core.Vector]:
    psi = (core.mv.tz * (-RAPIDITY / 2)).exp()
    right, left = core.PLUS(psi), core.MINUS(psi)
    return core.current(left), core.current(right), core.current(psi)


def majorana() -> tuple[core.Scalar, core.Scalar]:
    psi = core.MAJORANA(core.mv.scalar())
    psi = psi / core.density(psi).square_root()
    phase = (core.mv.yx * (PHASE_ANGLES / 2)).exp().squared()
    phased = psi * phase
    conjugated = core.CHARGE_CONJUGATION(phased)
    return core.density((phased + conjugated) / 2), core.density((phased - conjugated) / 2)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.relativity.spinors import render

    save_animation(render.animate_rotation(*rotation()), "spinor_rotation", DURATION_MS)
    save_figure(render.draw_chirality(*chirality()), "spinor_chirality")
    save_figure(render.draw_majorana(PHASE_ANGLES, *majorana()), "spinor_majorana")


if __name__ == "__main__":
    main()
