"""Polarized cobalt-60 beta-emission scenarios compare the Wu apparatus, its mirror image, and the mirrored coil."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.orientation.wu_experiment import core
from examples.orientation.wu_experiment.core import Asymmetry, Direction, Plane, Scalar, Spin, mv

# Effective beta-decay asymmetry `-v/c` for cobalt-60 (`A = -1` with electron speed `v/c = 0.6`).
decay_asymmetry = -0.6
# Angular sample count around the vertical plane containing the coil axis.
n_directions = 180


# --- math -----------------------------------------------------------------------------
def wu_comparison() -> tuple[Spin, Direction, Scalar, Asymmetry, Plane]:
    """Evaluating cobalt-60 beta emission around `n_directions` meridian directions for the original
    coil, its mirror image across `x = 0`, and the physically mirrored coil yields `[3] Spin-`,
    `[n_directions] Direction-`, `[3, n_directions] Scalar+`, `[] Asymmetry-`, and `[] Plane+`."""
    # Chiral weak-decay constant measured against the basis volume orientation.
    asymmetry = mv(Asymmetry, [decay_asymmetry])                                  # [] Asymmetry-
    # Horizontal circulation patch of a coil wound counterclockwise as viewed from above.
    spin = mv.xy.dual()                                                           # [] Spin-
    # Vertical mirror plane `x = 0` containing the coil symmetry axis.
    mirror = mv.x                                                                 # [] Plane+
    angles = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
    meridian_rotors = (mv.zx * (-angles / 2)).exp()                               # [n_directions] Rotor+
    directions = meridian_rotors >> mv.z.dual()                                   # [n_directions] Direction-

    spins, rates = core.parity_cases(spin, directions, asymmetry, mirror)         # [3] Spin-, [3, n_directions] Scalar+
    return spins, directions, rates, asymmetry, mirror


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.orientation.wu_experiment import render

    spins, directions, rates, asymmetry, mirror = wu_comparison()
    save_figure(render.experiment_figure(spins, directions, rates), "wu_experiment")

    # checks
    spin = spins[0]                                                               # [] Spin-
    _, mirror_image_rates, mirrored_apparatus_rates = rates                       # [n_directions] Scalar+
    # Hypothetically reflecting the weak constant alongside the coil would reproduce the optical mirror image.
    mirrored_law_rates = core.emission_rates(mirror >> spin, directions, mirror >> asymmetry)  # [n_directions] Scalar+
    assert ((mirrored_law_rates - mirror_image_rates).abs() < 1e-12).all()
    # Keeping the physical weak constant makes the mirrored coil emit into the opposite hemisphere from the mirror image.
    assert ((mirrored_apparatus_rates + mirror_image_rates - 2).abs() < 1e-12).all()
    # Both real coils emit preferentially against their own nuclear spin (downward for the original, upward for the mirrored coil).
    vertical_directions = stack([-mv.z.dual(), mv.z.dual()])                      # [2] Direction-
    original_down, original_up = core.emission_rates(spin, vertical_directions, asymmetry)
    mirrored_down, mirrored_up = core.emission_rates(mirror >> spin, vertical_directions, asymmetry)
    assert bool(original_down > original_up) and bool(mirrored_up > mirrored_down)


if __name__ == "__main__":
    main()
