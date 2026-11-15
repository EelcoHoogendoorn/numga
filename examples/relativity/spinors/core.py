"""Dirac, Weyl, and Majorana states occupy one eight-dimensional real spinor space in the spacetime algebra."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import STA as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                     # Scalar+
# Spacetime currents and spin axes are intrinsic directions (antivectors).
Antivector = ga.gatype.antivector()                             # Antivector-
Spinor = ga.gatype.even()                                       # Spinor+
# Negating the dual of the observer's spatial slice `mv.t` compensates for `t` squaring to -1 so `mv.t & time == 1`.
time = -mv.t.dual()                                             # [] Antivector-

# Electromagnetic phase, chirality, and charge conjugation act on `Spinor` by right multiplication.
phase_unit = Spinor * mv.yx                                     # [] Spinor+ <- Spinor+
chirality_map = Spinor * mv.zt                                  # [] Spinor+ <- Spinor+
charge_conjugation = Spinor * mv.yt                             # [] Spinor+ <- Spinor+
right_projector = (Spinor + chirality_map) / 2                  # [] Spinor+ <- Spinor+
left_projector = (Spinor - chirality_map) / 2                   # [] Spinor+ <- Spinor+
majorana_projector = (Spinor + charge_conjugation) / 2          # [] Spinor+ <- Spinor+


# --- math -----------------------------------------------------------------------------
def current(psi: Spinor) -> Antivector:
    """The two-sided sandwich of `psi` carries the observer's time direction into the future-directed Dirac current (`[...] Antivector-`)."""
    return psi >> time


def density(psi: Spinor) -> Scalar:
    """Pairing the observer's spatial slice with the Dirac current gives the measured probability density (`[...] Scalar+`)."""
    return mv.t & current(psi)


def spin(psi: Spinor) -> Antivector:
    """The two-sided sandwich of `psi` carries the reference z direction into the spatial spin current (`[...] Antivector-`)."""
    return psi >> mv.z.dual()


def rotation_states(angles: np.ndarray) -> Iterator[tuple[Antivector, Scalar, Scalar]]:
    """Yields the spatial spin current and the sum-port and difference-port densities at each rotation angle."""
    reference = mv.scalar()                                     # [] Spinor+
    for angle in angles:
        psi = (mv.xz * (angle / 2)).exp()                       # [] Spinor+
        sum_density = density((psi + reference) / 2)            # [] Scalar+
        difference_density = density((psi - reference) / 2)     # [] Scalar+
        yield spin(psi), sum_density, difference_density


def chiral_currents(rapidity: float) -> tuple[Antivector, Antivector, Antivector]:
    """Projecting a z-boosted Dirac spinor onto left and right Weyl subspaces splits its current into two null currents (`[] Antivector-`)."""
    psi = (mv.tz * (-rapidity / 2)).exp()                       # [] Spinor+
    left_spinor = left_projector(psi)                           # [] Spinor+
    right_spinor = right_projector(psi)                         # [] Spinor+
    return current(left_spinor), current(right_spinor), current(psi)


def majorana_densities(phase_angles: np.ndarray) -> tuple[Scalar, Scalar]:
    """Advancing a unit-density Majorana spinor through `phase_angles` shifts observer density between the two conjugation eigenspaces (`[...] Scalar+`)."""
    seed_spinor = majorana_projector(mv.scalar())               # [] Spinor+
    majorana_spinor = seed_spinor / density(seed_spinor).square_root()  # [] Spinor+
    phase_rotors = (mv.yx * phase_angles).exp()                 # [...] Spinor+
    phased_spinors = majorana_spinor * phase_rotors             # [...] Spinor+
    conjugate_spinors = charge_conjugation(phased_spinors)      # [...] Spinor+
    fixed_parts = (phased_spinors + conjugate_spinors) / 2      # [...] Spinor+
    negated_parts = (phased_spinors - conjugate_spinors) / 2    # [...] Spinor+
    return density(fixed_parts), density(negated_parts)
