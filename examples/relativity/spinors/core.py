"""Dirac, Weyl and Majorana states in one real spacetime spinor space."""

from __future__ import annotations

from numga import NumpyContext
from numga.algebras import STA as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# Currents are directions in spacetime, antivectors.
Antivector = ga.gatype.antivector()
Spinor = ga.gatype.even()
# The observer's direction in time, which pairs to one with its slice `mv.t`; the dual carries the minus of
# t's square.
time = -mv.t.dual()

# Phase, chirality and charge conjugation are right multiplications, maps from Spinor to Spinor.
phase_unit = Spinor * mv.yx
chirality_map = Spinor * mv.zt
charge_conjugation = Spinor * mv.yt
right_projector = (Spinor + chirality_map) / 2
left_projector = (Spinor - chirality_map) / 2
majorana_projector = (Spinor + charge_conjugation) / 2


# --- math -----------------------------------------------------------------------------
def current(psi: Spinor) -> Antivector:
    """The future current obtained by carrying the observer's direction in time with the spinor."""
    return psi >> time


def spin(psi: Spinor) -> Antivector:
    """The spin current obtained by carrying the z direction with the spinor."""
    return psi >> mv.z.dual()


def density(psi: Spinor) -> Scalar:
    """The probability density the fixed observer measures: its slice of space reading the current."""
    return mv.t & current(psi)


def interference(psi: Spinor, reference: Spinor) -> Scalar:
    """Intensity at one output of two equal, coherently recombined paths."""
    return density((psi + reference) / 2)
