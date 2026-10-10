"""Dirac, Weyl and Majorana states in one real spacetime spinor space."""

from __future__ import annotations

from numga import NumpyContext
from numga.algebras import STA as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Spinor = ga.gatype.even()

# Each operation leaves the spinor input open, so these are maps on the same carrier.
PHASE = Spinor * mv.yx
CHIRALITY = Spinor * mv.zt
CHARGE_CONJUGATION = Spinor * mv.yt
PLUS = (Spinor + CHIRALITY) / 2
MINUS = (Spinor - CHIRALITY) / 2
MAJORANA = (Spinor + CHARGE_CONJUGATION) / 2


# --- math -----------------------------------------------------------------------------
def current(psi: Spinor) -> Vector:
    """The future current obtained by carrying the observer's time axis with the spinor."""
    return psi >> mv.t


def spin(psi: Spinor) -> Vector:
    """The spin current obtained by carrying the spin axis with the spinor."""
    return psi >> mv.z


def density(psi: Spinor) -> Scalar:
    """The probability density measured by the fixed time observer."""
    return current(psi) | mv.t


def interference(psi: Spinor, reference: Spinor) -> Scalar:
    """Intensity at one output of two equal, coherently recombined paths."""
    return density((psi + reference) / 2)
