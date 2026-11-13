"""The projective plane PGA2D as the pictures of planar scenes share it: its context, its points and its origin."""

from __future__ import annotations

from numga import NumpyContext
from numga.algebras import PGA2D as ga

context = NumpyContext(ga)
mv = context.multivector
Point = ga.gatype.antivector()                                               # Antivector-
origin = mv.w.dual()                                                         # [] Point-
