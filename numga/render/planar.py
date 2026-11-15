"""`planar` holds the shared projective plane `PGA2D`, its `NumpyContext`, its antivector `Point` type, and its unit
origin `mv.w.dual()` for planar rendering setups."""

from __future__ import annotations

from numga import NumpyContext
from numga.algebras import PGA2D as ga

context = NumpyContext(ga)
mv = context.multivector
Point = ga.gatype.antivector()                                               # Antivector-
origin = mv.w.dual()                                                         # [] Point-
