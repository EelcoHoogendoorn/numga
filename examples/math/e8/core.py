"""The root system of E8: eight orbits of thirty roots under the Coxeter element, the product of eight
reflections, in the geometric algebra of eight dimensions.

The eight simple roots are mirrors, vectors, and their product is a rotor whose action on vectors is the
Coxeter element as one map. One root from each orbit, each acted on by the reflections before it in the
product, seeds eight orbits of thirty that hold all 240 roots.
"""

from __future__ import annotations

from collections.abc import Iterator

from numga import Algebra, NumpyContext
from numga.extensor import stack

ga = Algebra("a+b+c+d+e+f+g+h+")
context = NumpyContext(ga)
mv = context.multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Vector = ga.gatype.vector()                                                   # Vector+
Bivector = ga.gatype.bivector()                                               # Bivector+
Rotor = ga.gatype.rotor()                                                     # Rotor+
VectorMap = ga.gatype((Vector, Vector))
coxeter_order = 30


# --- plumbing -------------------------------------------------------------------------
def simple_roots() -> Vector:
    """Eight mirrors of squared length two whose reflections generate E8's root system."""
    return stack([
        (mv.a + mv.h - mv.b - mv.c - mv.d - mv.e - mv.f - mv.g) / 2,
        mv.a + mv.b,
        mv.b - mv.a,
        mv.c - mv.b,
        mv.d - mv.c,
        mv.e - mv.d,
        mv.f - mv.e,
        mv.g - mv.f,
    ])


# --- math -----------------------------------------------------------------------------
def reflection_seeds(roots: Vector) -> Iterator[Vector]:
    """One root from each Coxeter orbit, reflected in the mirrors preceding it."""
    prefix = Vector                                                           # [] Vector+ <- Vector+
    for root in roots:
        yield prefix(root)                                                    # [] Vector+
        prefix = prefix(root.normalized() >> Vector)                          # [] Vector+ <- Vector+


def orbit(action: VectorMap, seeds: Vector, n_steps: int) -> Iterator[Vector]:
    """Successive images of a batch of seeds under one motion."""
    for _ in range(n_steps):
        yield seeds                                                           # [orbits] Vector+
        seeds = action(seeds)                                                 # [orbits] Vector+
