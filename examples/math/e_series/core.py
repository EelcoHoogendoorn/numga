"""The exceptional root systems E6, E7 and E8 from nested sets of mirrors, in the geometric algebra of eight dimensions.

The first six, seven or eight of E8's simple roots generate E6, E7 or E8. Each system's reflections, composed as maps on planes, form its Coxeter motion, whose orbits through one seed per generator hold all its roots; averaged with its adjoint, the motion gives its invariant rotation planes by one eigensolve.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import reduce

from numga import Algebra, NumpyContext, stack

ga = Algebra("a+b+c+d+e+f+g+h+")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Plane = ga.gatype.vector()                                                    # Plane+
PlaneMap = ga.gatype((Plane, Plane))                                          # Plane+ <- Plane+


# --- plumbing -------------------------------------------------------------------------
def simple_roots() -> Plane:
    """Eight generating mirrors (`[8] Plane+`) of squared length two generate the E8 root system."""
    return stack([
        (mv.a + mv.h - mv.b - mv.c - mv.d - mv.e - mv.f - mv.g) / 2,
        mv.a + mv.b,
        mv.b - mv.a,
        mv.c - mv.b,
        mv.d - mv.c,
        mv.e - mv.d,
        mv.f - mv.e,
        mv.g - mv.f,
    ])                                                                        # [8] Plane+


# --- math -----------------------------------------------------------------------------
def orbit_starts(simple: Plane) -> Iterator[Plane]:
    """Yields one starting root (`[] Plane+`) per generator, reflected in the mirrors preceding it."""
    preceding = Plane                                                         # [] Plane+ <- Plane+
    for root in simple:
        yield preceding(root)                                                 # [] Plane+
        preceding = preceding(root.normalized() >> Plane)                     # [] Plane+ <- Plane+


def orbit(action: PlaneMap, seeds: Plane, n_steps: int) -> Iterator[Plane]:
    """Yields the successive images (`[...] Plane+`) of `seeds` under `n_steps` applications of `action`."""
    for _ in range(n_steps):
        yield seeds                                                           # [...] Plane+
        seeds = action(seeds)                                                 # [...] Plane+


def root_system(simple: Plane, n_steps: int) -> tuple[PlaneMap, Plane]:
    """The generating reflections of `simple` compose into the Coxeter motion (`[] Plane+ <- Plane+`) and carry one seed per generator through all roots (`[n_steps, ...] Plane+`)."""
    reflections = simple.normalized() >> Plane                                # [...] Plane+ <- Plane+
    # Right composition matches the prefix order used to choose the starting roots.
    action = reduce(lambda left, right: left(right), reflections)             # [] Plane+ <- Plane+
    seeds = stack(orbit_starts(simple))                                       # [...] Plane+
    roots = stack(orbit(action, seeds, n_steps))                              # [n_steps, ...] Plane+
    return action, roots


def rotation_planes(action: PlaneMap, n_rank: int) -> tuple[Plane, Plane, Scalar]:
    """Symmetric averaging of `action` recovers its orthogonal rotation planes (`[...] Plane+`, `[...] Plane+`) and turn angles (`[...] Scalar+`) from slowest to fastest."""
    symmetric = (action + action.adjoint()) / 2                               # [] Plane+ <- Plane+
    cosines, eigenvectors = symmetric.eigh()                                  # [8] Scalar+, Plane+

    # The eigenvectors outside the root span have cosine one and sort last.
    # Paired cosines describe turns; a lone minus one describes a reflection.
    plane_cosines = cosines[n_rank - 1:0:-2]                                  # [...] Scalar+
    plane_x = eigenvectors[n_rank - 1:0:-2]                                   # [...] Plane+
    plane_y = (action(plane_x) - plane_cosines * plane_x).normalized()        # [...] Plane+
    return plane_x, plane_y, plane_cosines.arccos()
