"""Spherical synchronization turns a population of directions on a sphere while attracting them toward
one another through a single evolving Möbius transformation in the sphere's conformal geometry.

Each point on the sphere is modelled as a null ray `direction + time` (`Antivector-`), and each circle
as a cut (`Vector+`) whose points satisfy `cut & point == 0`. A rotor moves points and cuts together
while preserving circles and their intersection angles. When every direction shares the same natural
rotation, one rotor carries the entire population: its bivector rate combines the shared rotation with
a boost `mean.commutator(time)` toward the population's mean direction, which acts on the sphere as a
tangential attraction.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext
from numga.algebras import STA as ga

mv = NumpyContext(ga).multivector
# Circles are cuts of the sphere, vectors; points are null rays, antivectors.
Cut = ga.gatype.vector()                                             # Vector+
Point = ga.gatype.antivector()                                       # Antivector-
Bivector = ga.gatype.bivector()                                      # Bivector+
Rotor = ga.gatype.rotor()                                            # Rotor+
# The rays' time direction pairs to one with `mv.t`; the dual carries the minus sign of `t`'s square.
time = -mv.t.dual()                                                  # [] Point-


# --- math -----------------------------------------------------------------------------
def directions(points: Point) -> Point:
    """Rescaling null rays (`[...] Point-`) to unit time and subtracting `time` recovers their unit
    spatial directions on the sphere."""
    return points / (mv.t & points) - time


def flow_rate(rotor: Rotor, seeds: Point, rotation: Bivector, coupling: float) -> Bivector:
    """The instantaneous bivector rate combines the shared `rotation` with a boost toward the
    population's mean direction (`[] Bivector+`)."""
    population = directions(rotor >> seeds)                         # [...] Point-
    mean = population.mean(axis=-1)                                 # [] Point-
    # Rescaling each null ray back to the sphere turns this boost into tangential attraction.
    return rotation - coupling * mean.commutator(time)              # [] Bivector+


def motion(initial: Rotor, seeds: Point, rotation: Bivector, coupling: float, dt: float) -> Iterator[Rotor]:
    """Yields the common Möbius rotor (`[] Rotor+`) from `initial` onward, advancing each step by the
    midpoint `flow_rate`."""
    rotor = initial                                                 # [] Rotor+
    while True:
        yield rotor
        halfway = (flow_rate(rotor, seeds, rotation, coupling) * (dt / 4)).exp() * rotor   # [] Rotor+
        rotor = (flow_rate(halfway, seeds, rotation, coupling) * (dt / 2)).exp() * rotor   # [] Rotor+


# --- plumbing -------------------------------------------------------------------------
def sphere_points(longitudes: np.ndarray, latitudes: np.ndarray) -> Point:
    """Turning the reference direction along `longitudes` and `latitudes` places unit spatial directions
    on the sphere (`[...] Point-`)."""
    longitude = (-mv.xy * (longitudes / 2)).exp()                   # [...] Rotor+
    latitude = (-mv.xz * (latitudes / 2)).exp()                     # [...] Rotor+
    return (longitude * latitude) >> mv.x.dual()                    # [...] Point-
