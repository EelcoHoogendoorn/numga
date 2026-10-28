"""Spherical synchronization: directions on a sphere turning while attracting one another, carried by
one evolving Möbius transformation, in the algebra `x+y+z+t-` of the sphere's conformal geometry.

A point of the sphere is modelled as a null ray, an antivector, `direction + time`, and a circle as a cut, a vector,
through which a point passes when `cut & point == 0`. A rotor of this algebra moves points and cuts alike
and keeps the sphere's circles and their angles: it is a Möbius transformation. When every direction has
the same natural rotation, the whole population follows one such rotor: its generator is the shared
rotation less a boost towards the population's mean, `mean.commutator(time)`, the turn from time towards
the mean, which on the sphere is a tangential attraction.
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
# The rays' time direction, which pairs to one with `mv.t`; the dual carries the minus of t's square.
time = -mv.t.dual()                                                  # [] Point-


# --- math -----------------------------------------------------------------------------
def directions(points: Point) -> Point:
    """The unit directions on the sphere of null rays: each scaled to unit time, less its time."""
    return points / (mv.t & points) - time


def generator(rotor: Rotor, seeds: Point, rotation: Bivector, coupling: float) -> Bivector:
    """The shared rotation, less a boost towards the population's mean direction."""
    population = directions(rotor >> seeds)                         # [particles] Point-
    mean = population.mean(axis=-1)                                 # [] Point-
    # A boost towards the mean becomes tangential attraction on the sphere.
    return rotation - coupling * mean.commutator(time)              # [] Bivector+


def motion(initial: Rotor, seeds: Point, rotation: Bivector, coupling: float, dt: float) -> Iterator[Rotor]:
    """The common transformation from the start and after each step, without end, the generator taken
    at the midpoint of each step."""
    rotor = initial                                                # [] Rotor+
    while True:
        yield rotor
        halfway = (generator(rotor, seeds, rotation, coupling) * (dt / 4)).exp() * rotor   # [] Rotor+
        rotor = (generator(halfway, seeds, rotation, coupling) * (dt / 2)).exp() * rotor    # [] Rotor+


# --- plumbing -------------------------------------------------------------------------
def sphere_points(longitudes: np.ndarray, latitudes: np.ndarray) -> Point:
    """Unit directions at the given longitudes and latitudes: a reference direction turned twice."""
    longitude = (-mv.xy * (longitudes / 2)).exp()                   # [...] Rotor+
    latitude = (-mv.xz * (latitudes / 2)).exp()                     # [...] Rotor+
    return (longitude * latitude) >> mv.x.dual()                    # [...] Point-
