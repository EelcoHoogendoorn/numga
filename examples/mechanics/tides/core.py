"""Newtonian gravity of soft masses, its tidal map, and a cluster of stars stretched by it.

Each mass is a Plummer sphere, a mass with a soft core: beyond the core it pulls as a point mass, as
one over the squared distance, and its density is smooth and finite at the centre. The acceleration
is a sum over the masses, and so is the tidal map at a point: a map on displacements,
`Point <- Point`, the change in acceleration over a small step, written in closed form.

Contracting the tidal map against the reciprocal of an open point, its reverse, gives the derivative of
gravity, `(Point.reverse() * tidal(Point)).contract(1, 2)`: its scalar part is minus four pi times the
density, which is Poisson's equation, and its bivector part, the curl, vanishes. A cluster of stars falls through the
field; its shape, to first order in its size, is a map too, the deformation, which the tidal map
drives along the cluster's centre: `deformation'' == tidal(deformation)`, the Jacobi equation. A
round cluster's spread is then its starting radius squared times `deformation(deformation.adjoint())`,
and its rim the offsets `x` with `x.dot(spread.solve(x)) == 1`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# Positions, displacements, velocities and accelerations are antivectors with intrinsic orientation.
Point = ga.gatype.antivector()
Even = ga.gatype.even()
Tidal = ga.gatype((Point, Point))                           # Point <- Point
Deformation = ga.gatype((Point, Point))                     # Point <- Point
Form = ga.gatype((Scalar, Point, Point))                    # Scalar <- (Point, Point)


# --- math -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Masses:
    centres: Point                                           # [masses] Point
    masses: np.ndarray                                       # [masses] mass of each
    core_radius: float

    def acceleration(self, points: Point) -> Point:
        """The pull of the masses at each point."""
        towards = self.centres - points[..., None]                           # [..., masses] Point
        pull, _ = self._profile(towards)                                     # [..., masses] Scalar
        return (pull * towards).sum(axis=-1)                                 # [...] Point

    def tidal(self, points: Point) -> Tidal:
        """The tidal map at each point: the change in acceleration for a small displacement."""
        towards = self.centres - points[..., None]                           # [..., masses] Point
        pull, pull_change = self._profile(towards)                           # [..., masses] Scalar each
        # A displacement moves away from each mass by itself, and changes the pull by its component
        # towards the mass, which acts along the direction to the mass.
        squeeze = -pull * Point                                              # [..., masses] Point <- Point
        stretch = -2 * pull_change * towards * towards.dot(Point)              # [..., masses] Point <- Point
        return (squeeze + stretch).sum(axis=-1)                              # [...] Point <- Point

    def _profile(self, towards: Point) -> tuple[Scalar, Scalar]:
        """The pull per unit distance towards each mass, and its change with the squared distance."""
        softened = towards.scalar_norm_squared() + self.core_radius**2       # [..., masses] Scalar
        pull = self.masses / (softened * softened.square_root())             # [..., masses] Scalar
        return pull, -1.5 * pull / softened


@dataclass(frozen=True)
class Stars:
    positions: Point                                         # [..., stars] Point
    velocities: Point                                        # [..., stars] Point

    def centre(self) -> Point:
        """The stars' centre of mass."""
        return self.positions.mean(axis=-1)                                  # [...] Point

    def moment(self) -> Deformation:
        """The stars' spread about their centre of mass: the mean of each offset times its component
        along the open point."""
        offsets = self.positions - self.centre()[..., None]                # [..., stars] Point
        return (offsets * offsets.dot(Point)).mean(axis=-1)                  # [...] Point <- Point

    def fall(self, masses: Masses, dt: float) -> Stars:
        """The stars moved for a time `dt` by the velocity Verlet rule."""
        start = masses.acceleration(self.positions)                          # [..., stars] Point
        positions = self.positions + self.velocities * dt + start * (dt**2 / 2)
        end = masses.acceleration(positions)                                 # [..., stars] Point
        return Stars(positions, self.velocities + (start + end) * (dt / 2))


@dataclass(frozen=True)
class Shape:
    """The cluster's centre and its deformation: how an offset from the centre at the start has moved."""
    centre: Point                                            # [] Point
    velocity: Point                                          # [] Point
    deformation: Deformation                                 # [] Point <- Point
    rate: Deformation                                        # [] Point <- Point

    def deform(self, masses: Masses, dt: float) -> Shape:
        """The centre moved and the deformation carried by the tidal map at it, by the velocity Verlet
        rule: the deformation's rate of change changes by the tidal map applied to the deformation."""
        start = masses.acceleration(self.centre)                             # [] Point
        start_tide = masses.tidal(self.centre)(self.deformation)             # [] Point <- Point
        centre = self.centre + self.velocity * dt + start * (dt**2 / 2)
        deformation = self.deformation + self.rate * dt + start_tide * (dt**2 / 2)
        end = masses.acceleration(centre)                                    # [] Point
        end_tide = masses.tidal(centre)(deformation)                         # [] Point <- Point
        return Shape(centre, self.velocity + (start + end) * (dt / 2),
                     deformation, self.rate + (start_tide + end_tide) * (dt / 2))

    def spread(self, radius: float) -> Deformation:
        """Where a round cluster of the given starting radius has spread to, to first order in its size:
        its rim's offsets `x` have `x.dot(spread.solve(x)) == 1`."""
        return radius**2 * self.deformation(self.deformation.adjoint())     # [] Point <- Point

    def rim(self, radius: float) -> Form:
        """The rim of a round cluster of the given starting radius, carried by the deformation: the form
        that is one, on the same offset twice, at the rim's offsets from the centre."""
        return Point.dot(self.spread(radius).solve(Point))                   # [] Scalar <- (Point, Point)


def derivative(tides: Tidal) -> Even:
    """The derivative of gravity: minus four pi times the density, plus the curl. The reciprocal of a unit
    antivector is its reverse."""
    return (Point.reverse() * tides(Point)).contract(1, 2)                  # [...] Even


# --- plumbing -------------------------------------------------------------------------
def grid(half_width: float, half_height: float, columns: int, rows: int) -> Point:
    """Displacements on a grid in the plane z = 0, at the centres of its cells."""
    x = (np.arange(columns) + 0.5) / columns * 2 * half_width - half_width
    y = (np.arange(rows) + 0.5) / rows * 2 * half_height - half_height
    return (mv.x * x[None, :] + mv.y * y[:, None]).dual().cast(Point)       # [rows, columns] Point
