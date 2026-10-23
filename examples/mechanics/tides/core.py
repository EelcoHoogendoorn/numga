"""Newtonian gravity of soft masses, its tidal map, and a cluster of stars stretched by it.

Each mass is a Plummer sphere, a mass with a soft core: beyond the core it pulls as a point mass, as
one over the squared distance, and its density is smooth and finite at the centre. The acceleration
is a sum over the masses, and so is the tidal map at a point: the change in gravity over a small step,
a map from displacements to the planes that read the change, `Tidal = Plane <- Point`, written in
closed form. The metric turns its reading into the change in acceleration.

The geometric derivative of gravity multiplies each plane the tidal map gives by the plane that reads
its step, `(Plane * tidal(Point)).trace(1, 2)`. Its outer half, the curl, vanishes with no metric
involved, since the tidal map is symmetric; its inner half, where the metric enters, is minus four pi
times the density, which is Poisson's equation. A cluster of stars falls through the field; its shape,
to first order in its size, is a map too, the deformation, which the tidal map drives along the
cluster's centre: `deformation'' == tidal(deformation).dual()`, the Jacobi equation. The stars' spread
is a map from planes to offsets, `Spread = Point <- Plane`, each offset weighted by its reading by the
plane; the deformation carries it forward and its adjugate carries the planes back,
`deformation(spread(deformation.adjugate()))`. A round cluster starts as its radius squared times
`Plane.dual()`, roundness being where the metric enters, and its rim is the offsets `x` with
`spread.solve(x) & x == 1`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# Positions, displacements, velocities and accelerations are antivectors with intrinsic orientation; the
# planes that read them are vectors, with extrinsic orientation.
Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
Even = ga.gatype.even()
Tidal = ga.gatype((Plane, Point))                           # Plane <- Point
Deformation = ga.gatype((Point, Point))                     # Point <- Point
Spread = ga.gatype((Point, Plane))                          # Point <- Plane
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
        """The tidal map at each point: the change in gravity for a small displacement, as the plane that
        reads it."""
        towards = self.centres - points[..., None]                           # [..., masses] Point
        pull, pull_change = self._profile(towards)                           # [..., masses] Scalar each
        # The planes across the line to each mass, which read a step's part towards it.
        across = towards.dual_inverse()                                      # [..., masses] Plane
        # A displacement moves away from each mass by itself, weakening the pull along every direction,
        # and changes the pull by its reading by the plane across the line to the mass, along that line.
        squeeze = -pull * Point.dual_inverse()                               # [..., masses] Plane <- Point
        stretch = -2 * pull_change * across * (across & Point)               # [..., masses] Plane <- Point
        return (squeeze + stretch).sum(axis=-1)                              # [...] Plane <- Point

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

    def moment(self) -> Spread:
        """The stars' spread about their centre of mass, a map from planes to offsets: the mean of each
        offset weighted by its reading by the plane."""
        offsets = self.positions - self.centre()[..., None]                # [..., stars] Point
        return (offsets * (Plane & offsets)).mean(axis=-1)                  # [...] Point <- Plane

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
        # The metric turns the tidal map's reading into the change in acceleration.
        start = masses.acceleration(self.centre)                             # [] Point
        start_tide = masses.tidal(self.centre)(self.deformation).dual()      # [] Point <- Point
        centre = self.centre + self.velocity * dt + start * (dt**2 / 2)
        deformation = self.deformation + self.rate * dt + start_tide * (dt**2 / 2)
        end = masses.acceleration(centre)                                    # [] Point
        end_tide = masses.tidal(centre)(deformation).dual()                  # [] Point <- Point
        return Shape(centre, self.velocity + (start + end) * (dt / 2),
                     deformation, self.rate + (start_tide + end_tide) * (dt / 2))

    def spread(self, radius: float) -> Spread:
        """Where a round cluster of the given starting radius has spread to, to first order in its size:
        the deformation carries the offsets forward and its adjugate carries the planes back. Its rim's
        offsets `x` have `spread.solve(x) & x == 1`."""
        # Round is where the metric enters: each plane reads the offsets across it, out to the radius.
        start = radius**2 * Plane.dual()                                     # [] Spread
        return self.deformation(start(self.deformation.adjugate()))          # [] Point <- Plane

    def rim(self, radius: float) -> Form:
        """The rim of a round cluster of the given starting radius, carried by the deformation: the form
        that is one, on the same offset twice, at the rim's offsets from the centre."""
        return self.spread(radius).solve(Point) & Point                      # [] Scalar <- (Point, Point)


def derivative(tides: Tidal) -> Even:
    """The geometric derivative of gravity: its inner half, where the metric enters, minus four pi times
    the density, and its outer half the curl, which needs none."""
    # The trace pairs each open plane with the displacement the tidal map reads, plane against point.
    return (Plane * tides(Point)).trace(1, 2)                               # [...] Even


# --- plumbing -------------------------------------------------------------------------
def grid(half_width: float, half_height: float, columns: int, rows: int) -> Point:
    """Displacements on a grid in the plane z = 0, at the centres of its cells."""
    x = (np.arange(columns) + 0.5) / columns * 2 * half_width - half_width
    y = (np.arange(rows) + 0.5) / rows * 2 * half_height - half_height
    return (mv.x * x[None, :] + mv.y * y[:, None]).dual().cast(Point)       # [rows, columns] Point
