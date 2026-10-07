"""Newtonian gravity of soft masses, its tidal map, and a cluster of stars stretched by it.

Each mass is a Plummer sphere, a mass with a soft core: beyond the core it pulls as a point mass, as
one over the squared distance, and its density is smooth and finite at the centre. The acceleration
is a sum over the masses, and so is the tidal map at a point: a map on displacements,
`Vector <- Vector`, the change in acceleration over a small step, written in closed form.

Contracting the tidal map against an open vector gives the derivative of gravity,
`(Vector * tidal(Vector)).contract(1, 2)`: its scalar part is minus four pi times the density, which is
Poisson's equation, and its bivector part, the curl, vanishes. A cluster of stars falls through the
field; its shape, to first order in its size, is a map too, the deformation, which the tidal map
drives along the cluster's centre: `deformation'' == tidal(deformation)`, the Jacobi equation. A
round cluster's spread is then its starting radius squared times `deformation(deformation.adjoint())`,
and its rim the offsets `x` with `x | spread.solve(x) == 1`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from numga import NumpyContext
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Even = ga.gatype.even()
Tidal = ga.gatype((Vector, Vector))                         # Vector <- Vector
Deformation = ga.gatype((Vector, Vector))                   # Vector <- Vector
Form = ga.gatype((Scalar, Vector, Vector))                  # Scalar <- (Vector, Vector)


@dataclass(frozen=True)
class Masses:
    centres: Vector                                          # [masses] Vector
    masses: np.ndarray                                       # [masses] mass of each
    core_radius: float


@dataclass(frozen=True)
class Stars:
    positions: Vector                                        # [..., stars] Vector
    velocities: Vector                                       # [..., stars] Vector

    def centre(self) -> Vector:
        """The stars' centre of mass."""
        return self.positions.sum(axis=-1) / self.positions.shape[-1]      # [...] Vector

    def moment(self) -> Deformation:
        """The stars' spread about their centre of mass: the mean of each offset times its component
        along the open vector."""
        offsets = self.positions - self.centre()[..., None]                # [..., stars] Vector
        return (offsets * (offsets | Vector)).sum(axis=-1) / self.positions.shape[-1]   # [...] Vector <- Vector


@dataclass(frozen=True)
class Shape:
    """The cluster's centre and its deformation: how an offset from the centre at the start has moved."""
    centre: Vector                                           # [] Vector
    velocity: Vector                                         # [] Vector
    deformation: Deformation                                 # [] Vector <- Vector
    rate: Deformation                                        # [] Vector <- Vector


# --- math -----------------------------------------------------------------------------
def acceleration(masses: Masses, points: Vector) -> Vector:
    """The pull of the masses at each point."""
    towards = masses.centres - points[..., None]                             # [..., masses] Vector
    pull, _ = _profile(masses, towards)                                      # [..., masses] Scalar
    return (pull * towards).sum(axis=-1)                                     # [...] Vector


def tidal(masses: Masses, points: Vector) -> Tidal:
    """The tidal map at each point: the change in acceleration for a small displacement."""
    towards = masses.centres - points[..., None]                             # [..., masses] Vector
    pull, pull_change = _profile(masses, towards)                            # [..., masses] Scalar each
    # A displacement moves away from each mass by itself, and changes the pull by its component
    # towards the mass, which acts along the direction to the mass.
    squeeze = -pull * Vector                                                 # [..., masses] Vector <- Vector
    stretch = -2 * pull_change * towards * (towards | Vector)                # [..., masses] Vector <- Vector
    return (squeeze + stretch).sum(axis=-1)                                  # [...] Vector <- Vector


def derivative(tides: Tidal) -> Even:
    """The derivative of gravity: minus four pi times the density, plus the curl."""
    return (Vector * tides(Vector)).contract(1, 2)                          # [...] Even


def fall(masses: Masses, stars: Stars, dt: float) -> Stars:
    """The stars moved for a time `dt` by the velocity Verlet rule."""
    start = acceleration(masses, stars.positions)                            # [..., stars] Vector
    positions = stars.positions + stars.velocities * dt + start * (dt**2 / 2)
    end = acceleration(masses, positions)                                    # [..., stars] Vector
    return Stars(positions, stars.velocities + (start + end) * (dt / 2))


def deform(masses: Masses, shape: Shape, dt: float) -> Shape:
    """The centre moved and the deformation carried by the tidal map at it, by the velocity Verlet
    rule: the deformation's rate of change changes by the tidal map applied to the deformation."""
    start = acceleration(masses, shape.centre)                               # [] Vector
    start_tide = tidal(masses, shape.centre)(shape.deformation)              # [] Vector <- Vector
    centre = shape.centre + shape.velocity * dt + start * (dt**2 / 2)
    deformation = shape.deformation + shape.rate * dt + start_tide * (dt**2 / 2)
    end = acceleration(masses, centre)                                       # [] Vector
    end_tide = tidal(masses, centre)(deformation)                            # [] Vector <- Vector
    return Shape(centre, shape.velocity + (start + end) * (dt / 2),
                 deformation, shape.rate + (start_tide + end_tide) * (dt / 2))


def spread(shape: Shape, radius: float) -> Deformation:
    """Where a round cluster of the given starting radius has spread to, to first order in its size:
    its rim's offsets `x` have `x | spread.solve(x) == 1`."""
    return radius**2 * shape.deformation(shape.deformation.adjoint())       # [] Vector <- Vector


def rim(shape: Shape, radius: float) -> Form:
    """The rim of a round cluster of the given starting radius, carried by the deformation: the form
    that is one, on the same offset twice, at the rim's offsets from the centre."""
    return Vector | spread(shape, radius).inverse()                  # [] Scalar <- (Vector, Vector)


def _profile(masses: Masses, towards: Vector) -> tuple[Scalar, Scalar]:
    """The pull per unit distance towards each mass, and its change with the squared distance."""
    softened = (towards | towards) + masses.core_radius**2                   # [..., masses] Scalar
    pull = masses.masses / (softened * softened.square_root())               # [..., masses] Scalar
    return pull, -1.5 * pull / softened


# --- plumbing -------------------------------------------------------------------------
def grid(half_width: float, half_height: float, columns: int, rows: int) -> Vector:
    """Displacements on a grid in the plane z = 0, at the centres of its cells."""
    x = (np.arange(columns) + 0.5) / columns * 2 * half_width - half_width
    y = (np.arange(rows) + 0.5) / rows * 2 * half_height - half_height
    return (mv.x * x[None, :] + mv.y * y[:, None]).cast(Vector)              # [rows, columns] Vector
