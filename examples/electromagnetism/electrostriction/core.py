"""An elastic lattice of induced dipoles, at mechanical and electrical equilibrium.

Particles and dipoles are confined to a plane; their interaction is the three-dimensional
electrostatic dipole field restricted to that plane. Units set four pi times permittivity to one.
Central springs join neighbours, and weak tethers to the reference sites support the lattice.
The energy at fixed applied field includes the work exchanged with the source of that field.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import jax
import numpy as np

from numga import Algebra
from numga.backend.jax import JaxContext, derivative
from numga.sparse import SparseExtensor

jax.config.update("jax_enable_x64", True)
ga = Algebra("x+y+")
mv = JaxContext(ga, np.float64).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Response = ga.gatype((Vector, Vector))


# --- math -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Lattice:
    rest: Vector                         # Vector[particles]
    bonds: np.ndarray                    # [bonds, ends] site indices
    polarizability: float
    stiffness: float
    tether: float

    @cached_property
    def bond_ends(self) -> SparseExtensor:
        """Each bond's second particle less its first, from the particles to the bonds."""
        particles = self.rest.gatype.site_shape[0]
        ends = (np.ones_like(self.bonds) * [-1, 1])[..., None]                # [bonds, ends, 1]
        return SparseExtensor.from_columns(self.bonds, mv.scalar(ends), particles)

    def links(self, positions: Vector) -> Vector:
        """Each elastic bond, directed from its first particle to its second."""
        return self.bond_ends * positions                                      # Vector[bonds]

    @cached_property
    def lengths(self) -> Scalar:
        return self.links(self.rest).norm()

    def response(self, positions: Vector) -> Response:
        """The dipoles' own restoring field, less the field supplied by the other dipoles."""
        sites = positions.batch()
        separation = sites[..., :, None] - sites[..., None, :]
        squared_distance = separation.squared()
        # A particle has no self interaction. Its numerator vanishes identically, so the value
        # added on the diagonal is arbitrary; it keeps the denominator finite.
        particles = self.rest.gatype.site_shape[0]
        distance = (squared_distance + np.eye(particles)).square_root()
        coupling = ((3 * separation * (separation | Vector) - squared_distance * Vector)
                    / distance**5).field(0, 1)                         # Vector[particles] <- Vector[particles]
        return Vector / self.polarizability - coupling

    def elastic_energy(self, positions: Vector) -> Scalar:
        """Stretching energy of the bonds and displacement energy of the supporting tethers."""
        extension = self.links(positions).norm() - self.lengths
        bonds = self.stiffness / 2 * extension.squared().sites.sum()
        tethers = self.tether / 2 * (positions - self.rest).scalar_norm_squared().sites.sum()
        return bonds + tethers

    def energy(self, positions: Vector, applied: Vector) -> Scalar:
        """Mechanical energy plus polarization energy after the dipoles have equilibrated."""
        dipoles = self.response(positions).solve(applied)
        # Building induced dipoles costs energy. At their minimum the total electric energy,
        # including that cost and mutual interactions, is minus half their pairing with the field.
        electrical = -(applied | dipoles).sites.sum() / 2
        return self.elastic_energy(positions) + electrical

    def equilibrium(self, applied: Vector, steps: int) -> Vector:
        """Newton relaxation from the reference lattice, for a batch of applied fields."""
        def energy(positions: Vector) -> Scalar:
            return self.energy(positions, applied)

        positions = self.rest.broadcast_to(applied.shape)
        for _ in range(steps):
            gradient = derivative(energy)(positions)                   # Scalar <- Vector[particles]
            curvature = derivative(derivative(energy))(positions)      # Scalar <- (Vector[particles], Vector[particles])
            # The polarization solve is inside the energy: its response to motion contributes
            # to the mechanical curvature as well as the first derivative.
            positions = positions - curvature.solve(gradient)
        return positions


def extent(positions: Vector, direction: Vector) -> Scalar:
    """Root mean square distance along a direction, measured from the field's centre."""
    centred = positions - positions.sites.mean()
    return (centred | direction).squared().sites.mean().square_root()


def strain(positions: Vector, rest: Vector, direction: Vector) -> Scalar:
    """The relative change of the root mean square extent along a direction, from the rest sites."""
    return extent(positions, direction) / extent(rest, direction) - 1


# --- plumbing -------------------------------------------------------------------------
def hexagonal_indices(rings: int) -> tuple[np.ndarray, np.ndarray]:
    """Axial indices `[particles, axes]` of a triangular lattice patch, and its nearest-neighbour
    bonds `[bonds, ends]`."""
    axis = np.arange(-rings, rings + 1)
    first, second = np.meshgrid(axis, axis, indexing="ij")
    inside = np.maximum.reduce([np.abs(first), np.abs(second), np.abs(first + second)]) <= rings
    first, second = first[inside], second[inside]
    delta_first = first[:, None] - first[None, :]
    delta_second = second[:, None] - second[None, :]
    neighbours = np.maximum.reduce([np.abs(delta_first), np.abs(delta_second),
                                    np.abs(delta_first + delta_second)]) == 1
    bonds = np.stack(np.nonzero(np.triu(neighbours, 1)), axis=-1)
    return np.stack([first, second], axis=-1), bonds
