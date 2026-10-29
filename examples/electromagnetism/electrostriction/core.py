"""Electrostriction: an elastic lattice of induced dipoles, at mechanical and electrical equilibrium.

Particles and dipoles are confined to a plane; their interaction is the three-dimensional
electrostatic dipole field restricted to that plane. Units set four pi times permittivity to one.
The particles' positions and their dipoles, a charge times a displacement, are antivectors,
intrinsic like all geometry. The electric field is a bivector of spacetime, extrinsic, through the
observer's time, and a dipole meets the field in the plane it sweeps through time, an energy
`field & (dipole & time)`. The field along a direction is a map from directions to fields, the dual
of the plane a direction sweeps through time, `(Point & time).dual()`: the vacuum's identification of a
direction with the field along it. The polarizability and the dipole law both act through it.
Central springs join neighbours, and weak tethers to the reference sites support the lattice.
The energy at fixed applied field includes the work exchanged with the source of that field.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np

from numga.algebras import STA as ga
from numga.backend.jax import JaxContext, derivative
from numga.sparse import SparseExtensor
from examples import instantiate

mv = JaxContext(ga).multivector
Mesh = instantiate("examples.util.mesh", ga, JaxContext).Mesh
Scalar = ga.gatype.scalar()                                # Scalar+
Vector = ga.gatype.vector()                                # Vector+
# Positions, displacements and dipoles are directions in the plane, antivectors; the electric field
# there is a bivector through time, its planes xt and yt.
Point = ga.gatype.antivector().derive.containing("z t")    # Antivector-
Field = ga.gatype.bivector().derive.within("x y t").derive.containing("t")    # Bivector+
Response = ga.gatype((Field, Point))                       # Field+ <- Point-

# The direction in time of an observer at rest with the lattice, which pairs to one with `mv.t`; the
# dual carries the minus of t's square.
time = -mv.t.dual()                                        # [] Antivector-
# The field along a direction, as strong as the direction is long: the dual of the plane the direction sweeps
# through time, one map with the direction left open. The metric enters here, as the vacuum's identification of a
# direction with the field along it.
field_along = (Point & time).dual()                         # [] Field+ <- Point-


# --- math -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Lattice:
    rest: Point                          # Point-[particles]
    bonds: np.ndarray                    # [bonds, ends] site indices
    polarizability: float
    stiffness: float
    tether: float

    @property
    def bond_ends(self) -> SparseExtensor:
        """Each bond's second particle less its first, from the particles to the bonds."""
        n_particles = self.rest.gatype.site_shape[0]
        return Mesh(n_particles, self.bonds).d[0]

    @property
    def lengths(self) -> Scalar:
        return (self.bond_ends * self.rest).squared().square_root()           # Scalar+[bonds]

    def response(self, positions: Point) -> Response:
        """The field each dipole needs, its own restoring field less the field supplied by the
        other dipoles: a map from the dipoles at all particles to the fields there."""
        sites = positions.batch()
        separation = sites[..., :, None] - sites[..., None, :]
        squared_distance = separation.squared()
        # A particle has no self interaction. Its numerator vanishes identically, so the value
        # added on the diagonal is arbitrary; it keeps the denominator finite.
        n_particles = self.rest.gatype.site_shape[0]
        distance = (squared_distance + np.eye(n_particles)).square_root()
        # A dipole's field at a separation: three times the field along the separation, weighted by
        # its pairing with the plane the dipole sweeps, less the separation's square times the field
        # along the dipole.
        along = field_along(separation)                                        # [particles, particles] Field+
        dipole_law = 3 * along * (along & (Point & time)) - squared_distance * field_along
        coupling = (dipole_law / distance**5).field(0, 1)                      # (Field+ <- Point-)[particles <- particles]
        return field_along / self.polarizability - coupling

    def elastic_energy(self, positions: Point) -> Scalar:
        """Stretching energy of the bonds and displacement energy of the supporting tethers."""
        extension = (self.bond_ends * positions).squared().square_root() - self.lengths
        bonds = self.stiffness / 2 * extension.squared().sites.sum()
        tethers = self.tether / 2 * (positions - self.rest).squared().sites.sum()
        return bonds + tethers

    def energy(self, positions: Point, applied: Field) -> Scalar:
        """Mechanical energy plus polarization energy after the dipoles have equilibrated."""
        dipoles = self.response(positions).solve(applied)
        # Building induced dipoles costs energy. At their minimum the total electric energy,
        # including that cost and mutual interactions, is minus half their pairing with the field.
        electrical = -(applied & (dipoles & time)).sites.sum() / 2
        return self.elastic_energy(positions) + electrical

    def equilibrium(self, applied: Field, n_steps: int) -> Point:
        """Newton relaxation from the reference lattice, for a batch of applied fields."""
        def energy(positions: Point) -> Scalar:
            return self.energy(positions, applied)

        @jax.jit
        def newton(positions: Point) -> Point:
            gradient = derivative(energy)(positions)                   # (Scalar+ <- Point-)[particles]
            curvature = derivative(derivative(energy))(positions)      # (Scalar+ <- (Point-, Point-))[particles, particles]
            # The polarization solve is inside the energy: its response to motion contributes
            # to the mechanical curvature as well as the first derivative.
            return positions - curvature.solve(gradient)

        positions = self.rest.broadcast_to(applied.shape)
        for _ in range(n_steps):
            positions = newton(positions)
        return positions


def extent(positions: Point, reference: Vector) -> Scalar:
    """Root mean square distance of the sites from a reference slice through their centre: the pairing
    of each centred position with the slice."""
    centred = positions - positions.sites.mean()
    return (reference & centred).squared().sites.mean().square_root()


def strain(positions: Point, rest: Point, reference: Vector) -> Scalar:
    """The relative change of the root mean square distance from a reference slice, from the rest sites."""
    return extent(positions, reference) / extent(rest, reference) - 1


# --- plumbing -------------------------------------------------------------------------
def hexagonal_indices(n_rings: int) -> tuple[np.ndarray, np.ndarray]:
    """Axial indices `[particles, axes]` of a triangular lattice patch, and its nearest-neighbour
    bonds `[bonds, ends]`."""
    axis = np.arange(-n_rings, n_rings + 1)
    first, second = np.meshgrid(axis, axis, indexing="ij")
    inside = np.maximum.reduce([np.abs(first), np.abs(second), np.abs(first + second)]) <= n_rings
    first, second = first[inside], second[inside]
    delta_first = first[:, None] - first[None, :]
    delta_second = second[:, None] - second[None, :]
    neighbours = np.maximum.reduce([np.abs(delta_first), np.abs(delta_second),
                                    np.abs(delta_first + delta_second)]) == 1
    bonds = np.stack(np.nonzero(np.triu(neighbours, 1)), axis=-1)
    return np.stack([first, second], axis=-1), bonds
