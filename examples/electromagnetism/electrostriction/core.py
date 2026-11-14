"""An elastic lattice of polarizable particles reaches coupled mechanical and electrical equilibrium in spacetime.

Particles and induced dipoles lie in a plane while interacting through the three-dimensional electrostatic
dipole field restricted to that plane, in units where four pi times vacuum permittivity equals one.
Positions, displacements, and induced dipoles are intrinsic spatial antivectors, while the electric field
is an extrinsic spacetime bivector whose pairing with the worldsheet patch swept by a dipole along the
observer's four-velocity gives the interaction energy. Taking the spacetime dual of that worldsheet map
identifies any spatial direction with the electric field along it and forms both the single-site
polarizability and the pairwise dipole-coupling operator. Harmonic springs connect nearest neighbours,
weak harmonic tethers prevent rigid drift, and keeping the coupled dipole solve inside the total free
energy includes dipole readjustment in both the force gradient and the mechanical Hessian.
"""

from __future__ import annotations

from collections.abc import Iterator

import jax
import numpy as np

from numga import Algebra, stack
from numga.algebras import STA as ga
from numga.backend.jax import JaxContext, derivative
from numga.sparse import Sparse

mv = JaxContext(ga).multivector
Scalar = ga.gatype.scalar()                                                    # Scalar+
Plane = ga.gatype.vector().derive.within(Algebra("x+y+"))                      # Vector+
# Particle positions, displacements, and induced dipoles are intrinsic spatial directions in the xy-plane;
# the electric field there is an extrinsic spacetime bivector with components xt and yt.
Point = Plane.derive.dual                                                      # Antivector-
Field = ga.gatype.bivector().derive.touching(Algebra("x+y+"), Algebra("t-"))   # Bivector+
Response = ga.gatype((Field, Point))                                           # Field+ <- Point-

# Observer four-velocity at rest with the lattice, pairing to one with `mv.t`.
time = -mv.t.dual()                                                            # Antivector-
# Spacetime duality map taking any spatial direction to the electric field along it.
field_along = (Point & time).dual()                                            # Field+ <- Point-


def lattice_positions(indices: np.ndarray, spacing: float) -> Point:
    """Integer axial coordinates (`[V, 2]`) map to reference lattice sites (`Point-[V]`) along two axes sixty degrees apart."""
    # Unit displacements along the two lattice directions, which meet at sixty degrees.
    first_axis = mv.x.dual()                                                   # [] Point-
    sixth_turn = (mv.xy * (-np.pi / 6)).exp()                                  # [] Rotor+
    axes = stack((first_axis, sixth_turn >> first_axis))                       # [2] Point-
    return (spacing * (axes * indices).sum(axis=-1)).field()


def dipole_response(positions: Point, polarizability: float) -> Response:
    """Single-site polarizability and pairwise dipole coupling form the linear field map (`[...] (Field+ <- Point-)[V <- V]`) from dipoles to applied fields."""
    sites = positions.batch()                                                  # [..., V] Point-
    n_particles = sites.shape[-1]
    separation = sites[..., :, None] - sites[..., None, :]                     # [..., V, V] Point-
    squared_distance = separation.squared()                                    # [..., V, V] Scalar+
    # Self pairs have a zero numerator, so adding one on the diagonal leaves them zero while keeping the denominator finite.
    distance = (squared_distance + np.eye(n_particles)).square_root()          # [..., V, V] Scalar+
    # Three times the radial field weighted by its pairing with the dipole's worldsheet patch, minus r^2 times the field along the dipole.
    separation_field = field_along(separation)                                 # [..., V, V] Field+
    dipole_law = 3 * separation_field * (separation_field & (Point & time)) - squared_distance * field_along
    coupling = (dipole_law / distance**5).field(0, 1)                          # [...] (Field+ <- Point-)[V <- V]
    return field_along / polarizability - coupling


def total_energy(
    positions: Point,
    rest_positions: Point,
    bond_ends: Sparse,
    applied_field: Field,
    polarizability: float,
    stiffness: float,
    tether_stiffness: float,
) -> Scalar:
    """Harmonic springs, weak site tethers, and equilibrated dipoles sum to the total free energy (`[...] Scalar+`) at fixed applied field."""
    rest_lengths = (bond_ends * rest_positions).antinorm()                     # Scalar+[E]
    extension = (bond_ends * positions).antinorm() - rest_lengths              # [...] Scalar+[E]
    bond_energy = stiffness / 2 * extension.squared().sites.sum()              # [...] Scalar+
    tether_energy = tether_stiffness / 2 * (positions - rest_positions).squared().sites.sum()
    # Solving for the dipoles inside the energy includes their readjustment in position derivatives.
    dipoles = dipole_response(positions, polarizability).solve(applied_field)  # [...] Point-[V]
    electric_energy = -(applied_field & (dipoles & time)).sites.sum() / 2      # [...] Scalar+
    return bond_energy + tether_energy + electric_energy


def mechanical_equilibrium(
    rest_positions: Point,
    bond_ends: Sparse,
    applied_field: Field,
    polarizability: float,
    stiffness: float,
    tether_stiffness: float,
    n_steps: int,
) -> Point:
    """Coupled Newton steps relax particle positions (`[...] Point-[V]`) from the reference lattice to electromechanical equilibrium."""
    @jax.jit
    def newton_step(positions: Point) -> Point:
        def energy(trial_positions: Point) -> Scalar:
            return total_energy(
                trial_positions, rest_positions, bond_ends, applied_field,
                polarizability, stiffness, tether_stiffness,
            )

        gradient = derivative(energy)(positions)                               # [...] (Scalar+ <- Point-)[V]
        curvature = derivative(derivative(energy))(positions)                  # [...] (Scalar+ <- (Point-, Point-))[V, V]
        return positions - curvature.solve(gradient)

    positions = rest_positions.broadcast_to(applied_field.shape)               # [...] Point-[V]
    for _ in range(n_steps):
        positions = newton_step(positions)
    return positions


def rms_extent(positions: Point, reference: Plane) -> Scalar:
    """Pairing centred particle offsets with a reference plane gives the root-mean-square extent (`[...] Scalar+`) along that direction."""
    offsets = positions - positions.sites.mean()                               # [...] Point-[V]
    return (reference & offsets).squared().sites.mean().square_root()


def axial_strain(positions: Point, rest_positions: Point, reference: Plane) -> Scalar:
    """Root-mean-square extent relative to the reference lattice gives the fractional axial strain (`[...] Scalar+`)."""
    return rms_extent(positions, reference) / rms_extent(rest_positions, reference) - 1


def turning_states(
    rest_positions: Point,
    bond_ends: Sparse,
    field_strength: float,
    polarizability: float,
    stiffness: float,
    tether_stiffness: float,
    n_frames: int,
    n_steps: int,
) -> Iterator[tuple[Point, Point, Field]]:
    """Yields equilibrium positions, induced dipoles, and applied field (`Point-[V]`, `Point-[V]`, `Field+`) across one revolution."""
    @jax.jit
    def frame_equilibrium(positions: Point, field: Field) -> tuple[Point, Point]:
        def energy(trial_positions: Point) -> Scalar:
            return total_energy(
                trial_positions, rest_positions, bond_ends, field,
                polarizability, stiffness, tether_stiffness,
            )

        for _ in range(n_steps):
            gradient = derivative(energy)(positions)                           # (Scalar+ <- Point-)[V]
            curvature = derivative(derivative(energy))(positions)              # (Scalar+ <- (Point-, Point-))[V, V]
            positions = positions - curvature.solve(gradient)
        dipoles = dipole_response(positions, polarizability).solve(field)      # Point-[V]
        return positions, dipoles

    angles = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    turns = (mv.xy * (-angles / 2)).exp()                                      # [n_frames] Rotor+
    fields = field_along(turns >> mv.x.dual()) * field_strength                # [n_frames] Field+
    positions = rest_positions                                                 # Point-[V]
    for field in fields:
        positions, dipoles = frame_equilibrium(positions, field)
        yield positions, dipoles, field
