"""Odd elasticity from active bonds that push sideways when stretched.

Positions and displacements are directions, antivectors; forces are covectors, vectors, and the work of
a force over a displacement is their pairing, `force & displacement`. A bond reads its extension with
the covector along it and answers with a force that has a sideways part, turned in the actuation plane,
so its response, `Vector <- Antivector`, does net work around a closed path. The pairing transpose,
`motion.adjugate()`, carries forces back through any map of displacements with their work kept, from
bonds to the deformations that move them and to the particles at their ends.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.sparse import SparseExtensor, spdiag

ga = Algebra("x+y+")
ctx = NumpyContext(ga)
mv = ctx.multivector
Scalar = ga.gatype.scalar()
# Positions and displacements are antivectors; forces are covectors, vectors.
Antivector = ga.gatype.antivector()
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Response = ga.gatype((Vector, Antivector))                   # from displacements to forces
# The two amplitudes of a shear, geometric like displacements.
Shear = Antivector

# The lines across the two axes, and the directions along them, which they pair with to the identity.
lines = stack((mv.x, mv.y))                                  # [2] Vector
directions = lines.dual()                                    # [2] Antivector


# --- plumbing -------------------------------------------------------------------------
def triangular_patch(n_columns: int, n_rows: int, spacing: float) -> tuple[Antivector, np.ndarray, SparseExtensor]:
    """A triangular lattice with free edges, its positions and oriented bond incidence."""
    columns, rows = np.meshgrid(np.arange(n_columns), np.arange(n_rows))
    positions = spacing * (
        directions[0] * (columns + (rows % 2) / 2 - (n_columns - 1) / 2)
        + directions[1] * ((rows - (n_rows - 1) / 2) * np.sqrt(3) / 2)
    )                                                                       # [rows, columns] Antivector
    positions = positions.reshape(-1).field()                                # Antivector[sites]

    indices = np.arange(n_columns * n_rows).reshape(n_rows, n_columns)
    horizontal = np.stack((indices[:, :-1], indices[:, 1:]), axis=-1).reshape(-1, 2)
    vertical = np.stack((indices[:-1], indices[1:]), axis=-1).reshape(-1, 2)
    lower = indices[:-1, :-1] + ((np.arange(n_rows - 1) + 1) % 2)[:, None]
    upper = indices[1:, :-1] + (np.arange(n_rows - 1) % 2)[:, None]
    diagonal = np.stack((lower, upper), axis=-1).reshape(-1, 2)
    edges = np.concatenate((horizontal, vertical, diagonal))
    incidence = SparseExtensor.from_columns(
        edges, mv.scalar((np.ones_like(edges) * [-1, 1])[..., None]), n_columns * n_rows,
    )                                                                       # Scalar[edges, sites]
    return positions, edges, incidence


# --- math -----------------------------------------------------------------------------
def bond_response(direction: Antivector, actuation_plane: Bivector, stiffness: float, activity: float) -> Response:
    """The restoring and sideways force caused by a bond's endpoint displacement."""
    # The covector that reads a displacement's component along the unit bond.
    along = direction.dual_inverse()                                         # Vector
    push = -stiffness * along + activity * (along | actuation_plane)         # Vector
    return push * (along & Antivector)                                       # Vector <- Antivector


def integrate_cycle(response: Response, cycle: Antivector) -> Iterator[tuple[Antivector, Vector, Scalar]]:
    """Displacement, force and work delivered along consecutive straight path segments."""
    work = mv.scalar([0])                                                    # Scalar
    yield cycle[0], response(cycle[0]), work
    for start, end in zip(cycle[:-1], cycle[1:]):
        # A linear force's work along a straight segment is its midpoint force
        # paired with the segment's displacement.
        work = work + (response((start + end) / 2) & (end - start))           # Scalar
        yield end, response(end), work


def evolution(incidence: SparseExtensor, response: Response, drags: Scalar) -> SparseExtensor:
    """The particles' velocities from their displacements, overdamped: no inertia, every force met by drag."""
    gradient = incidence * Antivector                                        # Antivector[bonds] <- Antivector[sites]
    # Drag turns each particle's force into its velocity, along the force's metric direction.
    mobility = spdiag(Vector.dual() / drags)                                 # Antivector[sites] <- Vector[sites]
    # Relative endpoint motion goes to each bond; equal opposite forces return to its endpoints.
    return mobility(gradient.adjugate()(spdiag(response)(gradient)))         # Antivector[sites] <- Antivector[sites]


def rk4_step(generator: SparseExtensor, displacement: Antivector, dt: float) -> Antivector:
    """One fourth-order Runge–Kutta step of an overdamped displacement field."""
    first_rate = generator(displacement)                                     # Antivector[sites]
    second_rate = generator(displacement + first_rate * (dt / 2))            # Antivector[sites]
    third_rate = generator(displacement + second_rate * (dt / 2))             # Antivector[sites]
    fourth_rate = generator(displacement + third_rate * dt)                   # Antivector[sites]
    return displacement + (first_rate + 2 * second_rate + 2 * third_rate + fourth_rate) * (dt / 6)


def wave_states(
    reference: Antivector, incidence: SparseExtensor, bond_directions: Antivector, generator: SparseExtensor,
    initial: Antivector, n_frames: int, dt: float,
) -> Iterator[tuple[Antivector, Scalar]]:
    """The moving lattice and bond extensions during its free overdamped response."""
    displacement = initial                                                  # Antivector[sites]
    for _ in range(n_frames):
        extensions = bond_directions.dual_inverse() & (incidence * displacement)   # Scalar[edges]
        yield reference + displacement, extensions
        displacement = rk4_step(generator, displacement, dt)                 # Antivector[sites]
