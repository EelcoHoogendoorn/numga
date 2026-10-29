"""Odd elasticity from active bonds that push sideways when stretched.

Positions and displacements are directions, antivectors; a force is a vector, extrinsic, read by its
pairing with a displacement, the work `force & displacement`. A bond reads its extension with the unit
force along it and answers with a force that has a sideways part, turned in the actuation plane, so its
response, `Force <- Displacement`, does net work around a closed path. The adjugate of a map of
displacements, `motion.adjugate()`, carries forces back through it with their work kept, from bonds to
the deformations that move them and to the particles at their ends.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import VGA2D
from numga.sparse import SparseExtensor, spdiag
from examples import instantiate

ga = VGA2D
ctx = NumpyContext(ga)
mv = ctx.multivector
Mesh = instantiate("examples.util.mesh", ga).Mesh
Scalar = ga.gatype.scalar()                                                  # Scalar+
# Positions and displacements are antivectors; a force is a vector, read by its pairing with a displacement.
Displacement = ga.gatype.antivector()                                        # Antivector-
Force = ga.gatype.vector()                                                   # Vector+
Bivector = ga.gatype.bivector()                                              # Bivector+
# A response maps displacements to forces.
Response = ga.gatype((Force, Displacement))
# The two amplitudes of a shear, geometric like displacements.
Shear = ga.gatype.antivector()                                               # Antivector-

# The unit displacements along the two axes, which the unit forces `mv.x` and `mv.y` read as one.
directions = stack((mv.x, mv.y)).dual()                                      # [2] Displacement-


# --- plumbing -------------------------------------------------------------------------
def triangular_patch(n_columns: int, n_rows: int, spacing: float) -> tuple[Displacement, np.ndarray, SparseExtensor]:
    """A triangular lattice with free edges, its positions and oriented bond incidence."""
    columns, rows = np.meshgrid(np.arange(n_columns), np.arange(n_rows))
    positions = spacing * (
        directions[0] * (columns + (rows % 2) / 2 - (n_columns - 1) / 2)
        + directions[1] * ((rows - (n_rows - 1) / 2) * np.sqrt(3) / 2)
    )                                                                        # [rows, columns] Displacement-
    positions = positions.reshape(-1).field()                                # Displacement-[sites]

    indices = np.arange(n_columns * n_rows).reshape(n_rows, n_columns)
    horizontal = np.stack((indices[:, :-1], indices[:, 1:]), axis=-1).reshape(-1, 2)
    vertical = np.stack((indices[:-1], indices[1:]), axis=-1).reshape(-1, 2)
    lower = indices[:-1, :-1] + ((np.arange(n_rows - 1) + 1) % 2)[:, None]
    upper = indices[1:, :-1] + (np.arange(n_rows - 1) % 2)[:, None]
    diagonal = np.stack((lower, upper), axis=-1).reshape(-1, 2)
    edges = np.concatenate((horizontal, vertical, diagonal))
    incidence = Mesh(n_columns * n_rows, edges).d[0]                         # Scalar+[bonds <- sites]
    return positions, edges, incidence


# --- math -----------------------------------------------------------------------------
def bond_response(direction: Displacement, actuation_plane: Bivector, stiffness: float, activity: float) -> Response:
    """The restoring and sideways force caused by a bond's endpoint displacement."""
    # The unit force along the bond, which reads a displacement's component along it.
    unit_force = direction.dual_inverse()                                    # Force+
    push = -stiffness * unit_force + activity * (unit_force | actuation_plane)   # Force+
    return push * (unit_force & Displacement)                                # Force+ <- Displacement-


def integrate_cycle(response: Response, cycle: Displacement) -> Iterator[tuple[Displacement, Force, Scalar]]:
    """Displacement, force and work delivered along consecutive straight path segments."""
    work = mv.scalar([0])                                                    # Scalar+
    yield cycle[0], response(cycle[0]), work
    for start, end in zip(cycle[:-1], cycle[1:]):
        # A linear force's work along a straight segment is its midpoint force
        # paired with the segment's displacement.
        work = work + (response((start + end) / 2) & (end - start))          # Scalar+
        yield end, response(end), work


def evolution(incidence: SparseExtensor, response: Response, drags: Scalar) -> SparseExtensor:
    """The particles' velocities from their displacements, overdamped: no inertia, every force met by drag."""
    gradient = incidence * Displacement                                      # (Displacement- <- Displacement-)[bonds <- sites]
    # Drag turns each particle's force into its velocity, along the force's metric direction.
    mobility = spdiag(Force.dual() / drags)                                  # (Velocity- <- Force+)[sites <- sites]
    # Relative endpoint motion goes to each bond; equal opposite forces return to its endpoints.
    return mobility(gradient.adjugate()(spdiag(response)(gradient)))         # (Velocity- <- Displacement-)[sites <- sites]


def rk4_step(generator: SparseExtensor, displacement: Displacement, dt: float) -> Displacement:
    """One fourth-order Runge–Kutta step of an overdamped displacement field."""
    first_rate = generator(displacement)                                     # Velocity-[sites]
    second_rate = generator(displacement + first_rate * (dt / 2))            # Velocity-[sites]
    third_rate = generator(displacement + second_rate * (dt / 2))            # Velocity-[sites]
    fourth_rate = generator(displacement + third_rate * dt)                  # Velocity-[sites]
    return displacement + (first_rate + 2 * second_rate + 2 * third_rate + fourth_rate) * (dt / 6)


def wave_states(
    reference: Displacement, incidence: SparseExtensor, bond_directions: Displacement, generator: SparseExtensor,
    initial: Displacement, n_frames: int, n_steps_per_frame: int, dt: float,
) -> Iterator[tuple[Displacement, Scalar]]:
    """The moving lattice and bond extensions during its free overdamped response, `n_steps_per_frame` steps per
    frame."""
    displacement = initial                                                   # Displacement-[sites]
    for _ in range(n_frames):
        extensions = bond_directions.dual_inverse() & (incidence * displacement)   # Scalar+[bonds]
        yield reference + displacement, extensions
        for _ in range(n_steps_per_frame):
            displacement = rk4_step(generator, displacement, dt)             # Displacement-[sites]
