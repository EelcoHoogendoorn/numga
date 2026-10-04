"""Flexible bodies joined by point constraints, in the projective geometric algebra of the plane.

Each body is a girder of bars. Its rigid motion is a motor and its deformation a combination of its
lowest vibration modes, the eigenfields of its sparse stiffness against its masses. Bodies are
connected by point constraints between anchor points. Time stepping is extended
position-based dynamics. For an anchor, `anchor.commutator(Twist)` maps a twist of its body to the
anchor's displacement, `Direction <- Twist`; over all constraints and bodies these maps form a sparse
extensor, and its adjugate maps forces at the constraints to forques on the bodies. Composed through the
inverse inertia, with the modes' and constraints' compliance added, they give one sparse system for all
constraints, solved once per step. Solving the constraints one at a time converges too slowly for stiff structures.

Inertia is evaluated in the rest configuration and the coupling of rotation and vibration is
neglected, which assumes rotation slow compared with the retained vibrations.

In the notation of extended position-based dynamics the system reads as
$S = J_b M_b^{-1} J_b^\\top + J_q W J_q^\\top + \\tilde\\alpha_p$, with
$W = \\tilde\\alpha_m / (1 + \\tilde\\alpha_m)$.
"""

from dataclasses import dataclass, replace

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA2D
from numga.sparse import SparseExtensor, spdiag
from examples.mechanics import lie_integrators as lie


ga = PGA2D
ctx = NumpyContext(ga)
mv = ctx.multivector
Scalar = ga.gatype.scalar()
Point = ga.gatype.antivector()
Twist = ga.gatype.bivector()
Forque = ga.gatype.antibivector()
Motor = ga.gatype.rotor()
# The lines x and y through the origin, the polars of the directions: a displacement or a force in
# the plane is one of them, and joined with a direction, `force & direction`, it gives the work done
# moving that way.
Force = ga.gatype(ga.subspace("x y"))
Direction = ga.gatype(Force.output_subspace.complement())
Inertia = ga.gatype((Forque, Twist))                                       # Forque <- Twist
InverseInertia = ga.gatype((Twist, Forque))                                # Twist <- Forque


# --- plumbing -------------------------------------------------------------------------
@dataclass(frozen=True)
class Shape:
    """A girder at rest and its retained vibrations."""
    rest: Point                    # [vertices] Point, centred on mass
    modes: Direction               # [modes, vertices] Direction
    frequencies: Scalar            # [modes] Scalar, angular frequencies
    compliance: Scalar             # [modes] Scalar, inverse squared frequencies
    masses: np.ndarray             # [vertices]
    edges: np.ndarray              # [edges, ends]
    inertia: Inertia               # [] Forque <- Twist


@dataclass(frozen=True)
class Bodies:
    """Bodies in motion, batched over cases."""
    motor: Motor                   # [..., bodies] Motor
    rate: Twist                    # [..., bodies] Twist, in each body's frame
    amplitudes: Scalar             # [..., bodies, modes] Scalar
    rates: Scalar                  # [..., bodies, modes] Scalar
    compliance: Scalar             # [..., bodies, modes] Scalar
    frequencies: Scalar            # [..., bodies, modes] Scalar
    damping: np.ndarray            # [..., bodies, modes], damping ratio
    masses: np.ndarray             # [..., bodies]
    inertia: Inertia               # [..., bodies] Forque <- Twist
    inverse_inertia: InverseInertia  # [..., bodies] Twist <- Forque, zero for a fixed body


@dataclass(frozen=True)
class Constraints:
    """Point constraints between anchor points of two bodies."""
    bodies: np.ndarray             # [constraints, ends]
    anchors: Point                 # [constraints, ends] Point
    modes: Direction               # [constraints, ends, modes] Direction
    compliance: Scalar             # [constraints] Scalar


def truss(cells: int, length: float, height: float) -> tuple[Force, np.ndarray]:
    """The girder's rest points and bars."""
    stations = np.arange(cells + 1)
    levels = np.array([-0.5, 0.5])
    positions = (mv.x * (stations[:, None] * length / cells) + mv.y * (levels[None, :] * height)).reshape(-1)  # [vertices] Force
    lower = 2 * stations[:-1]
    # Longitudinal bars, both diagonals, and one upright at every station.
    edges = np.concatenate([
        np.stack([lower, lower + 2], axis=-1),
        np.stack([lower + 1, lower + 3], axis=-1),
        np.stack([lower, lower + 3], axis=-1),
        np.stack([lower + 1, lower + 2], axis=-1),
        np.stack([2 * stations, 2 * stations + 1], axis=-1),
    ])
    return positions, edges


def incidence(cells, rows: np.ndarray, columns: np.ndarray, shape: tuple[int, int]) -> SparseExtensor:
    """Cells as a sparse extensor, at the given rows and columns."""
    return SparseExtensor(cells.reshape(cells.shape[:cells.ndim - rows.ndim] + (rows.size,)), rows.reshape(-1), columns.reshape(-1), shape)


# --- math -----------------------------------------------------------------------------
# Leading ... axes hold independent cases.
def girder(cells: int, length: float, height: float, stiffness: float, density: float, modes: int) -> Shape:
    """A cross-braced girder, reduced to its lowest vibration modes."""
    positions, edges = truss(cells, length, height)                       # [vertices] Force, [edges, ends]
    vertices = positions.shape[0]
    # Each bar's tail taken from its head.
    ends = SparseExtensor.from_columns(edges, mv.scalar((np.ones_like(edges) * [-1, 1])[..., None]), vertices)  # [edges, vertices] Scalar
    difference = ends * positions                                         # [edges] Force
    lengths = difference.norm()                                           # [edges] Scalar
    directions = difference / lengths                                     # [edges] Force
    # Each bar's stiffness acts along its direction.
    bars = spdiag(directions * (directions | Force) * stiffness / lengths)  # [edges, edges] Force <- Force
    masses = np.full(vertices, density * length * height / vertices)
    weights = spdiag(mv.scalar(masses[:, None])) * Force                    # [vertices, vertices] Force <- Force
    # The stiffness against the masses; the three zero modes are rigid motions, carried by the motor.
    values, fields = (~ends * bars(ends * Force)).eigh(weights, 3 + modes)  # [3 + modes] Scalar, [3 + modes, vertices] Force
    values, fields = values[3:], fields[3:]                               # [modes] Scalar, [modes, vertices] Force
    centre = (positions * masses).sum(axis=0) / masses.sum()              # [] Force
    rest = (positions - centre + mv.w).dual()                             # [vertices] Point
    inertia = ((rest & rest.commutator(Twist)) * masses).sum(axis=0)      # [] Forque <- Twist
    return Shape(rest, fields.dual(), values.square_root(), 1 / values, masses, edges, inertia)


def points(bodies: Bodies, shape: Shape) -> Point:
    """The bodies' points in the world."""
    local = shape.rest + (shape.modes * bodies.amplitudes[..., None]).sum(axis=-2)  # [..., bodies, vertices] Point
    return bodies.motor[..., None] >> local                               # [..., bodies, vertices] Point


def modal_terms(bodies: Bodies, previous: Scalar, dt: float) -> tuple[Scalar, Scalar, Scalar]:
    """Each mode's compliance, residual and response to a unit load over the step."""
    damping = 2 * bodies.damping * bodies.frequencies * bodies.compliance / dt  # [..., bodies, modes] Scalar
    compliance = bodies.compliance / dt**2 / (1 + damping)                # [..., bodies, modes] Scalar
    residual = (bodies.amplitudes + damping * (bodies.amplitudes - previous)) / (1 + damping)  # [..., bodies, modes] Scalar
    return compliance, residual, 1 / (1 + compliance)


def coupling(bodies: Bodies, constraints: Constraints) -> tuple[SparseExtensor, SparseExtensor, SparseExtensor, Direction]:
    """The constraints' sparse maps and gaps."""
    count, modes = bodies.motor.shape[-1], bodies.amplitudes.shape[-1]
    motor = bodies.motor[..., constraints.bodies]                                # [..., constraints, ends] Motor
    local = constraints.anchors + (constraints.modes * bodies.amplitudes[..., constraints.bodies, :]).sum(axis=-1)  # [..., constraints, ends] Point
    signs = np.array([-1, 1])
    # The displacement of each anchor for an open twist, second end minus first.
    motion = (motor >> local.commutator(Twist)) * signs                   # [..., constraints, ends] Direction <- Twist
    shapes = (motor[..., None] >> constraints.modes) * signs[:, None]            # [..., constraints, ends, modes] Direction
    gap = ((motor >> local) * signs).sum(axis=-1)                         # [..., constraints] Direction
    constraint = np.broadcast_to(np.arange(len(constraints.bodies))[:, None], constraints.bodies.shape)  # [constraints, ends]
    mode = constraints.bodies[..., None] * modes + np.arange(modes)              # [constraints, ends, modes]
    constraint_mode = np.broadcast_to(constraint[..., None], mode.shape)                # [constraints, ends, modes]
    return (
        incidence(motion, constraint, constraints.bodies, (len(constraints.bodies), count)),
        incidence(shapes, constraint_mode, mode, (len(constraints.bodies), count * modes)),
        incidence(Force & shapes, mode, constraint_mode, (count * modes, len(constraints.bodies))),
        gap,
    )


def project(bodies: Bodies, constraints: Constraints, previous: Scalar, lambdas: Scalar, dt: float) -> tuple[Bodies, Scalar]:
    """The bodies displaced to satisfy all constraints, and the modes' reactions."""
    compliance, residual, response = modal_terms(bodies, previous, dt)     # [..., bodies, modes] Scalar each
    motion, shapes, loads, gap = coupling(bodies, constraints)
    batch = bodies.amplitudes.shape[:-2]
    mode_load = -(residual + compliance * lambdas)                        # [..., bodies, modes] Scalar
    free_step = (response * mode_load).reshape(batch + (-1,))             # [..., bodies * modes] Scalar
    mobility = spdiag((response * compliance).reshape(batch + (-1,)))        # [..., bodies * modes, bodies * modes] Scalar
    constraint_compliance = spdiag(constraints.compliance / dt**2 * Force.dual())       # [constraints, constraints] Direction <- Force
    inertias = spdiag(bodies.inverse_inertia)                               # [..., bodies, bodies] Twist <- Forque
    system = motion(inertias(motion.adjugate())) + shapes * (mobility * loads) + constraint_compliance  # [..., constraints, constraints] Direction <- Force
    forces = system.solve((-gap - shapes * free_step).cast(Direction))    # [..., constraints] Force
    displacement = bodies.inverse_inertia(motion.adjugate()(forces))      # [..., bodies] Twist
    mode_forces = loads(forces).reshape(bodies.amplitudes.shape)          # [..., bodies, modes] Scalar
    modal_reaction = response * (mode_load - mode_forces)                  # [..., bodies, modes] Scalar
    moved = replace(
        bodies,
        motor=bodies.motor * (displacement * -0.5).exp(),
        amplitudes=bodies.amplitudes + mode_forces + modal_reaction,
    )
    return moved, lambdas + modal_reaction


def step(bodies: Bodies, constraints: Constraints, dt: float, gravity: Direction) -> Bodies:
    """One time step."""
    previous = bodies

    def weight(motor: Motor, rate: Twist) -> Forque:
        # Gravity acts at each body's centre of mass, the origin of its frame.
        return (mv.w.dual() & (motor << gravity)) * bodies.masses          # [..., bodies] Forque

    motor, _ = lie.explicit_rk4(bodies.motor, bodies.rate, bodies.inertia, bodies.inverse_inertia, dt, weight)
    bodies = replace(bodies, motor=motor, amplitudes=bodies.amplitudes + bodies.rates * dt)
    _, residual, response = modal_terms(bodies, previous.amplitudes, dt)  # [..., bodies, modes] Scalar each
    lambdas = -residual * response                                         # [..., bodies, modes] Scalar
    bodies, _ = project(replace(bodies, amplitudes=bodies.amplitudes + lambdas), constraints, previous.amplitudes, lambdas, dt)
    return replace(
        bodies,
        rate=(~previous.motor * bodies.motor).log() * (-2 / dt),
        rates=(bodies.amplitudes - previous.amplitudes) / dt,
    )
