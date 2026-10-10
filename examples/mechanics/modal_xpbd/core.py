"""Flexible bodies joined by point constraints, in the projective geometric algebra of the plane.

Each body is a girder of bars. Its rigid motion is a motor and its deformation a combination of its
lowest vibration modes, the eigenfields of its sparse stiffness against its masses. Bodies are
connected by point constraints between anchor points. Time stepping is modal extended position-based
dynamics: rigid bodies under compliant constraints, each mode one more constraint. For an anchor,
`anchor.commutator(Twist)` maps a twist of its body to the anchor's displacement,
`Direction <- Twist`; over all constraints and bodies these maps form a sparse extensor, and its
adjugate maps forces at the constraints to forques on the bodies. Composed through the inverse
inertia, with the modes' and constraints' compliance added, they give one sparse system for all
constraints, solved once per step. Solving the constraints one at a time converges too slowly for
stiff structures. Stiffness enters as compliance, so arbitrarily stiff modes and constraints stay
well conditioned, down to the rigid limit of zero compliance, and modes too fast for the step are
damped, not unstable.

Inertia is evaluated in the rest configuration and the coupling of rotation and vibration is
neglected, which assumes rotation slow compared with the retained vibrations.

In the notation of extended position-based dynamics the system reads as
$S = J_b M_b^{-1} J_b^\\top + J_q W J_q^\\top + \\tilde\\alpha_p$, with
$W = \\tilde\\alpha_m / (1 + \\tilde\\alpha_m)$.
"""

from dataclasses import dataclass, replace

import numpy as np

from numga import Algebra
from numga.backend.context import Context
from numga.sparse import SparseExtensor, spdiag
from examples.mechanics import lie_integrators as lie


# Supplied by examples.instantiate: the plane's algebra, PGA2D, and the context to compute in.
ga: Algebra
ctx: Context
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
    rest: Point                                                             # Point[vertices], centred on mass
    modes: Direction                                                        # [modes] Direction[vertices]
    frequencies: Scalar                                                     # [modes] Scalar, angular frequencies
    compliance: Scalar                                                      # [modes] Scalar, inverse squared frequencies
    masses: Scalar                                                          # Scalar[vertices]
    edges: np.ndarray                                                       # [edges, ends]
    inertia: Inertia                                                        # [] Forque <- Twist


@dataclass(frozen=True)
class Bodies:
    """Bodies in motion, batched over cases."""
    motor: Motor                                                            # [...] Motor[bodies]
    rate: Twist                                                             # [...] Twist[bodies], in each body's frame
    amplitudes: Scalar                                                      # [..., modes] Scalar[bodies]
    rates: Scalar                                                           # [..., modes] Scalar[bodies]
    compliance: Scalar                                                      # [..., modes] Scalar[bodies]
    frequencies: Scalar                                                     # [..., modes] Scalar[bodies]
    damping: Scalar                                                         # [..., modes] Scalar[bodies], damping ratio
    masses: Scalar                                                          # [...] Scalar[bodies]
    inertia: Inertia                                                        # [...] (Forque <- Twist)[bodies]
    inverse_inertia: InverseInertia                                         # [...] (Twist <- Forque)[bodies], zero for a fixed body


@dataclass(frozen=True)
class Constraints:
    """Point constraints between anchor points of two bodies."""
    body_idx: np.ndarray                                                    # [sides, constraints]
    ends: SparseExtensor                                                    # [sides] [constraints, bodies] Scalar, the body at each side
    anchors: Point                                                          # [sides] Point[constraints]
    modes: Direction                                                        # [modes, sides] Direction[constraints]
    compliance: Scalar                                                      # Scalar[constraints]


def truss(cells: int, length: float, height: float) -> tuple[Force, np.ndarray]:
    """The girder's rest points and bars."""
    stations = np.arange(cells + 1)
    levels = np.array([-0.5, 0.5])
    positions = (mv.x * (stations[:, None] * length / cells) + mv.y * (levels[None, :] * height)).reshape(-1).field()  # Force[vertices]
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


# --- math -----------------------------------------------------------------------------
# Leading ... axes hold independent cases.
def girder(cells: int, length: float, height: float, stiffness: float, density: float, modes: int) -> Shape:
    """A cross-braced girder, reduced to its lowest vibration modes."""
    positions, edges = truss(cells, length, height)                         # Force[vertices], [edges, ends]
    vertices = positions.batch().shape[-1]
    # Each bar's tail taken from its head.
    ends = SparseExtensor.from_columns(edges, mv.scalar((np.ones_like(edges) * [-1, 1])[..., None]), vertices)  # [edges, vertices] Scalar
    difference = ends * positions                                           # Force[edges]
    lengths = difference.norm()                                             # Scalar[edges]
    directions = difference / lengths                                       # Force[edges]
    # Each bar's stiffness acts along its direction.
    bars = spdiag(directions * (directions | Force) * stiffness / lengths)  # [edges, edges] Force <- Force
    masses = mv.scalar(np.full((vertices, 1), density * length * height / vertices)).field()  # Scalar[vertices]
    weights = spdiag(masses) * Force                                        # [vertices, vertices] Force <- Force
    # The stiffness against the masses; the three zero modes are rigid motions, carried by the motor.
    values, fields = (~ends * bars(ends * Force)).eigh(weights, 3 + modes)  # [3 + modes] Scalar, [3 + modes] Force[vertices]
    values, fields = values[3:], fields[3:]                                 # [modes] Scalar, [modes] Force[vertices]
    centre = (positions * masses).sites.sum() / masses.sites.sum()  # [] Force
    rest = (positions - centre + mv.w).dual()                               # Point[vertices]
    inertia = ((rest & rest.commutator(Twist)) * masses).sites.sum()  # [] Forque <- Twist
    # The modes as displacements, their angular frequencies, and their compliances.
    modes = fields.dual()                                                   # [modes] Direction[vertices]
    frequencies = values.square_root()                                      # [modes] Scalar
    compliance = 1 / values                                                 # [modes] Scalar
    return Shape(rest=rest, modes=modes, frequencies=frequencies, compliance=compliance, masses=masses, edges=edges, inertia=inertia)


def points(bodies: Bodies, shape: Shape) -> Point:
    """The bodies' points in the world."""
    # Each body's rest points moved by its modes, in its own frame.
    local_points = shape.rest + (shape.modes[:, None] * bodies.amplitudes.batch()).sum(axis=-2)  # [..., bodies] Point[vertices]
    return bodies.motor.batch() >> local_points                            # [..., bodies] Point[vertices]


def modal_terms(bodies: Bodies, previous_amplitudes: Scalar, dt: float) -> tuple[Scalar, Scalar, Scalar]:
    """Each mode's compliance, residual and response to a unit load over the step."""
    # The implicit dashpot relative to the spring: twice the damping ratio times the frequency, times the
    # spring's compliance, over the step.
    damping = 2 * bodies.damping * bodies.frequencies * bodies.compliance / dt  # [..., modes] Scalar[bodies]
    # The spring's compliance over the step squared, softened by the dashpot.
    compliance = bodies.compliance / dt**2 / (1 + damping)                 # [..., modes] Scalar[bodies]
    # The mode's amplitude, with its change over the step weighted in by the dashpot.
    residual = (bodies.amplitudes + damping * (bodies.amplitudes - previous_amplitudes)) / (1 + damping)  # [..., modes] Scalar[bodies]
    # A mode has unit mass, so a load on it moves it by 1 / (1 + compliance) of the load.
    response = 1 / (1 + compliance)                                         # [..., modes] Scalar[bodies]
    return compliance, residual, response


def coupling(bodies: Bodies, constraints: Constraints) -> tuple[SparseExtensor, SparseExtensor, SparseExtensor, Direction]:
    """The sparse maps from the bodies' rigid motion and modes to the constraint gaps, and the gaps."""
    body_count, constraint_count = bodies.motor.batch().shape[-1], constraints.body_idx.shape[-1]
    # The motors of the two bodies each constraint joins.
    motor = constraints.ends * bodies.motor[..., None]                     # [..., sides] Motor[constraints]
    # The constraint's anchor on each of the two bodies, in that body's frame, moved by its modes.
    local_anchors = constraints.anchors + (constraints.modes * (constraints.ends * bodies.amplitudes[..., None])).sum(axis=-2)  # [..., sides] Point[constraints]
    # A gap is the second anchor's position minus the first's.
    signs = np.array([-1, 1])
    # How each anchor moves for a twist of its body: the commutator with the open twist, in the world frame.
    anchor_motion = (motor >> local_anchors.commutator(Twist)) * signs     # [..., sides] (Direction <- Twist)[constraints]
    # How each anchor moves for each mode amplitude of its body: the mode's shape at the anchor, in the world frame.
    anchor_modes = (motor[..., None, :] >> constraints.modes) * signs      # [..., modes, sides] Direction[constraints]
    # Each constraint's gap in the world frame, the second anchor's position minus the first's.
    gap = ((motor >> local_anchors) * signs).sum(axis=-1)                  # [...] Direction[constraints]
    # Each constraint is a row; each of its two bodies a column.
    constraint_idx = np.arange(constraint_count)                           # [constraints]
    # The change in each gap for the bodies' twists.
    rigid = SparseExtensor.from_indices(anchor_motion.batch(), constraint_idx, constraints.body_idx, (constraint_count, body_count))  # [..., constraints, bodies] Direction <- Twist
    # The change in each gap for the bodies' amplitudes of each mode, one map per mode.
    modal = SparseExtensor.from_indices(anchor_modes.batch(), constraint_idx, constraints.body_idx, (constraint_count, body_count))  # [..., modes] [constraints, bodies] Direction
    # The load on each mode of each body for forces at the constraints, one map per mode.
    modal_load = SparseExtensor.from_indices((Force & anchor_modes).batch(), constraints.body_idx, constraint_idx, (body_count, constraint_count))  # [..., modes] [bodies, constraints] Scalar <- Force
    return rigid, modal, modal_load, gap


def project(bodies: Bodies, constraints: Constraints, previous_amplitudes: Scalar, mode_reactions: Scalar, dt: float) -> tuple[Bodies, Scalar]:
    """The bodies displaced to satisfy all constraints, and the modes' reactions."""
    compliance, residual, response = modal_terms(bodies, previous_amplitudes, dt)  # [..., modes] Scalar[bodies] each
    rigid, modal, modal_load, gap = coupling(bodies, constraints)
    # The load of each mode's spring, less the reactions it has already received this step.
    spring_load = -(residual + compliance * mode_reactions)                # [..., modes] Scalar[bodies]
    # How far each mode moves under its spring alone.
    unconstrained_step = response * spring_load                            # [..., modes] Scalar[bodies]
    # How far each mode moves under a unit load from a constraint, its spring holding back the rest.
    mobility = spdiag(response * compliance)                               # [..., modes] [bodies, bodies] Scalar
    # Each constraint's own compliance over the step squared, as a map from force to gap.
    constraint_compliance = spdiag(constraints.compliance / dt**2 * Force.dual())  # [constraints, constraints] Direction <- Force
    # Each body's inverse inertia, from forque to twist.
    inverse_inertias = spdiag(bodies.inverse_inertia)                      # [..., bodies, bodies] Twist <- Forque
    # The change in the gaps for forces at the constraints: through the bodies' rigid motion, through
    # their modes, summed over the modes, and through the constraints' own compliance.
    system = rigid(inverse_inertias(rigid.adjugate())) + (modal * (mobility * modal_load)).sum(axis=-1) + constraint_compliance  # [..., constraints, constraints] Direction <- Force
    # The reactions at the constraints that close every gap, given how far the modes move on their own.
    # They are position-level, force times the step squared, so through an inverse inertia they give a
    # displacement.
    reactions = system.solve((-gap - (modal * unconstrained_step).sum(axis=-1)).cast(Direction))  # [...] Force[constraints]
    # Each body's twist, from the forques the reactions exert on it.
    displacement = bodies.inverse_inertia(rigid.adjugate()(reactions))     # [...] Twist[bodies]
    # Each mode's load from the reactions, and its spring's reaction to the whole.
    mode_loads = modal_load(reactions[..., None])                          # [..., modes] Scalar[bodies]
    reaction_change = response * (spring_load - mode_loads)                # [..., modes] Scalar[bodies]
    # The motors moved by the twists, the modes by their loads and the change in their reactions.
    moved = replace(
        bodies,
        motor=bodies.motor * (displacement * -0.5).exp(),
        amplitudes=bodies.amplitudes + mode_loads + reaction_change,
    )
    # The reactions the modes have received this step.
    mode_reactions = mode_reactions + reaction_change                       # [..., modes] Scalar[bodies]
    return moved, mode_reactions


def step(bodies: Bodies, constraints: Constraints, dt: float, gravity: Direction) -> Bodies:
    """One time step."""
    previous = bodies

    def weight(motor: Motor, rate: Twist) -> Forque:
        # Gravity acts at each body's centre of mass, the origin of its frame.
        return (mv.w.dual() & (motor << gravity)) * bodies.masses            # [...] Forque[bodies]

    # Predict the rigid motion from the rigid-body equations under gravity, and move the modes by their rates.
    motor, _ = lie.explicit_rk4(bodies.motor, bodies.rate, bodies.inertia, bodies.inverse_inertia, dt, weight)
    bodies = replace(bodies, motor=motor, amplitudes=bodies.amplitudes + bodies.rates * dt)
    # Relax each mode's spring on its own.
    _, residual, response = modal_terms(bodies, previous.amplitudes, dt)   # [..., modes] Scalar[bodies] each
    mode_reactions = -residual * response                                  # [..., modes] Scalar[bodies]
    # Displace the bodies so that every constraint holds, the springs relaxing with them.
    bodies, _ = project(replace(bodies, amplitudes=bodies.amplitudes + mode_reactions), constraints, previous.amplitudes, mode_reactions, dt)
    # The rates are the change over the step, the motor's by the logarithm of its change.
    return replace(
        bodies,
        rate=(~previous.motor * bodies.motor).log() * (-2 / dt),
        rates=(bodies.amplitudes - previous.amplitudes) / dt,
    )
