"""Modal extended position-based dynamics: flexible bodies joined by point constraints, in the projective
geometric algebra of the plane.

Each body is a girder of bars. Its rigid motion is a motor and its deformation a combination of its
lowest vibration modes, the eigenfields of its sparse stiffness against its masses. Bodies are
connected by point constraints between anchor points. Time stepping is modal extended position-based
dynamics: rigid bodies under compliant constraints, each mode one more constraint. For an anchor,
`anchor.commutator(Twist)` maps a twist of its body to the anchor's displacement,
`Displacement <- Twist`; over all constraints and bodies these maps form one map, and its
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
from examples import instantiate
from examples.mechanics import lie_integrators as lie


# Supplied by examples.instantiate: the plane's algebra, PGA2D, and the context to compute in.
ga: Algebra
ctx: Context
mv = ctx.multivector
# A girder's bars and the constraints joining bodies are bond graphs, one-dimensional meshes in the same context.
Mesh = instantiate("examples.util.mesh", ga, type(ctx)).Mesh
Scalar = ga.gatype.scalar()
Point = ga.gatype.antivector()
Twist = ga.gatype.bivector()
Forque = ga.gatype.antibivector()
Motor = ga.gatype.rotor()
# A displacement is a point at infinity, an antivector through the null direction w; a force is a line through the
# origin, a sum of x and y, and a force joined with a displacement, `force & displacement`, is the work done moving
# that way.
Displacement = ga.gatype.antivector().derive.degenerate                    # Antivector-
Force = ga.gatype.vector().derive.nondegenerate                            # Vector+
Inertia = ga.gatype((Forque, Twist))                                       # Forque- <- Twist+
InverseInertia = ga.gatype((Twist, Forque))                                # Twist+ <- Forque-


# --- plumbing -------------------------------------------------------------------------
@dataclass(frozen=True)
class Shape:
    """A girder at rest and its retained vibrations."""
    # Centred on its centre of mass.
    rest: Point                                                             # Point-[vertices]
    modes: Displacement                                                        # [modes] Displacement-[vertices]
    # Angular frequencies.
    frequencies: Scalar                                                     # [modes] Scalar+
    # Inverse squared frequencies.
    compliance: Scalar                                                      # [modes] Scalar+
    masses: Scalar                                                          # Scalar+[vertices]
    edges: np.ndarray                                                       # [edges, ends]
    inertia: Inertia                                                        # [] Forque- <- Twist+


@dataclass(frozen=True)
class Bodies:
    """Bodies in motion, batched over cases."""
    motor: Motor                                                            # [...] Motor+[bodies]
    # In each body's frame.
    rate: Twist                                                             # [...] Twist+[bodies]
    amplitudes: Scalar                                                      # [..., modes] Scalar+[bodies]
    rates: Scalar                                                           # [..., modes] Scalar+[bodies]
    compliance: Scalar                                                      # [..., modes] Scalar+[bodies]
    frequencies: Scalar                                                     # [..., modes] Scalar+[bodies]
    # Damping ratios.
    damping: Scalar                                                         # [..., modes] Scalar+[bodies]
    masses: Scalar                                                          # [...] Scalar+[bodies]
    inertia: Inertia                                                        # [...] (Forque- <- Twist+)[bodies]
    # Zero for a fixed body.
    inverse_inertia: InverseInertia                                         # [...] (Twist+ <- Forque-)[bodies]


@dataclass(frozen=True)
class Constraints:
    """Point constraints between anchor points of two bodies."""
    body_idx: np.ndarray                                                    # [sides, constraints]
    # The body at each side.
    ends: SparseExtensor                                                    # [sides] Scalar+[constraints <- bodies]
    anchors: Point                                                          # [sides] Point-[constraints]
    modes: Displacement                                                        # [modes, sides] Displacement-[constraints]
    compliance: Scalar                                                      # Scalar+[constraints]


def truss(cells: int, length: float, height: float) -> tuple[Point, np.ndarray]:
    """The girder's rest points and bars."""
    stations = np.arange(cells + 1)
    levels = np.array([-0.5, 0.5])
    # The origin moved along x to each station and along y to each level.
    positions = (mv.w.dual() + mv.x.dual() * (stations[:, None] * length / cells) + mv.y.dual() * (levels[None, :] * height)).reshape(-1).field()  # Point-[vertices]
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
def girder(cells: int, length: float, height: float, stiffness: float, density: float, n_modes: int) -> Shape:
    """A cross-braced girder, reduced to its lowest vibration modes."""
    positions, edges = truss(cells, length, height)                         # Point-[vertices], [edges, ends]
    vertices = positions.batch().shape[-1]
    # Each bar's tail taken from its head: the bar as a displacement, its two points' weights cancelling.
    ends = Mesh(vertices, edges).d[0]                                       # Scalar+[edges <- vertices]
    difference = (ends * positions).cast(Displacement)                      # Displacement-[edges]
    # The metric enters once, turning each bar into the force along it; that force paired with the bar is its
    # squared length.
    pulls = difference.dual_inverse()                                       # Force+[edges]
    lengths = (pulls & difference).square_root()                            # Scalar+[edges]
    unit_forces = pulls / lengths                                           # Force+[edges]
    # A bar reads its stretch by pairing its unit force with the relative displacement of its ends, and pulls back
    # along itself.
    bars = spdiag(unit_forces * (unit_forces & Displacement) * stiffness / lengths)  # (Force+ <- Displacement-)[edges <- edges]
    # The relative displacements of the bars' ends.
    stretching = ends * Displacement                                        # (Displacement- <- Displacement-)[edges <- vertices]
    masses = mv.scalar(np.full((vertices, 1), density * length * height / vertices)).field()  # Scalar+[vertices]
    # The masses as maps from accelerations to the forces that cause them.
    weights = spdiag(masses * Displacement.dual_inverse())                  # (Force+ <- Displacement-)[vertices <- vertices]
    # The girder's stiffness: the bars' forces, at the relative displacements of their ends, carried back to the vertices.
    girder_stiffness = stretching.adjugate()(bars(stretching))              # (Force+ <- Displacement-)[vertices <- vertices]
    # The stiffness against the masses; the three zero modes are rigid motions, carried by the motor.
    values, modes = girder_stiffness.eigh(weights, 3 + n_modes)             # [3 + modes] Scalar+, [3 + modes] Displacement-[vertices]
    values, modes = values[3:], modes[3:]                                   # [modes] Scalar+, [modes] Displacement-[vertices]
    centre = (positions * masses).sites.sum() / masses.sites.sum()          # [] Point-
    rest = positions - centre + mv.w.dual()                                 # Point-[vertices]
    inertia = ((rest & rest.commutator(Twist)) * masses).sites.sum()        # [] Forque- <- Twist+
    # The modes' angular frequencies and their compliances.
    frequencies = values.square_root()                                      # [modes] Scalar+
    compliance = 1 / values                                                 # [modes] Scalar+
    return Shape(rest=rest, modes=modes, frequencies=frequencies, compliance=compliance, masses=masses, edges=edges, inertia=inertia)


def points(bodies: Bodies, shape: Shape) -> Point:
    """The bodies' points in the world."""
    # Each body's rest points moved by its modes, in its own frame.
    local_points = shape.rest + (shape.modes[:, None] * bodies.amplitudes.batch()).sum(axis=-2)  # [..., bodies] Point-[vertices]
    return bodies.motor.batch() >> local_points                            # [..., bodies] Point-[vertices]


def modal_terms(bodies: Bodies, previous_amplitudes: Scalar, dt: float) -> tuple[Scalar, Scalar, Scalar]:
    """Each mode's compliance, residual and response to a unit load over the step."""
    # The implicit dashpot relative to the spring: twice the damping ratio times the frequency, times the
    # spring's compliance, over the step.
    damping = 2 * bodies.damping * bodies.frequencies * bodies.compliance / dt  # [..., modes] Scalar+[bodies]
    # The spring's compliance over the step squared, softened by the dashpot.
    compliance = bodies.compliance / dt**2 / (1 + damping)                 # [..., modes] Scalar+[bodies]
    # The mode's amplitude, with its change over the step weighted in by the dashpot.
    residual = (bodies.amplitudes + damping * (bodies.amplitudes - previous_amplitudes)) / (1 + damping)  # [..., modes] Scalar+[bodies]
    # A mode has unit mass, so a load on it moves it by 1 / (1 + compliance) of the load.
    response = 1 / (1 + compliance)                                         # [..., modes] Scalar+[bodies]
    return compliance, residual, response


def coupling(bodies: Bodies, constraints: Constraints) -> tuple[SparseExtensor, SparseExtensor, SparseExtensor, Displacement]:
    """The sparse maps from the bodies' rigid motion and modes to the constraint gaps, and the gaps."""
    body_count, constraint_count = bodies.motor.batch().shape[-1], constraints.body_idx.shape[-1]
    # The motors of the two bodies each constraint joins.
    motor = constraints.ends * bodies.motor[..., None]                     # [..., sides] Motor+[constraints]
    # The constraint's anchor on each of the two bodies, in that body's frame, moved by its modes.
    local_anchors = constraints.anchors + (constraints.modes * (constraints.ends * bodies.amplitudes[..., None])).sum(axis=-2)  # [..., sides] Point-[constraints]
    # A gap is the second anchor's position minus the first's.
    signs = np.array([-1, 1])
    # How each anchor moves for a twist of its body: its commutator with the space of twists, in the world frame.
    # The body's motion is a twist, extrinsic, and the anchor's displacement under it intrinsic, so a force at the
    # anchor is extrinsic: its work is `force & displacement`, the regressive product of the two kinds.
    anchor_motion = (motor >> local_anchors.commutator(Twist)) * signs     # [..., sides] (Displacement- <- Twist+)[constraints]
    # How each anchor moves for each mode amplitude of its body: the mode's shape at the anchor, in the world frame.
    anchor_modes = (motor[..., None, :] >> constraints.modes) * signs      # [..., modes, sides] Displacement-[constraints]
    # Each constraint's gap in the world frame, the second anchor's position minus the first's.
    gap = ((motor >> local_anchors) * signs).sum(axis=-1)                  # [...] Displacement-[constraints]
    # Each constraint is a row; each of its two bodies a column.
    constraint_idx = np.arange(constraint_count)                           # [constraints]
    # The change in each gap for the bodies' twists.
    rigid = SparseExtensor.from_indices(anchor_motion.batch(), constraint_idx, constraints.body_idx, (constraint_count, body_count))  # [...] (Displacement- <- Twist+)[constraints <- bodies]
    # The change in each gap for the bodies' amplitudes of each mode, one map per mode.
    modal = SparseExtensor.from_indices(anchor_modes.batch(), constraint_idx, constraints.body_idx, (constraint_count, body_count))  # [..., modes] Displacement-[constraints <- bodies]
    # The load on each mode of each body for forces at the constraints, one map per mode: each force paired with
    # the mode's displacement at its anchor, `force & displacement`, the work it does per unit of amplitude.
    modal_load = SparseExtensor.from_indices((Force & anchor_modes).batch(), constraints.body_idx, constraint_idx, (body_count, constraint_count))  # [..., modes] (Scalar+ <- Force+)[bodies <- constraints]
    return rigid, modal, modal_load, gap


def project(bodies: Bodies, constraints: Constraints, previous_amplitudes: Scalar, mode_reactions: Scalar, dt: float) -> tuple[Bodies, Scalar]:
    """The bodies displaced to satisfy all constraints, and the modes' reactions."""
    compliance, residual, response = modal_terms(bodies, previous_amplitudes, dt)  # [..., modes] Scalar+[bodies], Scalar+[bodies], Scalar+[bodies]
    rigid, modal, modal_load, gap = coupling(bodies, constraints)
    # The load of each mode's spring, less the reactions it has already received this step.
    spring_load = -(residual + compliance * mode_reactions)                # [..., modes] Scalar+[bodies]
    # How far each mode moves under its spring alone.
    unconstrained_step = response * spring_load                            # [..., modes] Scalar+[bodies]
    # How far each mode moves under a unit load from a constraint, its spring holding back the rest.
    mobility = spdiag(response * compliance)                               # [..., modes] Scalar+[bodies <- bodies]
    # Each constraint's own compliance over the step squared, as a map from force to gap.
    constraint_compliance = spdiag(constraints.compliance / dt**2 * Force.dual())  # (Displacement- <- Force+)[constraints <- constraints]
    # Each body's inverse inertia, from forque to twist.
    inverse_inertias = spdiag(bodies.inverse_inertia)                      # [...] (Twist+ <- Forque-)[bodies <- bodies]
    # The adjugate reads the work pairing backwards: `rigid.adjugate()(force) & twist == force & rigid(twist)`
    # for every force and twist, so it turns forces at the anchors, extrinsic, into the forques they exert on the
    # bodies, intrinsic. This is the principle of virtual work, with no transpose and no metric.
    # The change in the gaps for forces at the constraints: through the bodies' rigid motion, through
    # their modes, summed over the modes, and through the constraints' own compliance.
    system = rigid(inverse_inertias(rigid.adjugate())) + (modal * (mobility * modal_load)).sum(axis=-1) + constraint_compliance  # [...] (Displacement- <- Force+)[constraints <- constraints]
    # The reactions at the constraints that close every gap, given how far the modes move on their own.
    # They are position-level, force times the step squared, so through an inverse inertia they give a
    # displacement.
    reactions = system.solve((-gap - (modal * unconstrained_step).sum(axis=-1)).cast(Displacement))  # [...] Force+[constraints]
    # Each body's twist, from the forques the reactions exert on it: the forces carried back to the bodies by
    # the adjugate, then turned into motion by the inverse inertia.
    displacement = bodies.inverse_inertia(rigid.adjugate()(reactions))     # [...] Twist+[bodies]
    # Each mode's load from the reactions, and its spring's reaction to the whole.
    mode_loads = modal_load(reactions[..., None])                          # [..., modes] Scalar+[bodies]
    reaction_change = response * (spring_load - mode_loads)                # [..., modes] Scalar+[bodies]
    # The motors moved by the twists, the modes by their loads and the change in their reactions.
    constrained_bodies = replace(
        bodies,
        motor=bodies.motor * (displacement * -0.5).exp(),
        amplitudes=bodies.amplitudes + mode_loads + reaction_change,
    )
    # The reactions the modes have received this step.
    mode_reactions = mode_reactions + reaction_change                       # [..., modes] Scalar+[bodies]
    return constrained_bodies, mode_reactions


def step(bodies: Bodies, constraints: Constraints, dt: float, gravity: Displacement) -> Bodies:
    """The bodies one time step later."""
    previous = bodies

    def weight(motor: Motor, rate: Twist) -> Forque:
        # Gravity acts at each body's centre of mass, the origin of its frame.
        return (mv.w.dual() & (motor << gravity)) * bodies.masses            # [...] Forque-[bodies]

    # Predict the rigid motion from the rigid-body equations under gravity, and move the modes by their rates.
    motor, _ = lie.explicit_rk4(bodies.motor, bodies.rate, bodies.inertia, bodies.inverse_inertia, dt, weight)
    bodies = replace(bodies, motor=motor, amplitudes=bodies.amplitudes + bodies.rates * dt)
    # Relax each mode's spring on its own.
    _, residual, response = modal_terms(bodies, previous.amplitudes, dt)   # [..., modes] Scalar+[bodies], Scalar+[bodies], Scalar+[bodies]
    mode_reactions = -residual * response                                  # [..., modes] Scalar+[bodies]
    # Displace the bodies so that every constraint holds, the springs relaxing with them.
    bodies, _ = project(replace(bodies, amplitudes=bodies.amplitudes + mode_reactions), constraints, previous.amplitudes, mode_reactions, dt)
    # The rates are the change over the step, the motor's by the logarithm of its change.
    return replace(
        bodies,
        rate=(~previous.motor * bodies.motor).log() * (-2 / dt),
        rates=(bodies.amplitudes - previous.amplitudes) / dt,
    )
