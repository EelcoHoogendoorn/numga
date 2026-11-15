"""Modal extended position-based dynamics steps flexible girders joined by point constraints in the
projective geometric algebra of space.

Each girder moves rigidly through a motor while deforming along its lowest vibration modes, computed
as the eigenfields of its bar stiffness against its vertex masses. Point constraints link anchor
points across girders. Within each time step, modal extended position-based dynamics treats every
retained vibration mode as an additional compliant constraint on a rigid body. At each anchor, the
commutator `anchor.commutator(Twist)` maps a body twist to the anchor's displacement
(`Displacement- <- Twist+`), and the adjugate of the assembled joint map carries constraint forces
back to body forques. Composing through the inverse inertia and adding the modal and joint
compliances forms one sparse system over the joint forces alone, solved once per step. Because
stiffness enters through compliance, stiff modes and rigid links (`0` compliance) remain well
conditioned in the same solver, and modes faster than the step damp smoothly rather than diverging.

Inertia is evaluated in the rest configuration and gyroscopic coupling between rotation and
vibration is omitted, assuming body rotation stays slow relative to the retained vibration
frequencies.
"""

from dataclasses import replace

import numpy as np

from numga import Algebra
from numga.backend.context import Context
from numga.sparse import SparseExtensor, spdiag
from numga.topology import Topology
from examples.mechanics import lie_integrators as lie
from examples.mechanics.modal_xpbd.scenarios import Bodies, Constraints, Shape, truss


# Supplied by `examples.instantiate`: the algebra (`PGA2D`) and backend context.
ga: Algebra
ctx: Context
mv = ctx.multivector
Scalar = ga.gatype.scalar()                                                # Scalar+
Point = ga.gatype.antivector()                                             # Antivector-
Twist = ga.gatype.bivector()                                               # Bivector+
Forque = ga.gatype.antibivector()                                          # Antibivector-
Motor = ga.gatype.rotor()                                                  # Rotor+
# A displacement is an ideal point through the null plane w, while a force is a plane through the
# origin; pairing `force & displacement` evaluates the mechanical work along that displacement.
Displacement = ga.gatype.antivector().derive.degenerate                    # Antivector-
Force = ga.gatype.vector().derive.nondegenerate                            # Vector+


# --- math -----------------------------------------------------------------------------
# Leading `...` axes batch independent parameter cases.
def girder(n_bays: int, length: float, height: float, stiffness: float, density: float, n_modes: int) -> Shape:
    """Bar stiffness solved against lumped vertex masses reduces a girder of `n_bays` bays to its
    `n_modes` lowest non-rigid vibration modes."""
    positions, edges = truss(mv, n_bays, length, height)                   # Antivector-[V], [E, 2]
    topology = Topology(positions.batch().shape[-1], edges)
    V, E = topology.n_elements
    # Subtracting each bar's tail from its head cancels unit point weights into an ideal displacement.
    ends = topology.d[0]                                                   # Scalar+[E <- V]
    bar_displacements = (ends * positions).cast(Displacement)              # Antivector-[E]
    # Dualizing once with the Euclidean metric turns each bar displacement into the force plane along
    # that bar, whose pairing with the displacement gives the squared bar length.
    pulls = bar_displacements.dual_inverse()                               # Vector+[E]
    lengths = (pulls & bar_displacements).square_root()                    # Scalar+[E]
    unit_forces = pulls / lengths                                          # Vector+[E]
    # Each bar measures axial stretch by pairing its unit force with relative end displacement and
    # pulls back along the same line of action.
    bars = unit_forces * (unit_forces & Displacement) * stiffness / lengths  # (Vector+ <- Antivector-)[E]
    stretching = ends * Displacement                                       # (Antivector- <- Antivector-)[E <- V]
    masses = mv.scalar(np.full((V, 1), density * length * height / V)).field()  # Scalar+[V]
    weights = spdiag(masses * Displacement.dual_inverse())                 # (Vector+ <- Antivector-)[V <- V]
    # Pull bar stiffnesses back to the vertices through the coboundary adjugate.
    girder_stiffness = stretching.adjugate()(bars(stretching))             # (Vector+ <- Antivector-)[V <- V]
    # Drop the three zero-frequency planar rigid motions carried by the body's motor.
    squared_frequencies, modes = girder_stiffness.eigh(weights, 3 + n_modes)  # [3 + n_modes] Scalar+, [3 + n_modes] Antivector-[V]
    squared_frequencies, modes = squared_frequencies[3:], modes[3:]        # [n_modes] Scalar+, [n_modes] Antivector-[V]
    centre = (positions * masses).sites.sum() / masses.sites.sum()         # [] Antivector-
    rest = positions - centre + mv.w.dual()                                # Antivector-[V]
    inertia = (masses * rest).regressive(rest.commutator(Twist)).sites.sum()  # [] Antibivector- <- Bivector+
    compliance = 1 / squared_frequencies                                   # [n_modes] Scalar+
    return Shape(rest=rest, modes=modes, compliance=compliance, masses=masses, edges=edges, inertia=inertia)


def world_points(bodies: Bodies, shape: Shape) -> Point:
    """Modal superposition in each body frame followed by the body's motor places all girder
    vertices in the world frame (`[..., n_bodies] Antivector-[V]`)."""
    local_points = shape.rest + (shape.modes[:, None] * bodies.amplitudes.batch()).sum(axis=-2)  # [..., n_bodies] Antivector-[V]
    return bodies.motor.batch() >> local_points                            # [..., n_bodies] Antivector-[V]


def modal_terms(bodies: Bodies, previous_amplitudes: Scalar, dt: float) -> tuple[Scalar, Scalar, Scalar]:
    """Implicit spring and dashpot weights over step `dt` determine each mode's effective
    compliance, predictor residual, and unit-load response (`[..., n_modes] Scalar+[n_bodies]`)."""
    # Unit modal mass gives omega * alpha == sqrt(alpha), vanishing cleanly on rigid bodies (alpha == 0).
    damping = 2 * bodies.damping * bodies.compliance.square_root() / dt    # [..., n_modes] Scalar+[n_bodies]
    compliance = bodies.compliance / dt**2 / (1 + damping)                 # [..., n_modes] Scalar+[n_bodies]
    residual = (bodies.amplitudes + damping * (bodies.amplitudes - previous_amplitudes)) / (1 + damping)  # [..., n_modes] Scalar+[n_bodies]
    response = 1 / (1 + compliance)                                        # [..., n_modes] Scalar+[n_bodies]
    return compliance, residual, response


def coupling(bodies: Bodies, constraints: Constraints) -> tuple[SparseExtensor, SparseExtensor, SparseExtensor, Displacement]:
    """Anchor kinematics and constraint incidence assemble the sparse rigid and modal gap maps, the
    modal load pullback, and the current world-frame gaps."""
    motor = constraints.ends * bodies.motor[..., None]                     # [..., 2] Rotor+[n_constraints]
    local_anchors = constraints.anchors + (constraints.modes * (constraints.ends * bodies.amplitudes[..., None])).sum(axis=-2)  # [..., 2] Antivector-[n_constraints]
    signs = np.array([-1, 1])                                              # [2]
    # Commuting each local anchor with `Twist` maps body-frame twists to intrinsic anchor
    # displacements, which `motor` sandwiches into the world frame.
    anchor_motion = (motor >> local_anchors.commutator(Twist)) * signs     # [..., 2] (Antivector- <- Bivector+)[n_constraints]
    anchor_modes = (motor[..., None, :] >> constraints.modes) * signs      # [..., n_modes, 2] Antivector-[n_constraints]
    gap = ((motor >> local_anchors) * signs).sum(axis=-1)                  # [...] Antivector-[n_constraints]
    rigid = (spdiag(anchor_motion) * constraints.ends).sum(axis=-1)        # [...] (Antivector- <- Bivector+)[n_constraints <- n_bodies]
    modal = (spdiag(anchor_modes) * constraints.ends).sum(axis=-1)         # [..., n_modes] Antivector-[n_constraints <- n_bodies]
    # Pairing extrinsic constraint forces with intrinsic modal anchor displacements measures virtual
    # work per unit modal amplitude and pulls it back across `ends.adjugate()`.
    modal_load = (constraints.ends.adjugate() * spdiag(Force & anchor_modes)).sum(axis=-1)  # [..., n_modes] (Scalar+ <- Vector+)[n_bodies <- n_constraints]
    return rigid, modal, modal_load, gap


def project_constraints(bodies: Bodies, constraints: Constraints, previous_amplitudes: Scalar, mode_reactions: Scalar, dt: float) -> tuple[Bodies, Scalar]:
    """Solving the joint Schur complement closes all constraint gaps at once while updating body
    motors, modal amplitudes, and accumulated modal reactions."""
    compliance, residual, response = modal_terms(bodies, previous_amplitudes, dt)  # [..., n_modes] Scalar+[n_bodies], Scalar+[n_bodies], Scalar+[n_bodies]
    rigid, modal, modal_load, gap = coupling(bodies, constraints)
    spring_load = -(residual + compliance * mode_reactions)                # [..., n_modes] Scalar+[n_bodies]
    unconstrained_step = response * spring_load                            # [..., n_modes] Scalar+[n_bodies]
    mobility = spdiag(response * compliance)                               # [..., n_modes] Scalar+[n_bodies <- n_bodies]
    constraint_compliance = constraints.compliance / dt**2 * Force.dual()  # (Antivector- <- Vector+)[n_constraints]
    # Because `rigid.adjugate()(force) & twist == force & rigid(twist)`, the adjugate maps joint
    # forces directly to body forques without a coordinate transpose or metric on twists.
    system = rigid(bodies.inverse_inertia(rigid.adjugate())) + (modal * (mobility * modal_load)).sum(axis=-1) + constraint_compliance  # [...] (Antivector- <- Vector+)[n_constraints <- n_constraints]
    reactions = system.solve((-gap - (modal * unconstrained_step).sum(axis=-1)).cast(Displacement))  # [...] Vector+[n_constraints]
    twist_step = bodies.inverse_inertia(rigid.adjugate()(reactions))       # [...] Bivector+[n_bodies]
    mode_loads = modal_load(reactions[..., None])                          # [..., n_modes] Scalar+[n_bodies]
    reaction_change = response * (spring_load - mode_loads)                # [..., n_modes] Scalar+[n_bodies]
    stepped_motor = bodies.motor * (twist_step * -0.5).exp()               # [...] Rotor+[n_bodies]
    constrained_bodies = replace(
        bodies,
        motor=stepped_motor,
        amplitudes=bodies.amplitudes + mode_loads + reaction_change,
    )
    mode_reactions = mode_reactions + reaction_change                      # [..., n_modes] Scalar+[n_bodies]
    return constrained_bodies, mode_reactions


def step(bodies: Bodies, constraints: Constraints, dt: float, gravity: Displacement) -> Bodies:
    """Free rigid-body prediction under `gravity` followed by simultaneous modal and joint
    projection advances `bodies` by one step `dt`."""
    previous = bodies

    def weight(motor: Motor, rate: Twist) -> Forque:
        # Gravity acts at each body's centre of mass, placed at the origin of the body frame.
        return (mv.w.dual() & (motor << gravity)) * bodies.masses          # [...] Antibivector-[n_bodies]

    motor, _ = lie.explicit_rk4(bodies.motor, bodies.rate, bodies.inertia, bodies.inverse_inertia, dt, weight)
    bodies = replace(bodies, motor=motor, amplitudes=bodies.amplitudes + bodies.rates * dt)
    _, residual, response = modal_terms(bodies, previous.amplitudes, dt)   # [..., n_modes] Scalar+[n_bodies], Scalar+[n_bodies], Scalar+[n_bodies]
    mode_reactions = -residual * response                                  # [..., n_modes] Scalar+[n_bodies]
    bodies, _ = project_constraints(
        replace(bodies, amplitudes=bodies.amplitudes + mode_reactions),
        constraints,
        previous.amplitudes,
        mode_reactions,
        dt,
    )
    return replace(
        bodies,
        rate=(~previous.motor * bodies.motor).log() * (-2 / dt),
        rates=(bodies.amplitudes - previous.amplitudes) / dt,
    )
