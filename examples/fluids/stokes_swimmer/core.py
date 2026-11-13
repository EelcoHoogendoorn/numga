r"""Swimming without inertia, in the projective geometric algebra of the plane.

This example computes how a swimmer moves through a viscous fluid without inertia, Purcell's three-link swimmer beside a reciprocal flapper. In slow, thick flow, such as a bacterium in water or a person in honey, nothing coasts, so a swimmer that stops changing its shape stops at once, and at every instant the forces its body exerts on the fluid add up to no net force and no net torque. The swimmer is modelled as a chain of beads, each pushing on the fluid through a regularized Stokeslet, the flow of a force spread over a blob of the bead's size. Because the body's motion is linear in the rate at which its shape changes, a stroke that goes back along the path it came undoes itself, which is Purcell's scallop theorem, while a swimmer that takes its two hinges around a loop advances with every cycle.

The beads are points and the body's motion is a rate, a bivector, so `sites.commutator(Twist)` is the map from the body's rate to every bead's velocity. Its adjugate sums the beads' forces into the forque on the body, because a forque paired with a rate is the power the forces deliver to beads moving at that rate, `rigid_velocities.adjugate()(forces) & rate`, with no lever arms and no reference point. One solve with the resistance `rigid_velocities.adjugate()(bead_mobility.solve(rigid_velocities))` gives the body's rate that keeps the forque zero, and the poses are the running product of the step motors `(body_rate * (-dt / 2)).exp()`.

In the matrix notation of Stokesian dynamics, the bead velocities $\mathbf U = \mathsf M\,\mathbf F$ come from a $2N\times 2N$ mobility matrix of blocks $\mathsf M_{ij} \propto \big((r_{ij}^2 + 2\epsilon^2)\,\mathsf I + \mathbf r_{ij}\mathbf r_{ij}^{\mathsf T}\big)/(r_{ij}^2 + \epsilon^2)^{3/2}$, rigid motion enters through the $2N\times 3$ matrix $\mathsf K$ with rows $\big(\mathsf I \;\; \mathsf J\,(\mathbf x_i - \mathbf x_0)\big)$, $\mathsf J = \begin{pmatrix}0 & -1\\ 1 & 0\end{pmatrix}$, about a chosen reference point $\mathbf x_0$, and the force and torque constraints $\mathsf K^{\mathsf T}\mathbf F = \big(\sum_i \mathbf F_i,\ \sum_i (\mathbf x_i - \mathbf x_0)\times\mathbf F_i\big) = 0$ give the body's velocity and angular velocity as $(\mathbf V, \Omega) = -(\mathsf K^{\mathsf T}\mathsf M^{-1}\mathsf K)^{-1}\,\mathsf K^{\mathsf T}\mathsf M^{-1}\,\mathbf u^{\mathrm{shape}}$. Written as $(\mathbf V, \Omega) = \mathsf A(\boldsymbol\theta)\,\dot{\boldsymbol\theta}$ over the hinge angles, this is the Shapere–Wilczek connection, and the net motion per cycle is the path-ordered exponential of $\oint \mathsf A\,d\boldsymbol\theta$ in $\mathrm{SE}(2)$.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, concatenate
from numga.algebras import PGA2D as ga

mv = NumpyContext(ga).multivector
Point = ga.gatype.antivector()                                                  # Antivector-
Twist = ga.gatype.bivector()                                                    # Bivector+
Motor = ga.gatype.rotor()                                                       # Rotor+
# A force is a plane through the origin, extrinsic, read by its pairing with a velocity or a displacement, an ideal
# point, `force & direction`.
Force = ga.gatype.vector().derive.nondegenerate                                 # Vector+
Direction = Force.derive.dual                                                   # Antivector-
Mobility = ga.gatype((Direction, Force))                                        # Direction- <- Force+

origin = mv.w.dual()                                                            # [] Point-


# --- math -----------------------------------------------------------------------------
def stokeslet(targets: Point, sources: Point, radius: float) -> Mobility:
    """The fluid velocity at the targets per unit force spread over a blob of the given radius at the sources."""
    # The difference of two points of one weight is a direction; as one, the map from forces to velocities is
    # square, as its solves need.
    separations = (targets - sources).cast(Direction)                           # [...] Direction-
    # The squared distance, the squared norm of the line joining the two points, softened by the radius: the
    # blob spreads each force over the bead's size, which keeps the flow finite where the force acts.
    blob_squares = (targets & sources).norm_squared() + radius**2               # [...] Scalar+
    # The fluid moves along the force, and further along the separation by the force's pairing with it;
    # both parts fall off with distance.
    return ((blob_squares + radius**2) * Force.dual() + separations * (Force & separations)) / (blob_squares * blob_squares.square_root())


def mobility(sites: Point, radius: float) -> Mobility:
    """The map from all the beads' forces to all their velocities, its blocks the Stokeslets between every pair of
    beads, formed over the beads as a batch and assembled into a map between the two fields."""
    beads = sites.batch()                                                       # [..., n_beads] Point-
    return stokeslet(beads[..., :, None], beads[..., None, :], radius).field(0, 1)   # [...] Direction-[n_beads] <- Force+[n_beads]


def force_free_motion(sites: Point, shape_velocities: Direction, radius: float) -> tuple[Twist, Force]:
    """The body's rate in its own frame and the beads' forces of a swimmer free of force and torque, as its beads
    change shape at the given velocities."""
    bead_mobility = mobility(sites, radius)                                     # [...] Direction-[n_beads] <- Force+[n_beads]
    # Over `Twist`, the space of all rates, each bead's commutator with the rate is that bead's velocity.
    rigid_velocities = sites.commutator(Twist)                                  # [...] Direction-[n_beads] <- Twist+
    # The forces that carry the beads along with the body at a given rate, summed into the forque on the
    # body: the adjugate carries forces back across the map that carried the rate out to velocities.
    resistance = rigid_velocities.adjugate()(bead_mobility.solve(rigid_velocities))   # [...] Forque- <- Twist+
    # Changing shape alone would push on the fluid too; the body moves at the rate whose forque cancels it.
    body_rate = -resistance.solve(rigid_velocities.adjugate()(bead_mobility.solve(shape_velocities)))   # [...] Twist+
    # The forces the beads push with under both motions together, balanced by construction.
    forces = bead_mobility.solve(rigid_velocities(body_rate) + shape_velocities)   # [...] Force+[n_beads]
    return body_rate, forces


def flow_field(sample_points: Point, bead_points: Point, bead_forces: Force, radius: float) -> Direction:
    """The fluid velocity at the sample points: the map from the beads' forces to the flow at each sample point,
    applied to the forces, sums the flows of all the beads."""
    return stokeslet(sample_points[..., None], bead_points.batch(), radius).field(1)(bead_forces)   # [...] Direction-


def poses(body_rate: Twist, dt: float, start_motor: Motor) -> Motor:
    """The body's poses at the start and after every step, each step's motor applied in the body's own frame."""
    step_motors = (body_rate * (-dt / 2)).exp()                                 # [...] Motor+
    # Each step moves the body in its own frame, so it multiplies the pose on the right. A running product
    # puts later factors on the left; taking it over the steps' reverses and reversing the result puts
    # them on the right.
    later_poses = start_motor * step_motors.reverse().cumprod(axis=0).reverse()   # [...] Motor+
    return concatenate([start_motor[None], later_poses], axis=0)               # [...] Motor+


def three_link_shape(
    hinge_half_span: float,
    centre_offsets: np.ndarray,
    arm_offsets: np.ndarray,
    angles: np.ndarray,
    rates: np.ndarray,
) -> tuple[Point, Direction]:
    """The beads of a three-link swimmer at the given hinge angles, and their velocities in the body's frame as the
    hinges turn at the given angular rates."""
    # The beads lie along the swimmer's axis, through the origin along x, at their offsets from the origin.
    axis = mv.direction([1, 0])                                                 # [] Direction-
    centre_beads = origin + axis * centre_offsets                               # [...] Point-
    # A hinge is the meet of two planes through it, the plane along the swimmer's axis and the plane across it, a
    # bivector generating turns about itself.
    left_hinge = (mv.x + hinge_half_span * mv.w) ^ mv.y                         # [] Twist+
    right_hinge = (mv.x - hinge_half_span * mv.w) ^ mv.y                        # [] Twist+
    left_turn = (left_hinge * (angles[..., 0, None] / 2)).exp()                 # [..., 1] Motor+
    right_turn = (right_hinge * (-angles[..., 1, None] / 2)).exp()              # [..., 1] Motor+

    left_rest = origin - axis * (hinge_half_span + arm_offsets[::-1])           # [...] Point-
    right_rest = origin + axis * (hinge_half_span + arm_offsets)                # [...] Point-
    left_beads = left_turn >> left_rest                                         # [...] Point-
    right_beads = right_turn >> right_rest                                      # [...] Point-
    centre_sites = centre_beads.broadcast_to(angles.shape[:-1] + centre_beads.shape)   # [...] Point-

    # Each arm turns about its hinge at the hinge's rate, so a bead's velocity is its commutator with that
    # rate; the left hinge's angle counts the other way round. The centre link does not move in the body's
    # frame.
    left_velocities = left_beads.commutator(-left_hinge * rates[..., 0, None])  # [...] Direction-
    right_velocities = right_beads.commutator(right_hinge * rates[..., 1, None])   # [...] Direction-
    centre_velocities = 0 * centre_sites.commutator(right_hinge)                # [...] Direction-

    sites = concatenate([left_beads, centre_sites, right_beads], axis=-1)      # [...] Point-
    velocities = concatenate([left_velocities, centre_velocities, right_velocities], axis=-1)   # [...] Direction-
    return sites.field(), velocities.field()
