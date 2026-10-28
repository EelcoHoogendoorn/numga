"""Swimming without inertia, in the projective geometric algebra of space.

In slow, thick flow nothing coasts: the moment a swimmer stops moving its body, it stops. At every
instant the pushes its body gives the fluid balance, with no net force and no net turning, and the
body moves at exactly the rate that keeps them balanced.

The swimmer is a chain of beads. A push on the fluid at one bead sets all of the fluid moving, less and
less with distance; a point force would move the fluid infinitely fast at the point itself, so each
bead's force is spread over a blob of the bead's size, a regularized Stokeslet. Over every pair of
beads these flows make one map from the forces the beads push with to the velocities they move at.

The body's own motion is a rate, a bivector, and the commutator of each bead with the rate is that
bead's velocity: over `Twist`, the space of all rates, `sites.commutator(Twist)` is the map from the body's
rate to every bead's velocity. The same map, carried across the regressive product, sums the beads' forces into the
forque on the body: its adjugate, because a forque paired with a rate is the power the forces deliver
to beads moving at that rate, `rigid.adjugate()(forces) & rate` the summed power of the forces. As the
swimmer changes its shape, one solve finds the body's rate that keeps that forque zero.

With nothing coasting, the body's rate is linear in the rate at which the shape changes. How far a
stroke carries the swimmer therefore depends only on the path its shape takes, not on how fast it goes,
and a stroke that goes back along the path it came undoes itself: Purcell's scallop theorem. Purcell's
three-link swimmer takes its two hinges around a loop instead, and steps ahead with every cycle.

In the matrix notation of microhydrodynamics, the map from forces to velocities is the beads' Oseen
matrix, the map from the body's rate to the beads' velocities a matrix of identity and lever-arm
blocks, and the body's rate comes from their three by three resistance matrix.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, concatenate
from numga.algebras import PGA2D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                     # Scalar+
Point = ga.gatype.antivector()                                                  # Antivector-
Twist = ga.gatype.bivector()                                                    # Bivector+
Forque = ga.gatype.antibivector()                                               # Antibivector-
Motor = ga.gatype.rotor()                                                       # Rotor+
# A force is a plane through the origin, extrinsic, read by its pairing with a velocity or a displacement, an ideal
# point, `force & direction`.
Force = ga.gatype.vector().derive.nondegenerate                                 # Vector+
Direction = Force.dual().output_gatype                                          # Antivector-
Mobility = ga.gatype((Direction, Force))                                        # Direction- <- Force+

origin = mv.w.dual()                                                            # [] Point-


# --- math -----------------------------------------------------------------------------
def stokeslet(targets: Point, sources: Point, radius: float) -> Mobility:
    """Fluid velocity (`Direction`) at the targets per unit blob force (`Force`) at the sources, for
    regularized Stokeslets of blob `radius`."""
    # The difference of two points of one weight is a direction; as one, the map from forces to velocities is
    # square, as its solves need.
    separations = (targets - sources).cast(Direction)                           # [...] Direction-
    # The squared distance, the squared norm of the line joining the two points, softened by the radius: the
    # blob spreads each force over the bead's size, which keeps the flow finite where the force acts.
    softened = (targets & sources).norm_squared() + radius**2                   # [...] Scalar+
    # The fluid moves along the force, and further along the separation by the force's pairing with it;
    # both parts fall off with distance.
    return ((softened + radius**2) * Force.dual() + separations * (Force & separations)) / (softened * softened.square_root())


def mobility(sites: Point, radius: float) -> Mobility:
    """The flow each bead's force drives at every other bead: one map from all the beads' forces to all
    their velocities, `[...] Direction[beads] <- Force[beads]`, for beads at `[...] Point[beads]`. Its
    blocks depend on every pair of beads, so they are formed over the beads as a batch and assembled
    into the map between the two fields."""
    beads = sites.batch()                                                       # [..., beads] Point-
    return stokeslet(beads[..., :, None], beads[..., None, :], radius).field(0, 1)   # [...] Direction-[beads] <- Force+[beads]


def swim(sites: Point, shape_velocity: Direction, radius: float) -> tuple[Twist, Force]:
    """Solve for the body-frame rigid rate `[...] Twist` and bead forces `[...] Force[beads]` of a
    force- and torque-free swimmer with bead positions `sites` `[...] Point[beads]` and prescribed
    body-frame deformation velocities `shape_velocity` `[...] Direction[beads]`."""
    bead_mobility = mobility(sites, radius)                                     # [...] Direction-[beads] <- Force+[beads]
    # Over `Twist`, the space of all rates, each bead's commutator with the rate is that bead's velocity.
    rigid = sites.commutator(Twist)                                             # [...] Direction-[beads] <- Twist+
    # The forces that carry the beads along with the body at a given rate, summed into the forque on the
    # body: the adjugate carries forces back across the map that carried the rate out to velocities.
    resistance = rigid.adjugate()(bead_mobility.solve(rigid))                   # [...] Forque- <- Twist+
    # Changing shape alone would push on the fluid too; the body moves at the rate whose forque cancels it.
    body_rate = -resistance.solve(rigid.adjugate()(bead_mobility.solve(shape_velocity)))
    # The forces the beads push with under both motions together, balanced by construction.
    forces = bead_mobility.solve(rigid(body_rate) + shape_velocity)             # [...] Force+[beads]
    return body_rate, forces


def flow_field(sample_points: Point, bead_points: Point, bead_forces: Force, radius: float) -> Direction:
    """Fluid velocity `[...] Direction` at `sample_points` induced by `bead_forces` `[...] Force[beads]` at
    `bead_points` `[...] Point[beads]`: the map from the beads' forces to the flow at each sample point,
    applied to the forces, sums the flows of all the beads."""
    return stokeslet(sample_points[..., None], bead_points.batch(), radius).field(1)(bead_forces)   # [...] Direction-


def integrate_pose(body_rate: Twist, dt: float, start_motor: Motor) -> Motor:
    """Step `start_motor` `[...] Motor` by `(body_rate * (-dt / 2)).exp()` in the body frame at each
    step `[steps, ...] Twist`, returning `[steps + 1, ...] Motor`."""
    steps = (body_rate * (-dt / 2.0)).exp()                                     # [steps, ...] Motor+
    # Each step moves the body in its own frame, so it multiplies the pose on the right. A running product
    # puts later factors on the left; taking it over the steps' reverses and reversing the result puts
    # them on the right.
    moved = start_motor * steps.reverse().cumprod(axis=0).reverse()             # [steps, ...] Motor+
    return concatenate([start_motor[None], moved], axis=0)                      # [steps + 1, ...] Motor+


# --- plumbing -------------------------------------------------------------------------
def three_link_shape(
    hinge_half_span: float,
    centre_offsets: np.ndarray,
    arm_offsets: np.ndarray,
    angles: np.ndarray,
    rates: np.ndarray,
) -> tuple[Point, Direction]:
    """Bead points `[...] Point[beads]` and body-frame deformation velocities `[...] Direction[beads]`
    for a three-link Purcell swimmer with hinge angles `angles` `[..., 2]` and angular rates `rates` `[..., 2]`."""
    # The beads lie along the swimmer's axis, through the origin along x, at their offsets from the origin.
    along = mv.direction([1.0, 0.0])                                            # [] Point-
    centre_beads = origin + along * centre_offsets                              # [centre] Point-
    # A hinge is the meet of two planes through it, the plane along the swimmer's axis and the plane across it, a
    # bivector generating turns about itself.
    left_hinge = (mv.x + hinge_half_span * mv.w) ^ mv.y                         # [] Twist+
    right_hinge = (mv.x - hinge_half_span * mv.w) ^ mv.y                        # [] Twist+
    left_turn = (left_hinge * (angles[..., 0, None] / 2.0)).exp()               # [..., 1] Motor+
    right_turn = (right_hinge * (-angles[..., 1, None] / 2.0)).exp()            # [..., 1] Motor+

    left_rest = origin - along * (hinge_half_span + arm_offsets[::-1])          # [arm] Point-
    right_rest = origin + along * (hinge_half_span + arm_offsets)               # [arm] Point-
    left_beads = left_turn >> left_rest                                         # [..., arm] Point-
    right_beads = right_turn >> right_rest                                      # [..., arm] Point-

    # Each arm turns about its hinge at the hinge's rate, so a bead's velocity is its commutator with that
    # rate; the left hinge's angle counts the other way round. The centre link does not move in the body's
    # frame.
    left_vel = left_beads.commutator(-left_hinge * rates[..., 0, None])         # [..., arm] Direction-
    right_vel = right_beads.commutator(right_hinge * rates[..., 1, None])       # [..., arm] Direction-
    centre_vel = (angles[..., :1] * 0.0) * centre_beads.commutator(right_hinge)   # [..., centre] Direction-

    shape = angles.shape[:-1]
    sites = concatenate([left_beads, centre_beads.broadcast_to(shape + centre_beads.shape), right_beads], axis=-1)
    velocities = concatenate([left_vel, centre_vel, right_vel], axis=-1)
    return sites.field(), velocities.field()
