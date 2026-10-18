"""Low-Reynolds-number swimming and regularized Stokeslets in the projective geometric algebra of the plane.

In a slow, viscous flow where inertia is negligible, a swimmer experiences no coasting: at every instant
the forces its body exerts on the fluid and their moments about any point sum to zero, a vanishing
wrench (`Forque`). Modeling the swimmer as a collection of small spheres of radius `radius` at points
`sites` (`Point`), a force `f` (`Force`) on one bead drives a fluid velocity (`Direction`) at separation
`r = (target & source) | ORIGIN` given by the Cortez regularized Stokeslet:

    stokeslet(r) = ((r2 + radius**2) * Force.dual() + r.dual() * (r | Force)) / (r2 * r2.square_root())

with `r2 = r.norm_squared() + radius**2`. At zero separation `r == 0`, this reduces to the bead's
self-mobility `(2 / radius) * Force.dual()`. Evaluated between all pairs of beads on the swimmer,
`.field(0, 1)` packages the pairwise blocks into the mobility operator `M: Direction[beads] <- Force[beads]`.

A rigid motion rate in the swimmer's body frame is a bivector `Twist`. Leaving `Twist` open in the
commutator with the bead points,

    rigid = sites.commutator(Twist).field(0)

gives the kinematic map `Direction[beads] <- Twist` from the body's twist to the velocity of every bead.
Its incidence adjugate `rigid.adjugate()` (`Forque <- Force[beads]`) sums the bead forces and their
lever-arm moments into the total body wrench (`Forque`), because `rigid.adjugate()(forces) & twist`
equals the virtual power `(forces & rigid(twist)).sites.sum()`. When the swimmer deforms its hinges with
prescribed body-frame velocities `shape_velocity`, zero net wrench `rigid.adjugate()(forces) == 0` gives
the Schur-complement grand resistance map `resistance = rigid.adjugate()(M.solve(rigid))` (`Forque <- Twist`)
and the instantaneous swimming twist

    body_rate = -resistance.solve(rigid.adjugate()(M.solve(shape_velocity)))

Each time step then multiplies `(body_rate * (-dt / 2)).exp()` on the right of the swimmer's pose `Motor`.
A reciprocal two-hinge stroke retraces a curve in hinge-angle space and returns to its starting pose
after every cycle (Purcell's scallop theorem); Purcell's non-reciprocal three-link stroke encloses area
in hinge-angle space and produces net locomotion.

In the matrix notation of microhydrodynamics, the mobility operator reads as a $2N \\times 2N$ block
Oseen matrix $M_{ij} = \\frac{(r_{ij}^2 + 2\\epsilon^2)I + \\mathbf{r}_{ij}\\mathbf{r}_{ij}^\\top}{(r_{ij}^2 + \\epsilon^2)^{3/2}}$,
the rigid kinematics as a $2N \\times 3$ matrix $R$ of identity and lever-arm blocks, and the grand
resistance map as the $3 \\times 3$ Schur complement $R^\\top M^{-1} R$.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, concatenate
from numga.algebras import PGA2D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Point = ga.gatype.antivector()
Twist = ga.gatype.bivector()
Forque = ga.gatype.antibivector()
Motor = ga.gatype.rotor()
# Lines x and y through the origin represent forces and Euclidean separations; their complements are
# ideal points representing velocities and displacements, paired by `force & direction`.
Force = ga.gatype.vector().derive.nondegenerate
Direction = Force.dual().output_gatype
Mobility = ga.gatype((Direction, Force))                                        # Direction <- Force

ORIGIN = mv.w.dual()                                                            # [] Point


# --- math -----------------------------------------------------------------------------
def separation(target: Point, source: Point) -> Force:
    """The Euclidean vector from `source` to `target`, read from their joining line at `ORIGIN`."""
    return (target & source) | ORIGIN                                           # [...] Force


def stokeslet(separations: Force, radius: float) -> Mobility:
    """Fluid velocity (`Direction`) per unit blob force (`Force`) at `separations`, for regularized
    Stokeslets of blob `radius`."""
    r2 = separations.norm_squared() + radius**2                                 # [...] Scalar
    return ((r2 + radius**2) * Force.dual() + separations.dual() * (separations | Force)) / (r2 * r2.square_root())


def swim(sites: Point, shape_velocity: Direction, radius: float) -> tuple[Twist, Force]:
    """Solve for the body-frame rigid rate `[...] Twist` and bead forces `[...] Force[beads]` of a
    force- and torque-free swimmer with bead positions `sites` `[..., beads] Point` and prescribed
    body-frame deformation velocities `shape_velocity` `[..., beads] Direction`."""
    pairs = separation(sites[..., :, None], sites[..., None, :])                # [..., beads, beads] Force
    M = stokeslet(pairs, radius).field(0, 1)                                    # [...] Direction[beads] <- Force[beads]
    rigid = sites.commutator(Twist).field(0)                                    # [...] Direction[beads] <- Twist
    resistance = rigid.adjugate()(M.solve(rigid))                               # [...] Forque <- Twist
    body_rate = -resistance.solve(rigid.adjugate()(M.solve(shape_velocity.field())))
    forces = M.solve(rigid(body_rate) + shape_velocity.field())                 # [...] Force[beads]
    return body_rate, forces


def flow_field(sample_points: Point, bead_points: Point, bead_forces: Force, radius: float) -> Direction:
    """Fluid velocity `[...] Direction` at `sample_points` induced by `bead_forces` at `bead_points`."""
    pairs = separation(sample_points[..., None], bead_points)                   # [..., beads] Force
    kernels = stokeslet(pairs, radius)                                          # [..., beads] Direction <- Force
    return kernels(bead_forces).sum(axis=-1)                                    # [...] Direction


def integrate_pose(body_rate: Twist, dt: float, start_motor: Motor) -> Motor:
    """Step `start_motor` `[...] Motor` by `(body_rate * (-dt / 2)).exp()` in the body frame at each
    step `[steps, ...] Twist`, returning `[steps + 1, ...] Motor`."""
    steps = (body_rate * (-dt / 2.0)).exp()                                     # [steps, ...] Motor
    moved = start_motor * steps.reverse().cumprod(axis=0).reverse()             # [steps, ...] Motor
    return concatenate([start_motor[None], moved], axis=0)                      # [steps + 1, ...] Motor


# --- plumbing -------------------------------------------------------------------------
def three_link_shape(
    hinge_half_span: float,
    centre_offsets: np.ndarray,
    arm_offsets: np.ndarray,
    angles: np.ndarray,
    rates: np.ndarray,
) -> tuple[Point, Direction]:
    """Bead points `[..., beads] Point` and body-frame deformation velocities `[..., beads] Direction`
    for a three-link Purcell swimmer with hinge angles `angles` `[..., 2]` and angular rates `rates` `[..., 2]`."""
    # The beads lie along the swimmer's axis, the line y = 0, at their offsets from the origin.
    along = mv.direction([1.0, 0.0])                                            # [] Point
    centre_beads = ORIGIN + along * centre_offsets                              # [centre] Point
    # In PGA2D a hinge is the meet of two lines through it, the swimmer's axis and the line across it, a
    # bivector generating turns about itself.
    left_hinge = (mv.x + hinge_half_span * mv.w) ^ mv.y                         # [] Twist
    right_hinge = (mv.x - hinge_half_span * mv.w) ^ mv.y                        # [] Twist
    left_turn = (left_hinge * (angles[..., 0, None] / 2.0)).exp()               # [..., 1] Motor
    right_turn = (right_hinge * (-angles[..., 1, None] / 2.0)).exp()            # [..., 1] Motor

    left_rest = ORIGIN - along * (hinge_half_span + arm_offsets[::-1])          # [arm] Point
    right_rest = ORIGIN + along * (hinge_half_span + arm_offsets)               # [arm] Point
    left_beads = left_turn >> left_rest                                         # [..., arm] Point
    right_beads = right_turn >> right_rest                                      # [..., arm] Point

    left_vel = left_beads.commutator(-left_hinge * rates[..., 0, None])         # [..., arm] Direction
    right_vel = right_beads.commutator(right_hinge * rates[..., 1, None])       # [..., arm] Direction
    centre_vel = (angles[..., :1] * 0.0) * centre_beads.commutator(right_hinge)   # [..., centre] Direction

    shape = angles.shape[:-1]
    sites = concatenate([left_beads, centre_beads.broadcast_to(shape + centre_beads.shape), right_beads], axis=-1)
    velocities = concatenate([left_vel, centre_vel, right_vel], axis=-1)
    return sites, velocities
