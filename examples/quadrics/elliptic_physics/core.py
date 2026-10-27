"""Rigid quadric bodies on a sphere: one engine for S² in Cl(3) and S³ in Cl(4).

A body is a dual quadric, an ellipsoid of the sphere, placed by a motor, a rotation of the ambient space. Its inertia
is the sum over its mass points of the regressive product of each point with its own motion,
a map from bivector rates to momenta, and its momentum is kept in the body frame, stepped by a Lie
midpoint rule. Two bodies meet when no blend A + B * weight of their forms, with a positive
weight, is positive semidefinite; the least eigenvector of the best blend is the deepest
point, and the first body's polar plane there is the contact plane, whose inner product with
its pole is the contact forque. Elastic impulses along that forque reverse the closing rate.

Nothing here knows its dimension. `ga` is supplied per instance, by
`examples.instantiate("examples.quadrics.elliptic_physics.core", Spherical3D)` for S², or
with the algebra of the S³ ray tracer for S³.
"""

from dataclasses import dataclass, fields, replace

import numpy as np

from numga import Algebra, concatenate, stack
from numga.backend.context import Context

# The algebra and the context, supplied by examples.instantiate.
ga: Algebra
ctx: Context
mv = ctx.multivector

Scalar = ga.gatype.scalar()                        # Scalar+
Point = ga.gatype.antivector()                     # Antivector-
Plane = ga.gatype.vector()                         # Vector+
Bivector = ga.gatype.bivector()                    # Bivector+
AntiBivector = ga.gatype.antibivector()            # Antibivector-
Motor = ga.gatype.rotor()                          # Rotor+
# The primal quadric maps a point to its polar plane and is negative inside; the dual quadric
# maps a plane to its pole. The inertia maps a rate to its momentum, its inverse the momentum
# back to the rate.
Quadric = ga.gatype((Plane, Point))
DualQuadric = ga.gatype((Point, Plane))
Inertia = ga.gatype((AntiBivector, Bivector))
InverseInertia = ga.gatype((Bivector, AntiBivector))
# The golden-section steps of the overlap test: they bracket the best blend to 0.005 rad, where five misreport near
# pairs as touching.
N_GOLDEN_STEPS = 12

# The poles of the basis planes, and the last of them, the centre of every body in its own frame.
basis = mv.antivector(np.eye(ga.dimension)).normalized()   # [dimension] Point-
origin = basis[-1]                                 # [] Point-


@dataclass(frozen=True)
class Bodies:
    """The whole population as batched extensors: one leading axis, one entry per body."""
    color: np.ndarray
    motor: Motor
    # The momentum is kept in the body frame.
    momentum: AntiBivector
    # The body's shape in its own frame, dual, mapping a plane to its pole, and its form, mapping a
    # point to its polar plane, negative inside.
    dual: DualQuadric
    form: Quadric
    inverse_inertia: InverseInertia
    # The angular radius of each bounding ball, for the broad phase.
    bounding_radius: Scalar

    @staticmethod
    def join(groups: list["Bodies"]) -> "Bodies":
        return Bodies(np.concatenate([g.color for g in groups]), *(
            concatenate([getattr(g, f.name) for g in groups]) for f in fields(Bodies)[1:]
        ))

    def select(self, keep: np.ndarray) -> "Bodies":
        return Bodies(*(getattr(self, f.name)[keep] for f in fields(Bodies)))

    def rate(self) -> Bivector:
        return self.inverse_inertia(self.momentum)

    def world(self) -> Quadric:
        """The primal forms of the bodies placed in the world."""
        return self.motor >> self.form(self.motor << Point)

    def kinetic_energy(self) -> Scalar:
        return (self.rate() & self.momentum).sum() * 0.5

    def total_momentum(self) -> AntiBivector:
        """The momenta of all bodies carried into the world frame and summed."""
        return (self.motor >> self.momentum).sum(axis=0)


@dataclass(frozen=True)
class Trajectory:
    """The simulation, frame by frame: world forms and body rates per body, totals per frame."""
    surfaces: Quadric                              # [frames, bodies]
    # The rate is in the body frame.
    rate: Bivector                                 # [frames, bodies]
    energy: Scalar                                 # [frames]
    # The total momentum is in the world frame.
    momentum: AntiBivector                         # [frames]
    impulses: int


# --- shapes and bodies ----------------------------------------------------------------
def quadric(diagonal: np.ndarray) -> DualQuadric:
    """The dual quadric diagonal on the basis points: the sum of each point paired with itself,
    weighted; negative weights mark the inside's core, positive ones its extents."""
    return (basis * (basis & Plane) * diagonal).sum(axis=-1)


def ellipsoid(half_widths: np.ndarray) -> DualQuadric:
    """The ellipsoid around the origin with the given half-widths along the axes, each the tangent
    of an angular half-width."""
    return quadric(np.concatenate([half_widths**2, -np.ones_like(half_widths[..., :1])], axis=-1))


def pointcloud_inertia(points: Point, masses: Scalar) -> Inertia:
    """The inertia of the cloud, a map from rates to momenta: each point's join with its own motion, mass-weighted."""
    return ((points & points.commutator(Bivector)) * masses).sum(axis=-1)


def body(color: np.ndarray, dual: DualQuadric, motor: Motor, rate: Bivector, points: Point, masses: Scalar) -> Bodies:
    """Bodies of any quadric shape, from mass points filling it: inertia from the points, momentum
    from the body-frame rate, and the farthest of the points from the pole, for the broad phase."""
    inertia = pointcloud_inertia(points, masses)
    spread = (points | origin).abs().clip(0.0, 1.0).arccos()
    bounding_radius = spread[np.arange(spread.shape[0]), spread.argmax(axis=-1)]
    return Bodies(color, motor, inertia(rate), dual, dual.inverse(), inertia.inverse(), bounding_radius)


def filled(dual: DualQuadric, mass: np.ndarray, count: int, rng: np.random.Generator) -> tuple[Point, Scalar]:
    """Mass points filling the inside of any quadric, where its form is negative. In the form's
    eigenbasis the inside is where the negative block outweighs the positive one, so a point is a
    core direction in the negative block, an extent direction in the positive block, both uniform
    on their spheres, and the angle between them, up to the angle where the blocks balance; the
    angle is drawn uniformly and weighted by the sphere's measure,
    angle.cos() ** (k - 1) * angle.sin() ** (n - 1 - k) for k core axes out of n, so the weighted
    points are uniform in the inside. Batched over quadrics of one signature."""
    n = ga.dimension
    # The negative block comes first; k counts its core axes, the same across the batch.
    values, principal = (Point & dual.inverse()).eigh()
    k = int((values < 0.0).sum(axis=-1).ravel()[0])
    core = rng.normal(size=values.shape[:-1] + (count, k))
    extent = rng.normal(size=values.shape[:-1] + (count, n - k))
    core, extent = core / np.linalg.norm(core, axis=-1, keepdims=True), extent / np.linalg.norm(extent, axis=-1, keepdims=True)
    inward = (-values[..., None, :k] * core**2).sum(axis=-1)
    outward = (values[..., None, k:] * extent**2).sum(axis=-1)
    balance = (inward / outward).square_root().arctan()
    angle = balance * rng.uniform(size=balance.shape)
    weight = angle.cos() ** (k - 1) * angle.sin() ** (n - 1 - k) * balance
    coordinates = concatenate([angle.cos()[..., None] * core, angle.sin()[..., None] * extent], axis=-1)
    return (principal[..., None, :] * coordinates).sum(axis=-1), weight * mass[..., None] / weight.sum(axis=-1)[..., None]


# --- math -----------------------------------------------------------------------------
def step_motor(motor: Motor, momentum: AntiBivector, inverse_inertia: InverseInertia, dt: float) -> tuple[Motor, AntiBivector]:
    """A body advanced by dt with a Lie midpoint step; momentum is kept in the body frame."""
    # Half a step ahead along the present rate, the rate there, and a whole step along it.
    half_step = (inverse_inertia(momentum) * (0.25 * dt)).exp()
    midpoint = (motor * half_step).normalized()
    rate = inverse_inertia((motor.inverse() * midpoint) << momentum)
    step = (rate * (0.5 * dt)).exp()
    next_motor = (motor * step).normalized()
    return next_motor, (motor.inverse() * next_motor) << momentum


def overlap(A: Quadric, B: Quadric) -> tuple[Scalar, Point]:
    """Whether the insides of two quadrics, where their forms are negative, meet: they are apart if
    and only if some blend A + B * weight of their forms, with a positive weight, is positive
    semidefinite (the S-lemma), so the largest over the blends of the least eigenvalue is negative
    exactly when they overlap. The least eigenvalue is concave in the weight, hence unimodal in
    phi = np.arctan(weight) between 0 and np.pi / 2, and golden section finds its maximum; the
    least eigenvector there is the deepest point, the touching point when the margin is zero.
    Batched over pairs."""
    def least(phi: np.ndarray) -> Scalar:
        blend = Point & (A + B * np.tan(phi))(Point)
        return blend.eigvalsh()[..., 0]

    # Golden section on phi: two probes c < d split the bracket from lo to hi in the golden ratio;
    # whichever side holds the larger value keeps the bracket, and the surviving probe already sits
    # at the golden point of the shrunk bracket, so each step evaluates one fresh probe only.
    golden = (np.sqrt(5.0) - 1.0) / 2.0
    pairs = np.broadcast_shapes(A.shape, B.shape)
    lo, hi = np.broadcast_to(0.0, pairs), np.broadcast_to(np.pi / 2, pairs)
    c, d = hi - golden * (hi - lo), lo + golden * (hi - lo)
    fc, fd = least(c), least(d)
    for _ in range(N_GOLDEN_STEPS):
        # The maximum lies between lo and d, else between c and hi.
        left = fc > fd
        # Shrink the bracket to that side.
        lo, hi = np.where(left, lo, c), np.where(left, d, hi)
        # The next probes; one coincides with the survivor.
        c, d = hi - golden * (hi - lo), lo + golden * (hi - lo)
        # Evaluate only the other.
        fresh = least(np.where(left, c, d))
        # The survivor's value moves to its place.
        fc, fd = fd.at[left].set(fresh[left]), fresh.at[left].set(fc[left])
    values, points = (Point & (A + B * np.tan((lo + hi) / 2))(Point)).eigh()
    return values[..., 0], points[..., 0]


def candidates_near(bodies: Bodies) -> tuple[np.ndarray, np.ndarray]:
    """Broad phase: the pairs whose poles are closer than their bounding radii summed."""
    poles = bodies.motor >> origin
    apart = (poles[:, None] | poles).abs().clip(0.0, 1.0).arccos()
    return np.nonzero(np.triu(apart < bodies.bounding_radius[:, None] + bodies.bounding_radius[None, :], 1))


def collide(bodies: Bodies, i: np.ndarray, j: np.ndarray, dt: float) -> tuple[Bodies, int]:
    """Contact test and elastic impulses for the candidate pairs (i, j), each in the frame of its first body.

    The overlap margin is negative while the bodies meet. An impulse is applied only while the
    overlap deepens, judged by a virtual step of all bodies along their current rates, so no
    orientation of the contact forque is ever assumed; the impulse reflects the closing rate along
    the forque whatever its sign. The geometry is batched over the pairs; the impulses go one pair
    at a time, each against the momenta the earlier ones left. Returns the bodies after the
    impulses and the number applied.
    """
    def margins(motors: Motor):
        relative = motors[i].inverse() * motors[j]
        return *overlap(bodies.form[i], relative >> bodies.form[j](relative << Point)), relative

    margin, deepest, relative = margins(bodies.motor)
    half_step = (bodies.rate() * (0.5 * dt)).exp()
    ahead = (bodies.motor * half_step).normalized()
    touching = (margin < 0.0) & (margins(ahead)[0] < margin)
    # The first body's polar plane at the deepest point is the contact plane, and its pole is the
    # contact point. The forque is the line through the contact point normal to the contact
    # plane: their inner product.
    contact_plane = bodies.form[i](deepest).normalized()
    contact_point = bodies.dual[i](contact_plane)
    forque_one = contact_plane | contact_point
    forque_other = relative << forque_one
    momentum = bodies.momentum
    for a, b, one, other in zip(i[touching], j[touching], forque_one[touching], forque_other[touching]):
        response_one, response_other = bodies.inverse_inertia[a](one), bodies.inverse_inertia[b](other)
        closing = (response_one & momentum[a]) - (response_other & momentum[b])
        compliance = (one & response_one) + (other & response_other)
        # Elastic: the closing rate reverses.
        impulse = -2.0 * closing / compliance
        momentum = momentum.at[a].set(momentum[a] + one * impulse).at[b].set(momentum[b] - other * impulse)
    return replace(bodies, momentum=momentum), int(touching.sum())


def advance(bodies: Bodies, dt: float) -> tuple[Bodies, int]:
    """Every body stepped freely by dt, and the contacts among the pairs within reach resolved."""
    motor, momentum = step_motor(bodies.motor, bodies.momentum, bodies.inverse_inertia, dt)
    free = replace(bodies, motor=motor, momentum=momentum)
    return collide(free, *candidates_near(free), dt)


def simulate(bodies: Bodies, frames: int, dt: float, substeps: int) -> Trajectory:
    """The initial state and frames - 1 more, each dt apart in substeps."""
    states, impulses = [bodies], 0
    for _ in range(frames - 1):
        for _ in range(substeps):
            bodies, applied = advance(bodies, dt / substeps)
            impulses += applied
        states.append(bodies)
    return Trajectory(
        stack([state.world() for state in states]),
        stack([state.rate() for state in states]),
        stack([state.kinetic_energy() for state in states]),
        stack([state.total_momentum() for state in states]),
        impulses,
    )
