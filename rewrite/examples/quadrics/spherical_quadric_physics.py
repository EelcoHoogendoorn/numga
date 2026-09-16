"""Spherical Quadric Rigid Body Physics in Cl(3) = R_{3,0,0}.

A unified Geometric Algebra simulation engine for oriented dual quadrics gliding,
rotating, and colliding across the 2-sphere S².
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from numga import NumpyContext
from numga.algebras import Spherical3D

# ---------------------------------------------------------------------------
# 1. Cl(3) Spherical Geometric Algebra Setup
# ---------------------------------------------------------------------------
ga = Spherical3D
ctx = NumpyContext(ga)
mv = ctx.multivector

# Whole-extensor types (GATypes):
Point = ga.gatype.point()
Plane = ga.gatype.plane()
Line = Plane  # On S², geodesic lines are great circles / cutting planes
Bivector = ga.gatype.bivector()
AntiBivector = ga.gatype.antibivector()
Rotor = ga.gatype.rotor()
Motor = Rotor
Quadric = ga.gatype((Point, Plane))
InertiaMap = ga.gatype((AntiBivector, Bivector))
InverseInertiaMap = ga.gatype((Bivector, AntiBivector))
Scalar = ga.gatype.scalar()
Form = ga.gatype((Scalar, Point, Point))       # a quadric with both of its points open: scalar <= point, point


# ---------------------------------------------------------------------------
# 2. Quadric & Inertia Geometry
# ---------------------------------------------------------------------------
def pointcloud_inertia(pts: Point, masses: np.ndarray) -> InertiaMap:
    """Derive physical inertia extensor I : Bivector -> Plane directly from points on S²."""
    per_point = pts.regressive(pts.commutator(Bivector))
    return (per_point * mv.scalar(masses[..., None])).sum()


def compose_quadric(coeffs: np.ndarray, poles: Point) -> Quadric:
    """Construct dual quadric from batched principal poles and coefficients."""
    dual_projectors = poles * Plane.regressive(poles)
    weights = mv.scalar(coeffs[..., None])
    return (dual_projectors * weights).sum(axis=0)


def eigenpairs(form: Form) -> tuple[np.ndarray, Point]:
    """The eigenvalues, ascending, and the eigenvectors as points of a symmetric form on points."""
    values, vectors = np.linalg.eigh(form.kernel[..., 0, :, :])
    return values, mv(Point, np.swapaxes(vectors, -1, -2))


def make_spherical_quadric(th_x: float, th_y: float) -> Quadric:
    """Construct a canonical dual spherical quadric Q : Plane -> Point."""
    poles = mv.point(np.eye(3))
    coeffs = np.array([np.tan(th_x)**2, np.tan(th_y)**2, -1.0])
    return compose_quadric(coeffs, poles)


# ---------------------------------------------------------------------------
# 3. Rigid Body State Container & Lie Integrator
# ---------------------------------------------------------------------------
@dataclass
class SphericalBody:
    """Plain data container for spherical rigid body state and geometry."""
    name: str
    color: str
    mass: float
    motor: Motor
    momentum: AntiBivector
    Q: Quadric
    cap_pts: Point
    I_inv: InverseInertiaMap


def step_motor(motor: Motor, momentum: AntiBivector, I_inv: InverseInertiaMap, dt: float) -> tuple[Motor, AntiBivector]:
    """Advance motor and local momentum by dt using Lie midpoint integration."""
    w1 = I_inv(momentum)
    dR1 = (w1 * (0.25 * dt)).exp()
    motor_mid = (motor * dR1).normalized()
    w_mid = I_inv((motor.inverse() * motor_mid) << momentum)
    dR = (w_mid * (0.5 * dt)).exp()
    motor_next = (motor * dR).normalized()
    dR_step = motor.inverse() * motor_next
    return motor_next, dR_step << momentum


# ---------------------------------------------------------------------------
# 4. Projective Dual Pencil Collision Engine
# ---------------------------------------------------------------------------
def overlap(A: Quadric, B: Quadric, iterations: int = 12) -> tuple[np.ndarray, Point]:
    """Whether the insides of two quadrics, where their primal forms are negative, meet: they are
    apart if and only if some member of the pencil A + λB, λ > 0, is positive semidefinite (the
    S-lemma), so the largest over the pencil of the least eigenvalue is negative exactly when they
    overlap. The least eigenvalue is concave in λ, hence unimodal in φ = arctan λ over (0, π/2),
    and golden section finds its maximum; the least eigenvector there is the deepest point, the
    touching point when the margin is zero. Twelve iterations bracket φ to 0.005 rad; five
    misreport a separated pair of the hyperbolic scene as touching. Batched over pairs."""
    def least(phi: np.ndarray) -> np.ndarray:
        return eigenpairs(Point & (A + B * np.tan(phi))(mv.rotor() >> Point))[0][..., 0]

    golden = (np.sqrt(5.0) - 1.0) / 2.0
    lo, hi = np.broadcast_to(0.0, np.broadcast_shapes(A.shape, B.shape)), np.broadcast_to(np.pi / 2, np.broadcast_shapes(A.shape, B.shape))
    c, d = hi - golden * (hi - lo), lo + golden * (hi - lo)
    fc, fd = least(c), least(d)
    for _ in range(iterations):
        left = fc > fd                                            # the maximum lies in [lo, d]; the kept probe becomes the other one
        lo, hi = np.where(left, lo, c), np.where(left, d, hi)
        c, d = hi - golden * (hi - lo), lo + golden * (hi - lo)
        fresh = least(np.where(left, c, d))
        fc, fd = np.where(left, fresh, fd), np.where(left, fc, fresh)
    values, points = eigenpairs(Point & (A + B * np.tan((lo + hi) / 2))(mv.rotor() >> Point))
    return values[..., 0], points[..., 0]


def resolve_collision(
    body1: SphericalBody,
    body2: SphericalBody,
    deepest: Point,
    M_rel: Motor,
    restitution: float = 1.0,
) -> bool:
    """Detect and resolve elastic collision impulse entirely inside Geometric Algebra: the first
    body's polar plane at the deepest point is the contact plane, its pole the contact point."""
    contact_plane = body1.Q.inverse()(deepest)
    contact_point = body1.Q(contact_plane)
    contact_wrench_1: Line = contact_plane.commutator(contact_point)
    contact_wrench_2: Line = M_rel << contact_wrench_1

    rate_response_1: Bivector = body1.I_inv(contact_wrench_1)
    rate_response_2: Bivector = body2.I_inv(contact_wrench_2)

    closing_velocity: Scalar = (
        rate_response_1.regressive(body1.momentum)
        - rate_response_2.regressive(body2.momentum)
    )

    if closing_velocity.kernel.item() < 0.0:
        effective_compliance: Scalar = (
            contact_wrench_1.regressive(rate_response_1)
            + contact_wrench_2.regressive(rate_response_2)
        )
        magnitude: Scalar = (-closing_velocity * (1.0 + restitution)) / effective_compliance

        body1.momentum = body1.momentum + contact_wrench_1 * magnitude
        body2.momentum = body2.momentum - contact_wrench_2 * magnitude
        return True

    return False


# ---------------------------------------------------------------------------
# 5. Unified Simulation Loop
# ---------------------------------------------------------------------------
def simulate(
    bodies: list[SphericalBody],
    num_frames: int = 160,
    dt: float = 0.015,
    substeps: int = 6,
) -> tuple[
    list[SphericalBody],
    list[list[Motor]],
    list[float],
    list[float],
    list,
    dict[int, list[Motor]],
    list[int],
]:
    """Simulate spherical quadric rigid body dynamics on S²."""
    sub_dt = dt / float(substeps)
    frame_motors: list[list[Motor]] = []
    energy_history: list[float] = []
    momentum_history: list[float] = []
    omega_history: list = []

    for frame_idx in range(num_frames):
        for _ in range(substeps):
            for body in bodies:
                body.motor, body.momentum = step_motor(body.motor, body.momentum, body.I_inv, sub_dt)

            for i in range(len(bodies)):
                for j in range(i + 1, len(bodies)):
                    body_i, body_j = bodies[i], bodies[j]
                    M_rel = body_i.motor.inverse() * body_j.motor
                    Q2_in_1 = M_rel >> body_j.Q(M_rel << Plane)
                    margin, deepest = overlap(body_i.Q.inverse(), Q2_in_1.inverse())
                    if margin < 0.0:
                        resolve_collision(body_i, body_j, deepest, M_rel=M_rel, restitution=1.0)

        # Physical invariants:
        frame_w = [np.array(body.I_inv(body.momentum).kernel, copy=True) for body in bodies]
        omega_history.append(frame_w[0] if len(bodies) == 1 else frame_w)

        total_energy = sum(
            abs(float(((body.I_inv(body.momentum)).regressive(body.momentum) * 0.5).kernel.item()))
            for body in bodies
        )
        total_mom_ext = bodies[0].motor >> bodies[0].momentum
        for body in bodies[1:]:
            total_mom_ext = total_mom_ext + (body.motor >> body.momentum)
        total_momentum = float(np.linalg.norm(total_mom_ext.kernel))

        energy_history.append(total_energy)
        momentum_history.append(total_momentum)

        frame_motors.append([body.motor for body in bodies])

    # Keyframe selection:
    if len(bodies) == 1:
        w_yz = np.array([w[0] for w in omega_history])
        crossings = np.where(np.diff(np.sign(w_yz)))[0]
        if len(crossings) >= 2:
            diagnostic_frame_indices = [0, int(crossings[0]), int(crossings[1])]
        else:
            diagnostic_frame_indices = [0, num_frames // 3, 2 * num_frames // 3]
    else:
        diagnostic_frame_indices = [0, num_frames // 3, 2 * num_frames // 3, num_frames - 1]

    keyframe_snapshots = {f_idx: frame_motors[f_idx] for f_idx in diagnostic_frame_indices}

    return (
        bodies,
        frame_motors,
        energy_history,
        momentum_history,
        omega_history,
        keyframe_snapshots,
        diagnostic_frame_indices,
    )
