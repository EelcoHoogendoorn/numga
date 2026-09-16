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


def decompose_quadric(Q: Quadric) -> tuple[np.ndarray, Plane]:
    """Dimension-agnostic SVD decomposition of a dual quadric into singular values and dual tangent planes."""
    output_space, input_space = Q.axes
    _, s, vt = np.linalg.svd(Q.kernel)
    tangents = mv(input_space, vt).normalized()
    return s, tangents


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
def find_contact_parameter(Q1: Quadric, Q2: Quadric) -> tuple[float, float]:
    """Find peak parameter λ* maximizing det(Q(λ)) analytically in closed form."""
    y0 = float(np.linalg.det(Q1.kernel))
    y1 = float(np.linalg.det(Q2.kernel))
    y2 = float(np.linalg.det((Q2 * 2.0 - Q1).kernel))
    y3 = float(np.linalg.det((Q1 * 2.0 - Q2).kernel))

    c3 = (3.0 * y0 - 3.0 * y1 + y2 - y3) / 6.0
    c2 = -y0 + 0.5 * y1 + 0.5 * y3
    c1 = -0.5 * y0 + y1 - y2 / 6.0 - y3 / 3.0
    c0 = y0

    disc = max(c2 * c2 - 3.0 * c3 * c1, 0.0)
    lam_star = (-c2 - np.sqrt(disc)) / (3.0 * c3)
    lam_star = float(np.clip(lam_star, 0.001, 0.999))
    max_det = c3 * lam_star**3 + c2 * lam_star**2 + c1 * lam_star + c0
    return lam_star, max_det


def extract_contact_geometry(
    Q1: Quadric,
    Q2: Quadric,
    lam_star: float,
) -> tuple[Point, Plane]:
    """Extract contact point p* and contact normal plane L* at parameter λ*."""
    Q_star = Q1 * (1.0 - lam_star) + Q2 * lam_star
    _, tangents = decompose_quadric(Q_star)
    L_star = tangents[-1]
    p_star = Q1(L_star)
    return p_star, L_star


def resolve_collision(
    body1: SphericalBody,
    body2: SphericalBody,
    lam_star: float,
    M_rel: Motor,
    Q2_in_1: Quadric,
    restitution: float = 1.0,
) -> bool:
    """Detect and resolve elastic collision impulse entirely inside Geometric Algebra."""
    contact_point, contact_plane = extract_contact_geometry(body1.Q, Q2_in_1, lam_star)
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
                    lam_star, max_det = find_contact_parameter(body_i.Q, Q2_in_1)
                    if max_det < 0.0:
                        resolve_collision(body_i, body_j, lam_star, M_rel=M_rel, Q2_in_1=Q2_in_1, restitution=1.0)

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
