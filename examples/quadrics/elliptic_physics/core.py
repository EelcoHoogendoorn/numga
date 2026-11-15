"""Rigid quadric bodies spinning and colliding on the 2-sphere and 3-sphere.

Each body is modelled as a dual quadric placed on the sphere by a motor, with its inertia formed over mass points inside the shape as the regressive product of each mass point with its velocity under a bivector rate. Body-frame momenta advance by a Lie midpoint step, while candidate pairs are tested for overlap through the concave least eigenvalue of positive blends of their primal forms; elastic impulses along the contact forque at the deepest point reverse the closing rate.

Instantiating this module on a 3D or 4D positive-definite Euclidean algebra supplies the same rigid-body engine on the 2-sphere and on the 3-sphere.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, concatenate
from numga.backend.context import Context

# Algebra and context supplied by `examples.instantiate`.
ga: Algebra
ctx: Context
mv = ctx.multivector

Scalar = ga.gatype.scalar()                                                     # Scalar+
Point = ga.gatype.antivector()                                                  # Antivector-
Plane = ga.gatype.vector()                                                      # Vector+
Rate = ga.gatype.bivector()                                                     # Bivector+
Momentum = ga.gatype.antibivector()                                             # Antibivector-
Motor = ga.gatype.rotor()                                                       # Rotor+
# A primal quadric maps a point to its polar plane and evaluates negative inside; a dual quadric maps a plane to its pole.
Quadric = ga.gatype((Plane, Point))                                             # Plane+ <- Point-
DualQuadric = ga.gatype((Point, Plane))                                         # Point- <- Plane+
Inertia = ga.gatype((Momentum, Rate))                                           # Momentum- <- Rate+
InverseInertia = ga.gatype((Rate, Momentum))                                    # Rate+ <- Momentum-

# Golden-section iterations bracketing the optimal quadric blend angle to within 0.005 rad.
n_golden_steps = 12

# Unit poles of the coordinate planes, with the final pole serving as each body's reference centre.
basis = mv.vector(np.eye(ga.dimension)).dual().normalized()                     # [...] Point-
origin = basis[-1]                                                              # [] Point-


# --- math -----------------------------------------------------------------------------
def diagonal_quadric(diagonal: np.ndarray) -> DualQuadric:
    """Coordinate poles weighted by `diagonal` (`[..., n_axes]`) sum into diagonal dual quadrics (`[...] Point- <- Plane+`)."""
    return (basis * (basis & Plane) * diagonal).sum(axis=-1)                    # [...] Point- <- Plane+


def ellipsoid_quadric(half_widths: np.ndarray) -> DualQuadric:
    """Tangent `half_widths` (`[..., n_axes - 1]`) around `origin` form central ellipsoid dual quadrics (`[...] Point- <- Plane+`)."""
    weights = np.concatenate([half_widths**2, -np.ones_like(half_widths[..., :1])], axis=-1)
    return diagonal_quadric(weights)                                            # [...] Point- <- Plane+


def cloud_inertia(points: Point, masses: Scalar) -> Inertia:
    """Mass-weighted regressive products of `points` (`[..., n_samples] Point-`) with their commutator velocities form body inertias (`[...] Momentum- <- Rate+`)."""
    return ((masses * points).regressive(points.commutator(Rate))).sum(axis=-1)  # [...] Momentum- <- Rate+


def world_quadrics(motors: Motor, forms: Quadric) -> Quadric:
    """Sandwiching body-frame `forms` (`[...] Plane+ <- Point-`) by `motors` (`[...] Motor+`) places the primal quadrics in the world frame."""
    return motors >> forms(motors << Point)                                     # [...] Plane+ <- Point-


def kinetic_energy(rates: Rate, momenta: Momentum) -> Scalar:
    """Half the regressive pairing of body-frame `rates` (`[n_bodies] Rate+`) with `momenta` (`[n_bodies] Momentum-`) sums to total kinetic energy (`[] Scalar+`)."""
    return (rates & momenta).sum() / 2                                          # [] Scalar+


def world_momentum(motors: Motor, momenta: Momentum) -> Momentum:
    """Carrying body-frame `momenta` (`[n_bodies] Momentum-`) by `motors` (`[n_bodies] Motor+`) sums to total world-frame momentum (`[] Momentum-`)."""
    return (motors >> momenta).sum(axis=0)                                      # [] Momentum-


def midpoint_step(
    motors: Motor,
    momenta: Momentum,
    inverse_inertias: InverseInertia,
    dt: float,
) -> tuple[Motor, Momentum]:
    """Advancing `motors` and body-frame `momenta` by `dt` with a Lie midpoint step updates placements and body momenta (`[n_bodies] Motor+, [n_bodies] Momentum-`)."""
    # Step halfway along the present body rate, evaluate the midpoint rate, and advance across the full step.
    half_step = (inverse_inertias(momenta) * (dt / 4)).exp()                    # [...] Motor+
    midpoint = (motors * half_step).normalized()                                # [...] Motor+
    midpoint_rate = inverse_inertias((motors.inverse() * midpoint) << momenta)  # [...] Rate+
    whole_step = (midpoint_rate * (dt / 2)).exp()                               # [...] Motor+
    next_motors = (motors * whole_step).normalized()                            # [...] Motor+
    return next_motors, (motors.inverse() * next_motors) << momenta


def quadric_overlap(first: Quadric, second: Quadric) -> tuple[Scalar, Point]:
    """Golden-section search over positive blends of `first` and `second` (`[...] Plane+ <- Point-`) finds the overlap margin (`[...] Scalar+`) and deepest point (`[...] Point-`).

    By the S-lemma, two quadric interiors where both quadratic forms are negative are disjoint if and only if some positive blend is positive semidefinite; maximizing the concave least eigenvalue over blend angle therefore yields a negative margin exactly on overlap.
    """
    def least_eigenvalue(phi: np.ndarray) -> Scalar:
        blend = Point & (first + second * np.tan(phi))(Point)                   # [...] Scalar+ <- (Point-, Point-)
        return blend.eigvalsh()[..., 0]                                         # [...] Scalar+

    golden = (np.sqrt(5.0) - 1.0) / 2.0
    pair_shape = np.broadcast_shapes(first.shape, second.shape)
    lower, upper = np.broadcast_to(0.0, pair_shape), np.broadcast_to(np.pi / 2, pair_shape)
    left_probe = upper - golden * (upper - lower)
    right_probe = lower + golden * (upper - lower)
    left_value, right_value = least_eigenvalue(left_probe), least_eigenvalue(right_probe)  # [...] Scalar+, [...] Scalar+
    for _ in range(n_golden_steps):
        keep_left = left_value > right_value
        lower = np.where(keep_left, lower, left_probe)
        upper = np.where(keep_left, right_probe, upper)
        left_probe = upper - golden * (upper - lower)
        right_probe = lower + golden * (upper - lower)
        fresh = least_eigenvalue(np.where(keep_left, left_probe, right_probe))  # [...] Scalar+
        left_value, right_value = (
            right_value.at[keep_left].set(fresh[keep_left]),
            fresh.at[keep_left].set(left_value[keep_left]),
        )
    eigenvalues, eigenvectors = (Point & (first + second * np.tan((lower + upper) / 2))(Point)).eigh()
    return eigenvalues[..., 0], eigenvectors[..., 0]


def candidate_pairs(motors: Motor, bounding_radii: Scalar) -> tuple[np.ndarray, np.ndarray]:
    """Geodesic distances between body centres (`[n_bodies] Motor+`) filter upper-triangular index pairs within the sum of `bounding_radii` (`[n_bodies] Scalar+`)."""
    poles = motors >> origin                                                    # [n_bodies] Point-
    separations = (poles[:, None] | poles).abs().clip(0.0, 1.0).arccos()        # [n_bodies, n_bodies] Scalar+
    within_reach = separations < bounding_radii[:, None] + bounding_radii[None, :]
    return np.nonzero(np.triu(within_reach, 1))


def contact_forques(
    first_forms: Quadric,
    first_duals: DualQuadric,
    second_forms: Quadric,
    first_motors: Motor,
    second_motors: Motor,
) -> tuple[Scalar, Momentum, Momentum]:
    """Overlap detection between `first_forms` and `second_forms` returns the margin (`[...] Scalar+`) and the contact forques (`[...] Momentum-`) in both body frames."""
    relative = first_motors.inverse() * second_motors                           # [...] Motor+
    margin, deepest = quadric_overlap(first_forms, relative >> second_forms(relative << Point))  # [...] Scalar+, [...] Point-
    # Contracting the first body's unit polar plane at the deepest point with its pole forms the normal contact line.
    contact_plane = first_forms(deepest).normalized()                           # [...] Plane+
    contact_point = first_duals(contact_plane)                                  # [...] Point-
    first_forque = contact_plane | contact_point                                # [...] Momentum-
    second_forque = relative << first_forque                                    # [...] Momentum-
    return margin, first_forque, second_forque


def elastic_collision(
    motors: Motor,
    momenta: Momentum,
    forms: Quadric,
    duals: DualQuadric,
    inverse_inertias: InverseInertia,
    first_indices: np.ndarray,
    second_indices: np.ndarray,
    dt: float,
) -> tuple[Momentum, int]:
    """Sequential elastic impulses across approaching overlapping pairs `(first_indices, second_indices)` update `momenta` (`[n_bodies] Momentum-`) and count applied impulses."""
    margin, first_forque, second_forque = contact_forques(
        forms[first_indices],
        duals[first_indices],
        forms[second_indices],
        motors[first_indices],
        motors[second_indices],
    )
    # Compare against a virtual half-step forward so impulses act only while an overlap is deepening, independent of forque sign.
    half_step = (inverse_inertias(momenta) * (dt / 2)).exp()                    # [n_bodies] Motor+
    ahead = (motors * half_step).normalized()                                   # [n_bodies] Motor+
    relative_ahead = ahead[first_indices].inverse() * ahead[second_indices]     # [...] Motor+
    margin_ahead, _ = quadric_overlap(
        forms[first_indices],
        relative_ahead >> forms[second_indices](relative_ahead << Point),
    )
    approaching = (margin < 0) & (margin_ahead < margin)
    for first_body, second_body, one, other in zip(
        first_indices[approaching],
        second_indices[approaching],
        first_forque[approaching],
        second_forque[approaching],
    ):
        first_response = inverse_inertias[first_body](one)                      # [] Rate+
        second_response = inverse_inertias[second_body](other)                  # [] Rate+
        closing = (first_response & momenta[first_body]) - (second_response & momenta[second_body])  # [] Scalar+
        compliance = ( first_response & one ) + ( second_response & other )     # [] Scalar+
        impulse = -2 * closing / compliance                                     # [] Scalar+
        momenta = (
            momenta.at[first_body]
            .set(momenta[first_body] + one * impulse)
            .at[second_body]
            .set(momenta[second_body] - other * impulse)
        )
    return momenta, int(approaching.sum())


def collision_frames(
    motors: Motor,
    momenta: Momentum,
    forms: Quadric,
    duals: DualQuadric,
    inverse_inertias: InverseInertia,
    bounding_radii: Scalar,
    n_frames: int,
    dt: float,
    n_substeps: int,
) -> Iterator[tuple[Motor, Momentum, int]]:
    """Yields `n_frames` states `(motors, momenta, cumulative_impulses)` separated by `n_substeps` midpoint and collision substeps across `dt`."""
    substep_dt = dt / n_substeps
    impulses = 0
    for _ in range(n_frames):
        yield motors, momenta, impulses
        for _ in range(n_substeps):
            motors, momenta = midpoint_step(motors, momenta, inverse_inertias, substep_dt)
            first_indices, second_indices = candidate_pairs(motors, bounding_radii)
            momenta, applied = elastic_collision(
                motors, momenta, forms, duals, inverse_inertias, first_indices, second_indices, substep_dt,
            )
            impulses += applied


# --- plumbing -------------------------------------------------------------------------
def body_state(
    duals: DualQuadric,
    motors: Motor,
    rates: Rate,
    points: Point,
    masses: Scalar,
) -> tuple[Motor, Momentum, DualQuadric, Quadric, InverseInertia, Scalar]:
    """Mass clouds (`[n_bodies, n_samples] Point-`, `[n_bodies, n_samples] Scalar+`) determine initial momenta, primal forms, inverse inertias, and bounding radii for `duals`."""
    inertias = cloud_inertia(points, masses)                                    # [n_bodies] Momentum- <- Rate+
    momenta = inertias(rates)                                                   # [n_bodies] Momentum-
    forms = duals.inverse()                                                     # [n_bodies] Plane+ <- Point-
    inverse_inertias = inertias.inverse()                                       # [n_bodies] Rate+ <- Momentum-
    bounding_radii = (points | origin).abs().clip(0.0, 1.0).arccos().max(axis=-1)  # [n_bodies] Scalar+
    return motors, momenta, duals, forms, inverse_inertias, bounding_radii


def interior_samples(
    duals: DualQuadric,
    masses: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[Point, Scalar]:
    """Stratified eigenbasis directions inside `duals` (`[n_bodies] Point- <- Plane+`) form `n_samples` interior points (`[n_bodies, n_samples] Point-`) carrying total `masses` (`[n_bodies]`)."""
    n_axes = ga.dimension
    eigenvalues, principal = (Point & duals.inverse()).eigh()                   # [n_bodies, n_axes] Scalar+, [n_bodies, n_axes] Point-
    n_core = int((eigenvalues < 0)[0].sum())
    core = rng.normal(size=eigenvalues.shape[:-1] + (n_samples, n_core))
    extent = rng.normal(size=eigenvalues.shape[:-1] + (n_samples, n_axes - n_core))
    core = core / np.linalg.norm(core, axis=-1, keepdims=True)
    extent = extent / np.linalg.norm(extent, axis=-1, keepdims=True)
    inward = (-eigenvalues[..., None, :n_core] * core**2).sum(axis=-1)          # [n_bodies, n_samples] Scalar+
    outward = (eigenvalues[..., None, n_core:] * extent**2).sum(axis=-1)        # [n_bodies, n_samples] Scalar+
    balance = (inward / outward).square_root().arctan()                         # [n_bodies, n_samples] Scalar+
    angle = balance * rng.uniform(size=balance.shape)                           # [n_bodies, n_samples] Scalar+
    weights = angle.cos() ** (n_core - 1) * angle.sin() ** (n_axes - 1 - n_core) * balance  # [n_bodies, n_samples] Scalar+
    coordinates = concatenate([angle.cos()[..., None] * core, angle.sin()[..., None] * extent], axis=-1)  # [n_bodies, n_samples, n_axes] Scalar+
    points = (principal[..., None, :] * coordinates).sum(axis=-1)               # [n_bodies, n_samples] Point-
    sample_masses = weights * masses[..., None] / weights.sum(axis=-1)[..., None]  # [n_bodies, n_samples] Scalar+
    return points, sample_masses
