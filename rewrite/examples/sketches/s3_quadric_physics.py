"""Spherical quadric physics on the 3-sphere, Cl(4): the S² engine one dimension up, rendered by projection.

Bodies are dual quadric caps, rigid motion is a motor (a rotation of R⁴), the inertia of a body is the
sum over its mass points of the regressive product of the point with its own motion under an open
bivector rate, and contact between two bodies is read off the dual pencil (1-λ) Q₁ + λ Q₂: the pencil's
determinant peaks negative when the caps overlap, and the plane of the degenerate pencil member at the
peak is the contact plane. All of that is the S² example's code with the algebra swapped. The one line
that knew its dimension was the fit of the pencil determinant, a cubic for 3×3 quadrics; here it is a
quartic, so it is fitted at degree n. Frames are a wide pinhole view from the eye, every cap projected onto the image sphere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from numga import Extensor

from examples import PLOT_DIR
from examples.animation import save_gif
from examples.sketches.spherical_raytracer import DualQuadric, Motor, Plane, Point, Quadric, direction, ga, mv, origin, pixel_chart, render, unit

Bivector = ga.gatype.bivector()
Inertia = ga.gatype((Bivector, Bivector))      # momentum <= rate; in four dimensions the antibivector is a bivector
Scalar = ga.gatype.scalar()
Form = ga.gatype((Scalar, Point, Point))       # a quadric with both of its points open: scalar <= point, point


@dataclass
class Bodies:
    """The whole population as batched extensors: one leading axis, one entry per body."""
    color: np.ndarray
    motor: Motor
    momentum: Bivector
    Q: DualQuadric                                # the body's shape in its own frame, dual: point <= plane
    C: Quadric                                    # and primal: polar plane <= point, negative inside
    I_inv: Inertia
    reach: np.ndarray                             # angular radius of each bounding cap, for the broad phase

    @staticmethod
    def join(*groups: Bodies) -> Bodies:
        return Bodies(
            np.concatenate([g.color for g in groups]),
            Extensor.concatenate([g.motor for g in groups]),
            Extensor.concatenate([g.momentum for g in groups]),
            Extensor.concatenate([g.Q for g in groups]),
            Extensor.concatenate([g.C for g in groups]),
            Extensor.concatenate([g.I_inv for g in groups]),
            np.concatenate([g.reach for g in groups]),
        )


# --- plumbing -------------------------------------------------------------------------
def eigenpairs(form: Form) -> tuple[np.ndarray, Point]:
    """The eigenvalues, ascending, and the eigenvectors as points of a symmetric form on points."""
    values, vectors = np.linalg.eigh(form.kernel[..., 0, :, :])
    return values, mv(Point, np.swapaxes(vectors, -1, -2))


def filled(Q: DualQuadric, mass: np.ndarray, count: int, rng: np.random.Generator) -> tuple[Point, np.ndarray]:
    """Mass points filling the inside of any quadric, where its form is negative. In the form's
    eigenbasis the inside is where the negative block outweighs the positive one, so a point is a
    core direction in the negative block, an extent direction in the positive block, both uniform
    on their spheres, and the angle between them, up to the angle where the blocks balance; the
    angle is drawn uniformly and weighted by the sphere's measure, cosᵏ⁻¹ sin³⁻ᵏ for k core axes,
    so the weighted points are uniform in the inside. Batched over quadrics of one signature."""
    values, principal = eigenpairs(Point & Q.inverse()(mv.rotor() >> Point))   # the negative block comes first
    k = int((values < 0.0).sum(axis=-1).ravel()[0])                # core axes, the same across the batch
    core = rng.normal(size=values.shape[:-1] + (count, k))
    extent = rng.normal(size=values.shape[:-1] + (count, 4 - k))
    core, extent = core / np.linalg.norm(core, axis=-1, keepdims=True), extent / np.linalg.norm(extent, axis=-1, keepdims=True)
    balance = np.arctan(np.sqrt(np.einsum("...c,...nc->...n", -values[..., :k], core**2) / np.einsum("...e,...ne->...n", values[..., k:], extent**2)))
    angle = rng.uniform(size=balance.shape) * balance
    weight = np.cos(angle) ** (k - 1) * np.sin(angle) ** (3 - k) * balance
    coordinates = np.concatenate([np.cos(angle)[..., None] * core, np.sin(angle)[..., None] * extent], axis=-1)
    return (principal[..., None, :] * coordinates).sum(axis=-1), weight * mass[..., None] / weight.sum(axis=-1, keepdims=True)


def placed(place: Point, spin: Bivector) -> Motor:
    """Motors carrying the origin to the given points, spun by the given rotation bivectors."""
    return (unit(place) / origin).square_root() * (spin * 0.5).exp()


def body(color: np.ndarray, Q: DualQuadric, motor: Motor, rate: Bivector, mass: np.ndarray, rng: np.random.Generator) -> Bodies:
    """Bodies of any quadric shape: inertia from mass points filling the quadric, momentum from the
    body-frame rate, and the reach of the mass points from the pole for the broad phase."""
    pts, masses = filled(Q, mass, 400, rng)
    inertia = pointcloud_inertia(pts, masses)
    reach = np.arccos(np.clip(np.abs((pts | origin).kernel[..., 0]), 0.0, 1.0)).max(axis=-1)
    return Bodies(color, motor, inertia(rate), Q, Q.inverse(), inertia.inverse(), reach)


def candidates_near(bodies: Bodies) -> tuple[np.ndarray, np.ndarray]:
    """Broad phase: the pairs whose poles are closer than their reaches summed."""
    poles = bodies.motor >> origin
    apart = np.arccos(np.clip(np.abs((poles.reshape(-1, 1) | poles).kernel[..., 0]), 0.0, 1.0))
    return np.nonzero(np.triu(apart < bodies.reach[:, None] + bodies.reach[None, :], 1))


# --- math -----------------------------------------------------------------------------
def pointcloud_inertia(pts: Point, masses: np.ndarray) -> Inertia:
    """Momentum of the cloud under an open rate: each point's join with its own motion, mass-weighted."""
    return (pts.regressive(pts.commutator(Bivector)) * masses).sum(axis=-1)


def quadric(diagonal: np.ndarray) -> DualQuadric:
    """The dual quadric diagonal on the basis points x, y, z and the origin: the sum of each point
    paired with itself, weighted; negative weights mark the inside's core, positive ones its extents."""
    basis = mv.antivector(np.eye(4)).normalized()
    return (basis * (basis & Plane) * diagonal).sum(axis=-1)


def cap_quadric(half_widths: np.ndarray) -> DualQuadric:
    """The cap around the origin with the given half-widths tan(θ) along the axes."""
    return quadric(np.concatenate([half_widths**2, -np.ones_like(half_widths[..., :1])], axis=-1))


def step_motor(motor: Motor, momentum: Bivector, I_inv: Inertia, dt: float) -> tuple[Motor, Bivector]:
    """Advance a body by dt with a Lie midpoint step; momentum is kept in the body frame."""
    half = (motor * (I_inv(momentum) * (0.25 * dt)).exp()).normalized()
    rate = I_inv((motor.inverse() * half) << momentum)
    step = (rate * (0.5 * dt)).exp()
    moved = (motor * step).normalized()
    return moved, (motor.inverse() * moved) << momentum


def overlap(A: Quadric, B: Quadric, iterations: int = 12) -> tuple[np.ndarray, Point]:
    """Whether the insides of two quadrics, where their forms are negative, meet: they are apart if
    and only if some member of the pencil A + λB, λ > 0, is positive semidefinite (the S-lemma), so
    the largest over the pencil of the least eigenvalue is negative exactly when they overlap. The
    least eigenvalue is concave in λ, hence unimodal in φ = arctan λ over (0, π/2), and golden
    section finds its maximum; the least eigenvector there is the deepest point, the touching
    point when the margin is zero. Twelve iterations bracket φ to 0.005 rad; five misreport near
    pairs as touching. Batched over pairs."""
    def least(phi: np.ndarray) -> np.ndarray:
        member = Point & (A + B * np.tan(phi))(mv.rotor() >> Point)   # the pencil member at λ = tan φ, both points open
        return eigenpairs(member)[0][..., 0]

    # Golden section on φ: two probes c < d split the bracket [lo, hi] in the golden ratio; whichever
    # side holds the larger value keeps the bracket, and the surviving probe already sits at the
    # golden point of the new bracket, so each step evaluates one fresh probe only.
    golden = (np.sqrt(5.0) - 1.0) / 2.0
    lo, hi = np.broadcast_to(0.0, np.broadcast_shapes(A.shape, B.shape)), np.broadcast_to(np.pi / 2, np.broadcast_shapes(A.shape, B.shape))
    c, d = hi - golden * (hi - lo), lo + golden * (hi - lo)
    fc, fd = least(c), least(d)
    for _ in range(iterations):
        left = fc > fd                                            # the maximum lies in [lo, d], else in [c, hi]
        lo, hi = np.where(left, lo, c), np.where(left, d, hi)     # shrink the bracket to that side
        c, d = hi - golden * (hi - lo), lo + golden * (hi - lo)   # new probes; one coincides with the survivor
        fresh = least(np.where(left, c, d))                       # evaluate only the other
        fc, fd = np.where(left, fresh, fd), np.where(left, fc, fresh)   # survivor's value moves to its new slot
    values, points = eigenpairs(Point & (A + B * np.tan((lo + hi) / 2))(mv.rotor() >> Point))
    return values[..., 0], points[..., 0]


def collide(bodies: Bodies, i: np.ndarray, j: np.ndarray, dt: float, restitution: float = 1.0) -> int:
    """Contact test and elastic impulses for the candidate pairs (i, j), each in the frame of its first body.

    The overlap margin is negative while the bodies meet. An impulse is applied only while the
    overlap deepens, judged by a virtual step of all bodies along their current rates, so no
    orientation of the contact wrench is ever assumed; the impulse reflects the closing rate along
    the wrench whatever its sign. The geometry is batched over the pairs; the impulses go one pair
    at a time, each against the momenta the earlier ones left. Returns the number applied.
    """
    def margins(motors: Motor) -> tuple[np.ndarray, Point, Motor]:
        relative = motors[i].inverse() * motors[j]
        return *overlap(bodies.C[i], relative >> bodies.C[j](relative << Point)), relative

    margin, deepest, relative = margins(bodies.motor)
    ahead = (bodies.motor * (bodies.I_inv(bodies.momentum) * (0.5 * dt)).exp()).normalized()
    touching = (margin < 0.0) & (margins(ahead)[0] < margin)
    # The first body's polar plane at the deepest point is the contact plane, and its pole is the
    # contact point. The wrench is the line through the contact point normal to the contact
    # plane: their inner product.
    contact_plane = bodies.C[i](deepest).normalized()
    contact_point = bodies.Q[i](contact_plane)
    wrench_one = contact_plane | contact_point
    wrench_other = relative << wrench_one
    for a, b, one, other in zip(i[touching], j[touching], wrench_one[touching], wrench_other[touching]):
        response_one, response_other = bodies.I_inv[a](one), bodies.I_inv[b](other)
        closing = response_one.regressive(bodies.momentum[a]) - response_other.regressive(bodies.momentum[b])
        compliance = one.regressive(response_one) + other.regressive(response_other)
        impulse = (-closing * (1.0 + restitution)) / compliance
        bodies.momentum = bodies.momentum.at[a].set(bodies.momentum[a] + one * impulse).at[b].set(bodies.momentum[b] - other * impulse)
    return int(touching.sum())


def energy(bodies: Bodies) -> float:
    return 0.5 * bodies.I_inv(bodies.momentum).regressive(bodies.momentum).kernel.sum()


def population(rng: np.random.Generator, count: int, candidates: int, sizes: tuple[float, float], *seeded: Bodies, drift: float = 0.6) -> Bodies:
    """The seeded bodies plus `count` random caps out of `candidates` draws: half-widths log-uniform
    per axis between the sizes, so needles, discs and blobs; placed uniformly on the 3-sphere; kept
    where they overlap neither the seeded bodies nor any kept before them (one batched overlap test
    over the pairs within reach, then a greedy pass over its boolean matrix). Tennis-racket rates:
    spin about the intermediate axis, the middle half-width, whose bivector is the plane it is
    normal to (axis x -> yz, y -> zx, z -> xy), plus a nudge and a drift across the sphere."""
    half_widths = 10 ** rng.uniform(np.log10(sizes[0]), np.log10(sizes[1]), size=(candidates, 3))
    rate = rng.normal(size=(candidates, 6)) * np.array([0.3, 0.3, 0.3, drift, drift, drift])
    rate[np.arange(candidates), np.argsort(half_widths, axis=-1)[:, 1]] = rng.choice([-5.0, 5.0], size=candidates)
    place, spin = mv.antivector(rng.normal(size=(candidates, 4))), mv.bivector(rng.normal(size=(candidates, 6)) * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]))
    drawn = body(plt.get_cmap("hsv")(rng.permutation(candidates) / candidates)[:, :3], cap_quadric(half_widths), placed(place, spin), mv.bivector(rate), np.prod(half_widths, axis=-1) * 200.0, rng)
    everyone = Bodies.join(*seeded, drawn)
    given = len(everyone.reach) - candidates
    i, j = candidates_near(everyone)
    relative = everyone.motor[i].inverse() * everyone.motor[j]
    overlapping = np.zeros((len(everyone.reach),) * 2, dtype=bool)
    overlapping[i, j] = overlapping[j, i] = overlap(everyone.C[i], relative >> everyone.C[j](relative << Point))[0] < 0.0
    keep = list(range(given))
    for candidate in range(given, len(everyone.reach)):
        if not overlapping[candidate, keep].any():
            keep.append(candidate)
    assert len(keep) >= given + count, f"only {len(keep) - given} admissible non-overlapping candidates"
    keep = np.array(keep[:given + count])
    return Bodies(everyone.color[keep], everyone.motor[keep], everyone.momentum[keep], everyone.Q[keep], everyone.C[keep], everyone.I_inv[keep], everyone.reach[keep])


def run(
    bodies: Bodies, view: Callable[[Bodies], tuple[Motor, Point]],
    frames: int, chart: Point, shape: tuple[int, int], supersample: int, dt: float = 0.02, substeps: int = 8,
) -> tuple[list[np.ndarray], int, list[float]]:
    """Integrate, collide and render: per frame `view` gives the eye frame and the light from the
    current bodies. Returns the frames, the impulse count, and the energy per frame."""
    def snapshot() -> np.ndarray:
        eye, light = view(bodies)
        world = bodies.motor >> bodies.Q(bodies.motor << Plane)
        return (render(eye, world, world.inverse(), bodies.color, light, chart, shape, supersample) * 255).astype(np.uint8)

    frames_out, collisions, energies = [snapshot()], 0, [energy(bodies)]
    for step in range(frames - 1):
        for _ in range(substeps):
            bodies.motor, bodies.momentum = step_motor(bodies.motor, bodies.momentum, bodies.I_inv, dt / substeps)
            collisions += collide(bodies, *candidates_near(bodies), dt / substeps)
        energies.append(energy(bodies))
        frames_out.append(snapshot())
    print(f"{collisions} collisions in {len(frames_out)} frames; energy drift {np.ptp(energies) / energies[0]:.2e}")
    return frames_out, collisions, energies


def main(
    plot_path: str = str(PLOT_DIR / "sketch_s3_quadric_physics.png"),
    animation_path: str = str(PLOT_DIR / "sketch_s3_quadric_physics.gif"),
    shape: tuple[int, int] = (180, 240),
    supersample: int = 4,
    frames: int = 240,
) -> plt.Figure:
    rng = np.random.default_rng(3)
    bodies = population(rng, 28, 120, (0.05, 0.5))

    # The light: a point 0.8 rad from the eye, above and behind it, 70° off the line of sight so it
    # sits just outside the 120° frustum (the image corners reach 65°) in either direction.
    light = unit(((mv.xw * -0.34 + mv.zw * 0.94) * (0.8 / 2)).exp() >> origin)
    # A 120° pinhole view from the eye; with bodies all over the 3-sphere, whatever drifts
    # behind the eye reappears ahead near the antipode, which is what the sphere looks like.
    chart = pixel_chart(np.radians(120.0), (shape[0] * supersample, shape[1] * supersample))
    frames_out, collisions, energies = run(bodies, lambda bodies: (mv.rotor(), light), frames, chart, shape, supersample)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.imshow(frames_out[-1], interpolation="nearest"); ax.axis("off"); ax.set_title(f"{len(bodies.reach)} caps on the 3-sphere after {len(frames_out)} frames")
    if plot_path:
        fig.savefig(plot_path, bbox_inches="tight")
        print(f"Figure saved to {plot_path}")
    if animation_path:
        save_gif(frames_out, animation_path, duration_ms=50)

    # --- checks: kernel-level assertions, deliberately outside the demonstration ----------
    assert collisions > 0
    assert np.ptp(energies) / energies[0] < 2e-2                  # elastic impulses and a symplectic step
    return fig


if __name__ == "__main__":
    main()
