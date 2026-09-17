"""Pose filtering on the motor manifold in PGA2D: the covariance is a map on bivectors.

The state is a motor; its uncertainty is a covariance over body-frame bivector perturbations,
which is a unary extensor bivector -> bivector. Prediction transports it with the adjoint of
the step, the sandwich with a bivector hole, and the update solves with the map inverse. The
3x3 Jacobians of the motion and measurement models are never derived by hand.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import Extensor, NumpyContext
from numga.algebras import PGA2D

from examples import PLOT_DIR

ga = PGA2D
ctx = NumpyContext(ga)
mv = ctx.multivector
B = ga.subspace.bivector()
P = ga.subspace.antivector()
Point = ga.gatype.antivector()
Bivector = ga.gatype.bivector()
Covariance = ga.gatype((B, B))
Position = ga.gatype.from_blades("yw wx")
PositionCovariance = ga.gatype((Position, Position))


# --- plumbing -------------------------------------------------------------------------
def covariance(std: np.ndarray) -> Covariance:
    """Diagonal covariance over the bivector coordinates (yw, wx, xy)."""
    return Extensor(ctx, Covariance, np.diag(std ** 2))


def sample(cov: Covariance, rng: np.random.Generator) -> Bivector:
    """One bivector drawn from a covariance map."""
    return cov.cholesky()(mv.bivector(rng.normal(size=3)))


def xy(point: Point) -> np.ndarray:
    k = point.cast(P).kernel
    return k[..., :2] / k[..., 2:]


def draw_ellipse(ax, centre: np.ndarray, cov: PositionCovariance, color: str) -> None:
    values, vectors = cov.eigh()
    values, vectors = values.kernel[..., 0], vectors.kernel.T
    t = np.linspace(0.0, 2.0 * np.pi, 40)
    ring = vectors @ (2.0 * np.sqrt(np.maximum(values, 0.0))[:, None] * np.stack([np.cos(t), np.sin(t)]))
    ax.plot(centre[0] + ring[0], centre[1] + ring[1], color=color, linewidth=0.8)


def draw_tracking(track, measurements, steps: int, dt: float, plot_path: str) -> plt.Figure:
    columns = tuple(zip(*track))
    true_xy, dead_xy, est_xy = (np.array([xy(value) for value in column]) for column in columns[:3])
    covs = Extensor.stack(columns[3])
    print(f"mean position error, dead reckoning: {np.linalg.norm(dead_xy - true_xy, axis=1).mean():.3f}")
    print(f"mean position error, filtered:       {np.linalg.norm(est_xy - true_xy, axis=1).mean():.3f}")

    fig, (ax, ax_err) = plt.subplots(1, 2, figsize=(11, 5), dpi=120)
    ax.plot(*true_xy.T, color="black", linewidth=1.5, label="truth")
    ax.plot(*dead_xy.T, color="tab:red", linestyle="--", linewidth=1, label="dead reckoning")
    ax.plot(*est_xy.T, color="tab:blue", linewidth=1, label="filtered")
    for k, measured in measurements:
        measured_xy = xy(measured)
        ax.plot(*measured_xy, marker="x", color="tab:green", linestyle="none")
        draw_ellipse(ax, est_xy[k], covs[k], "tab:blue")
    ax.plot([], [], marker="x", color="tab:green", linestyle="none", label="pose measurements")
    ax.set_aspect("equal"); ax.legend(loc="upper left", fontsize=8)
    ax.set_title("paths, with the 2σ position ellipse at each measurement")
    time = np.arange(steps) * dt
    ax_err.plot(time, np.linalg.norm(dead_xy - true_xy, axis=1), color="tab:red", linestyle="--", label="dead reckoning")
    ax_err.plot(time, np.linalg.norm(est_xy - true_xy, axis=1), color="tab:blue", label="filtered")
    ax_err.plot(time, 2 * np.sqrt(covs.eigvalsh().kernel[..., 0].max(axis=1)), color="tab:blue", linestyle=":", label="filter's own 2σ")
    for k, _ in measurements:
        ax_err.axvline(k * dt, color="tab:green", linewidth=0.5, alpha=0.5)
    ax_err.set_xlabel("time"); ax_err.set_ylabel("position error"); ax_err.legend(fontsize=8)
    ax_err.set_title("error over time; green lines are measurements")
    if plot_path:
        fig.savefig(plot_path, bbox_inches="tight")
        print(f"Figure saved to {plot_path}")
    return fig


# --- math -----------------------------------------------------------------------------
def main(plot_path: str = str(PLOT_DIR / "sketch_kalman.png")) -> plt.Figure:
    rng = np.random.default_rng(2)
    dt, steps, every = 0.1, 300, 25
    # Body-frame command: drive 1 m/s along +x (exp(-wx) moves +x) while the turn rate
    # wanders, so the path meanders instead of retracing itself.
    turn = 0.9 * np.sin(0.2 * np.arange(steps) * dt)
    Q = covariance(np.array([0.05, 0.05, 0.12]) * np.sqrt(dt))
    R = covariance(np.array([0.3, 0.3, 0.1]))

    origin = mv.xy
    truth = estimate = dead = mv.rotor()
    sigma = covariance(np.zeros(3))
    track: list[tuple[Point, Point, Point, PositionCovariance]] = []
    measurements: list[tuple[int, Point]] = []
    for k in range(steps):
        control = mv.xy * turn[k] - mv.wx * 1.0
        truth = truth * ((control * dt + sample(Q, rng)) * 0.5).exp()
        step = (control * (dt / 2)).exp()
        dead = dead * step

        # Predict. A perturbation on the right of the estimate, m exp(δ/2), is carried through
        # the step by step⁻¹ δ step: the adjoint of the step is its sandwich with a hole.
        estimate = estimate * step
        adjoint = step << B
        sigma = adjoint(sigma(adjoint.transpose())) + Q

        # Update on a noisy pose measurement. The innovation is the log of the relative
        # motor, the gain is a ratio of covariance maps, and the correction is exponentiated.
        if k % every == every - 1:
            measured = truth * (sample(R, rng) * 0.5).exp()
            innovation = (estimate.inverse() * measured).log() * 2
            gain = sigma((sigma + R).inverse())
            estimate = estimate * (gain(innovation) * 0.5).exp()
            sigma = sigma - gain(sigma)
            measurements.append((k, measured >> origin))

        # Position uncertainty. A body-frame perturbation moves the position point by the
        # commutator of its world image with that point: a linear map from bivectors to ideal
        # points. Pushing the covariance through it gives the position covariance.
        here = estimate >> origin
        shift = B.commutator(here)(estimate >> B)
        track.append((truth >> origin, dead >> origin, here, shift(sigma(shift.transpose()))))

    return draw_tracking(track, measurements, steps, dt, plot_path)


if __name__ == "__main__":
    main()
