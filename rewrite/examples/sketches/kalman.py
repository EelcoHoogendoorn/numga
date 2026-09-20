"""Pose filtering on the motor manifold in PGA2D: the covariance is a map on bivectors.

The state is a motor; its uncertainty is a covariance over body-frame bivector perturbations,
which is a unary extensor bivector -> bivector. Prediction transports it with the adjoint of
the step, the sandwich with a bivector hole, and the update solves with the map inverse. The
3x3 Jacobians of the motion and measurement models are never derived by hand.
"""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import Extensor, NumpyContext
from numga.algebras import PGA2D

from examples import PLOT_DIR

# --- scenario algebra -----------------------------------------------------------------
ga = PGA2D
ctx = NumpyContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()
Motor = ga.gatype.rotor()
Bivector = ga.gatype.bivector()
Scalar = ga.gatype.scalar()
Covariance = ga.gatype((Bivector, Bivector))


# --- math -----------------------------------------------------------------------------
def kalman_filter(estimate: Motor, sigma: Covariance, steps: Motor,
                  measurements: list[Motor], Q: Covariance, R: Covariance) -> Iterator[tuple[Motor, Covariance]]:
    """Kalman filter on motors, driven by body-frame steps and noisy full-pose measurements.

    Pose errors are right perturbations, estimate * exp(delta / 2), with sigma
    their covariance on bivectors. Q adds motion uncertainty at each prediction;
    R is the covariance of measurement error in the same perturbation convention.
    Each measurement is paired with the motion steps since the previous reading.
    Predict through those steps, then yield the corrected pose and covariance.
    """
    for prediction_steps, measured in zip(steps, measurements):
        # Predict. A perturbation on the right of the estimate, m exp(δ/2), is carried through
        # the step by step⁻¹ δ step: the adjoint of the step is its sandwich with a hole.
        for step in prediction_steps:
            estimate = estimate * step
            adjoint = step << Bivector
            sigma = adjoint(sigma(adjoint.transpose())) + Q

        # Update on a noisy pose measurement. The innovation is the log of the relative
        # motor, the gain is a ratio of covariance maps, and the correction is exponentiated.
        innovation = (estimate.inverse() * measured).log() * 2
        gain = sigma((sigma + R).inverse())
        estimate = estimate * (gain(innovation) * 0.5).exp()
        sigma = sigma - gain(sigma)

        yield estimate, sigma


# --- plumbing: covariance and sampling -------------------------------------------------
def covariance(std: np.ndarray) -> Covariance:
    """Diagonal covariance over the bivector coordinates (yw, wx, xy)."""
    return Extensor(ctx, Covariance, np.diag(std ** 2))


def sample(cov: Covariance, rng: np.random.Generator) -> Bivector:
    """One bivector drawn from a covariance map."""
    return cov.cholesky()(mv("yw wx xy", rng.normal(size=3)))


# --- plotting -------------------------------------------------------------------------
def xy(point: Point) -> np.ndarray:
    k = point.cast(ga.subspace("yw wx xy")).kernel
    return k[..., :2] / k[..., 2:]


def draw_ellipse(ax, centre: np.ndarray, values: Scalar, vectors: Point, color: str) -> None:
    values, vectors = values.kernel[..., 0], vectors.cast(ga.subspace("yw wx")).kernel.T
    t = np.linspace(0.0, 2.0 * np.pi, 40)
    ring = vectors @ (2.0 * np.sqrt(np.maximum(values, 0.0))[:, None] * np.stack([np.cos(t), np.sin(t)]))
    ax.plot(centre[0] + ring[0], centre[1] + ring[1], color=color, linewidth=0.8)


def draw_tracking(truth: Motor, dead: Motor, track, measurements: list[Motor],
                  origin: Point, times: np.ndarray, plot_path: str) -> plt.Figure:
    estimates, covariances = zip(*track)
    estimate, sigma = Extensor.stack(estimates), Extensor.stack(covariances)

    # Push pose uncertainty through the position Jacobian to draw its ellipse.
    here = estimate >> origin
    shift = Bivector.commutator(here)(estimate >> Bivector)
    values, vectors = shift(sigma(shift.transpose())).eigh()
    true_xy, dead_xy, est_xy = xy(truth >> origin), xy(dead >> origin), xy(here)
    print(f"mean position error, dead reckoning: {np.linalg.norm(dead_xy - true_xy, axis=1).mean():.3f}")
    print(f"mean position error, filtered:       {np.linalg.norm(est_xy - true_xy, axis=1).mean():.3f}")

    fig, (ax, ax_err) = plt.subplots(1, 2, figsize=(11, 5), dpi=120)
    ax.plot(*true_xy.T, color="black", linewidth=1.5, label="truth")
    ax.plot(*dead_xy.T, color="tab:red", linestyle="--", linewidth=1, label="dead reckoning")
    ax.plot(*est_xy.T, color="tab:blue", linewidth=1, label="filtered")
    measured_xy = xy(Extensor.stack(measurements) >> origin)
    ax.plot(*measured_xy.T, marker="x", color="tab:green", linestyle="none", alpha=0.4, label="pose measurements")
    for centre, variance, axes in zip(est_xy, values, vectors):
        draw_ellipse(ax, centre, variance, axes, "tab:blue")
    ax.set_aspect("equal"); ax.legend(loc="upper left", fontsize=8)
    ax.set_title("paths, with the 2σ position ellipse at each measurement")
    ax_err.plot(times, np.linalg.norm(dead_xy - true_xy, axis=1), color="tab:red", linestyle="--", label="dead reckoning")
    ax_err.plot(times, np.linalg.norm(est_xy - true_xy, axis=1), color="tab:blue", label="filtered")
    ax_err.plot(times, 2 * np.sqrt(values.kernel[..., 0].max(axis=1)), color="tab:blue", linestyle=":", label="filter's own 2σ")
    ax_err.set_xlabel("time"); ax_err.set_ylabel("position error"); ax_err.legend(fontsize=8)
    ax_err.set_title("error over time")
    if plot_path:
        fig.savefig(plot_path, bbox_inches="tight")
        print(f"Figure saved to {plot_path}")
    return fig


# --- scenario -------------------------------------------------------------------------
def simulate_motion(initial: Motor, increments: Bivector, Q: Covariance, R: Covariance,
                    rng: np.random.Generator) -> tuple[Motor, Motor, list[Motor]]:
    """Simulate noisy motion and a pose measurement after each group of increments."""
    truth = dead = initial
    true_path, dead_path, measurements = [], [], []
    for segment in increments:
        for increment in segment:
            truth = truth * ((increment + sample(Q, rng)) * 0.5).exp()
            dead = dead * (increment * 0.5).exp()
        true_path.append(truth)
        dead_path.append(dead)
        measurements.append(truth * (sample(R, rng) * 0.5).exp())
    return Extensor.stack(true_path), Extensor.stack(dead_path), measurements


def main(plot_path: str = str(PLOT_DIR / "sketch_kalman.png")) -> plt.Figure:
    rng = np.random.default_rng(2)
    dt, readings, steps_per_reading = 0.1, 12, 25
    # Body-frame command: drive 1 m/s along +x (exp(-wx) moves +x) while the turn rate
    # wanders, so the path meanders instead of retracing itself.
    turn = 0.9 * np.sin(0.2 * np.arange(readings * steps_per_reading) * dt)
    increments = ((mv.xy * turn - mv.wx) * dt).reshape(readings, steps_per_reading)
    times = np.arange(1, readings + 1) * steps_per_reading * dt
    Q = covariance(np.array([0.05, 0.05, 0.12]) * np.sqrt(dt))
    R = covariance(np.array([0.3, 0.3, 0.1]))

    origin = mv.xy
    initial = mv.rotor()
    sigma = covariance(np.zeros(3))
    truth, dead, measurements = simulate_motion(initial, increments, Q, R, rng)
    track = kalman_filter(initial, sigma, (increments * 0.5).exp(), measurements, Q, R)
    return draw_tracking(truth, dead, track, measurements, origin, times, plot_path)


if __name__ == "__main__":
    main()
