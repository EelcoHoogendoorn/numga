"""Pose filtering on the motor manifold in PGA2D: the covariance is a map from readouts to bivectors.

The state is a motor; its uncertainty is a covariance over body-frame bivector perturbations.
A linear readout of a bivector is a line, so the covariance is a unary extensor line -> bivector:
the twist correlated with a readout. Prediction moves it like any map, pulling the readout
through the step and pushing the twist back, and the update solves with the map inverse. The
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
Line = ga.gatype.vector()
Covariance = ga.gatype((Bivector, Line))      # twist <- linear readout of a twist (a line)


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
            # The covariance takes a readout of the perturbation to the twist correlated with it,
            # so it moves like any map: pull the readout through the step, push the twist back.
            sigma = step << sigma(step >> Line) + Q

        # Update on a noisy pose measurement. The innovation is the log of the relative
        # motor, the gain is a ratio of covariance maps, and the correction is exponentiated.
        innovation = (estimate.inverse() * measured).log() * 2
        gain = sigma((sigma + R).inverse())
        estimate = estimate * (gain(innovation) * 0.5).exp()
        sigma = sigma - gain(sigma)

        yield estimate, sigma


def position_ellipse(estimate: Motor, sigma: Covariance, origin: Point) -> tuple[Point, Scalar, Line]:
    """The estimated position, with the principal variances and axes of its uncertainty.

    A readout of position along a line, l & shift(δ), is a readout of the twist through the
    incidence pairing, so the covariance of those readouts is a form on lines. Against the
    line metric, which measures a line's normal and not its offset, its principal readouts
    are the ellipse axes. The offset mode is zero in both forms and has no variance.
    """
    here = estimate >> origin                                  # [n] Point
    shift = Bivector.commutator(here)(estimate >> Bivector)    # [n] Point <- Bivector
    readout = (Line & Bivector).solve(Line & shift)            # [n] Line <- Line
    position = readout & sigma(readout)                        # [n] Scalar <- (Line, Line)
    variances, axes = position.eig()                           # [n, 3] Scalar, [n, 3] Line; the eigenpairs of this symmetric pencil are real
    return here, variances.real(), axes.real()


# --- plumbing: covariance and sampling -------------------------------------------------
def covariance(translation_std: float, rotation_std: float) -> Covariance:
    """Body-frame noise: isotropic translation, and rotation about the body origin."""
    translation = mv.yw * (mv.yw & Line) + mv.wx * (mv.wx & Line)   # [] Bivector <- Line
    rotation = mv.xy * (mv.xy & Line)                               # [] Bivector <- Line
    return translation * translation_std**2 + rotation * rotation_std**2


def sample(cov: Covariance, rng: np.random.Generator) -> Bivector:
    """One bivector drawn from a covariance map.

    Readouts orthonormal in the covariance's own form carry independent unit normal draws:
    the twists they select, cov(l), then have covariance cov (Line & cov)^-1 cov = cov.
    """
    readouts = Line & cov                                      # [] Scalar <- (Line, Line)
    _, lines = readouts.eigh(readouts)                         # [3] Line: orthonormal in the form itself
    return (cov(lines) * rng.normal(size=3)).sum(axis=0)       # [] Bivector


# --- plotting -------------------------------------------------------------------------
def xy(point: Point) -> np.ndarray:
    k = point.cast(ga.subspace("yw wx xy")).kernel
    return k[..., :2] / k[..., 2:]


def principal_axes(variances: Scalar, axes: Line) -> tuple[np.ndarray, np.ndarray]:
    """(n, 2) finite variances and (n, 2, 2) unit axis directions, the normals of the axis lines."""
    values = variances.to_array()
    finite = np.isfinite(values)
    normals = axes.cast(ga.subspace("x y")).kernel
    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True).clip(1e-300)
    n = values.shape[0]
    return values[finite].reshape(n, 2), normals[finite].reshape(n, 2, 2).swapaxes(-1, -2)


def draw_ellipse(ax, centre: np.ndarray, values: np.ndarray, vectors: np.ndarray, color: str) -> None:
    t = np.linspace(0.0, 2.0 * np.pi, 40)
    ring = vectors @ (2.0 * np.sqrt(np.maximum(values, 0.0))[:, None] * np.stack([np.cos(t), np.sin(t)]))
    ax.plot(centre[0] + ring[0], centre[1] + ring[1], color=color, linewidth=0.8)


def draw_tracking(truth: Motor, dead: Motor, here: Point, variances: Scalar, axes: Line,
                  measurements: list[Motor], origin: Point, times: np.ndarray, plot_path: str) -> plt.Figure:
    values, vectors = principal_axes(variances, axes)
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
    ax_err.plot(times, 2 * np.sqrt(values.max(axis=1)), color="tab:blue", linestyle=":", label="filter's own 2σ")
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
    Q = covariance(0.05 * np.sqrt(dt), 0.12 * np.sqrt(dt))
    R = covariance(0.3, 0.1)

    origin = mv.xy
    initial = mv.rotor()
    sigma = covariance(0.0, 0.0)
    truth, dead, measurements = simulate_motion(initial, increments, Q, R, rng)
    estimates, covariances = zip(*kalman_filter(initial, sigma, (increments * 0.5).exp(), measurements, Q, R))
    here, variances, axes = position_ellipse(Extensor.stack(estimates), Extensor.stack(covariances), origin)
    return draw_tracking(truth, dead, here, variances, axes, measurements, origin, times, plot_path)


if __name__ == "__main__":
    main()
