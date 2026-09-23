"""Drawing for the Kalman example: paths, 2σ position ellipses, and position errors over time."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.geometry.kalman.core import Line, Point, Scalar, ga


def xy(point: Point) -> np.ndarray:
    k = point.cast(ga.subspace("yw wx xy")).kernel
    return k[..., :2] / k[..., 2:]


def principal_axes(variances: Scalar, axes: Line) -> tuple[np.ndarray, np.ndarray]:
    """(n, 2) finite variances and (n, 2, 2) unit axis directions, the normals of the axis lines.

    The third mode, the offset of a line, carries no variance: its eigenvalue is infinite.
    """
    values = variances.to_array()
    finite = np.isfinite(values)
    n = values.shape[0]
    normals = axes.cast(ga.subspace("x y")).kernel[finite].reshape(n, 2, 2)
    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    return values[finite].reshape(n, 2), normals.swapaxes(-1, -2)


def draw_ellipse(ax, centre: np.ndarray, values: np.ndarray, vectors: np.ndarray, color: str) -> None:
    t = np.linspace(0.0, 2.0 * np.pi, 40)
    ring = vectors @ (2.0 * np.sqrt(values)[:, None] * np.stack([np.cos(t), np.sin(t)]))
    ax.plot(centre[0] + ring[0], centre[1] + ring[1], color=color, linewidth=0.8)


def draw_tracking(truth: Point, dead: Point, measured: Point, here: Point, variances: Scalar, axes: Line,
                  dead_error: Scalar, filtered_error: Scalar, times: np.ndarray) -> plt.Figure:
    """The true, dead-reckoned and filtered paths with ellipses; and both errors over time."""
    values, vectors = principal_axes(variances, axes)
    true_xy, dead_xy, est_xy, measured_xy = xy(truth), xy(dead), xy(here), xy(measured)

    fig, (ax, ax_err) = plt.subplots(1, 2, figsize=(11, 5), dpi=120)
    ax.plot(*true_xy.T, color="black", linewidth=1.5, label="truth")
    ax.plot(*dead_xy.T, color="tab:red", linestyle="--", linewidth=1, label="dead reckoning")
    ax.plot(*est_xy.T, color="tab:blue", linewidth=1, label="filtered")
    ax.plot(*measured_xy.T, marker="x", color="tab:green", linestyle="none", alpha=0.4, label="pose measurements")
    for centre, variance, vector in zip(est_xy, values, vectors):
        draw_ellipse(ax, centre, variance, vector, "tab:blue")
    ax.set_aspect("equal"); ax.legend(loc="upper left", fontsize=8)
    ax.set_title("paths, with the 2σ position ellipse at each measurement")
    ax_err.plot(times, dead_error.to_array(), color="tab:red", linestyle="--", label="dead reckoning")
    ax_err.plot(times, filtered_error.to_array(), color="tab:blue", label="filtered")
    ax_err.plot(times, 2 * np.sqrt(values.max(axis=1)), color="tab:blue", linestyle=":", label="filter's own 2σ")
    ax_err.set_xlabel("time"); ax_err.set_ylabel("position error"); ax_err.legend(fontsize=8)
    ax_err.set_title("error over time")
    return fig
