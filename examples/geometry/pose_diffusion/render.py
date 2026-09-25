"""Drawing for the station-keeping example: simulated positions, their 2σ ellipses, and the
ellipse's size against the controller gain. Lengths are in the units of the example, metres in the
notebook."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.animation import capture
from examples.geometry.pose_diffusion import core



def xy(points: core.Point) -> np.ndarray:
    """Euclidean coordinates of points: their pairings with the coordinate lines, at unit weight."""
    return np.stack([((line & points) / (core.mv.w & points)).to_array() for line in (core.mv.x, core.mv.y)], axis=-1)


def ellipse(spread: core.Spread) -> np.ndarray:
    """The 2σ ellipse of a position's readouts: its finite principal variances and their normals,
    as a ring of points about the commanded position."""
    variances, lines = spread.eig()                                # [modes] Scalar, [modes] Line
    variances, lines = variances.real().to_array(), lines.real()
    # The offset of a line carries no variance.
    finite = np.isfinite(variances)
    normals = lines.cast(core.ga.subspace("x y")).kernel[finite]
    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    angle = np.linspace(0.0, 2.0 * np.pi, 80)
    return normals.T @ (2.0 * np.sqrt(variances[finite])[:, None] * np.stack([np.cos(angle), np.sin(angle)]))


def draw_cloud(ax, poses: core.Motor, predicted: core.Covariance, limit: core.Covariance, title: str,
               extent: float) -> None:
    """The bodies as short arrows through their positions along their headings, over the predicted
    ellipse (solid) and the settled one (dashed), in a square of half-width extent."""
    # A point a short way ahead of the set point.
    heading = core.ORIGIN + core.mv.yw * (0.08 * extent)          # [] Point
    here = xy(poses >> core.ORIGIN)                                # [bodies, 2]
    ahead = xy(poses >> heading) - here                            # [bodies, 2]
    ax.quiver(here[:, 0], here[:, 1], ahead[:, 0], ahead[:, 1], angles="xy", scale_units="xy", scale=1,
              pivot="mid", width=0.002, color="#4a6fa5", alpha=0.35)
    for covariance, style in ((limit, "--"), (predicted, "-")):
        ring = ellipse(core.position_spread(covariance))
        ax.plot(ring[0], ring[1], style, color="#c0392b", linewidth=1.4)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("forward (m)")
    ax.set_ylabel("sideways (m)")


def draw_clouds(states: dict, seconds: float, extent: float) -> plt.Figure:
    figure, row = plt.subplots(1, len(states), figsize=(5.5 * len(states), 5.5))
    for ax, (name, (poses, predicted, limit)) in zip(np.atleast_1d(row), states.items()):
        draw_cloud(ax, poses, predicted, limit, f"{name}, t = {seconds:.1f} s", extent)
    figure.tight_layout()
    return figure


def draw_settled(states: dict, extent: float) -> plt.Figure:
    """The clouds at the end of the run, over the settled ellipses."""
    figure, row = plt.subplots(1, len(states), figsize=(5.5 * len(states), 5.5))
    for ax, (name, (poses, predicted, limit)) in zip(np.atleast_1d(row), states.items()):
        draw_cloud(ax, poses, predicted, limit, name, extent)
    figure.tight_layout()
    return figure


def animate_clouds(runs: dict, dt: float, extent: float) -> list[np.ndarray]:
    """One frame per yielded state: the clouds spreading while the predicted ellipse grows to the
    dashed settled one."""
    frames = []
    for index, states in enumerate(zip(*runs.values())):
        figure = draw_clouds(dict(zip(runs, states)), index * dt, extent)
        frames.append(capture(figure))
        plt.close(figure)
    return frames


def draw_envelope(gains: np.ndarray, spreads: core.Spread, tolerance: float) -> plt.Figure:
    """The 2σ half-widths of the position ellipse, along its long and its short axis, against the
    controller gain, with the tolerance on the long axis and the smallest gain that meets it."""
    variances = spreads.eigvals().real().to_array()                # [gains, modes]
    half_widths = 2.0 * np.sqrt(np.sort(np.where(np.isfinite(variances), variances, 0.0), axis=-1)[:, ::-1][:, :2])
    figure, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(gains, half_widths[:, 0], color="#c0392b", label="long axis")
    ax.plot(gains, half_widths[:, 1], color="#4a6fa5", label="short axis")
    ax.axhline(tolerance, color="0.4", linestyle="--", linewidth=1, label="tolerance")
    # The gains are ascending.
    least = gains[half_widths[:, 0] <= tolerance][0]              # 1/s
    ax.axvline(least, color="0.4", linestyle=":", linewidth=1)
    ax.annotate(f"gain {least:.3f} /s", (least, tolerance), textcoords="offset points", xytext=(6, 8))
    ax.set_xlabel("controller gain (1/s)")
    ax.set_ylabel("2σ half-width (m)")
    ax.legend()
    figure.tight_layout()
    return figure
