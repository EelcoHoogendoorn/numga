"""Drawing for the station-keeping example: simulated positions, their 2σ ellipses, and the
ellipse's size against the controller gain. Lengths are in the units of the example, metres in the
notebook."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np

from examples.util.animation import capture
from examples.estimation.pose_diffusion import core



def xy(points: core.Point) -> np.ndarray:
    """Euclidean coordinates of points: their pairings with the coordinate lines, at unit weight."""
    return np.stack([((line & points) / (core.mv.w & points)).to_array() for line in (core.mv.x, core.mv.y)], axis=-1)


def ellipse(spread: core.Spread) -> np.ndarray:
    """The 2σ ellipse of a position's readouts: its principal variances and their normals,
    as a ring of points about the commanded position."""
    variances, lines = spread.eigh()                               # [modes] Scalar, [modes] Line
    variances = variances.to_array()
    normals = lines.cast(core.ga.subspace("x y")).kernel
    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    angle = np.linspace(0.0, 2.0 * np.pi, 80)
    return normals.T @ (2.0 * np.sqrt(variances)[:, None] * np.stack([np.cos(angle), np.sin(angle)]))


def arrows(poses: core.Motor, extent: float) -> tuple[np.ndarray, np.ndarray]:
    """The bodies' positions, and their arrows a short way along their headings, `[bodies, 2]` each."""
    # A point a short way ahead of the set point.
    heading = core.ORIGIN + core.mv.direction([0.08 * extent, 0.0])   # [] Point
    here = xy(poses >> core.ORIGIN)                                # [bodies, 2]
    return here, xy(poses >> heading) - here


def draw_cloud(ax, poses: core.Motor, predicted: core.Covariance, limit: core.Covariance, title: str,
               extent: float) -> tuple:
    """The bodies as short arrows through their positions along their headings, over the predicted
    ellipse (solid) and the settled one (dashed), in a square of half-width extent. Returns the arrows and
    the predicted ellipse drawn."""
    here, ahead = arrows(poses, extent)
    quiver = ax.quiver(here[:, 0], here[:, 1], ahead[:, 0], ahead[:, 1], angles="xy", scale_units="xy", scale=1,
                       pivot="mid", width=0.002, color="#4a6fa5", alpha=0.35)
    ax.plot(*ellipse(core.position_spread(limit)), "--", color="#c0392b", linewidth=1.4)
    spread, = ax.plot(*ellipse(core.position_spread(predicted)), "-", color="#c0392b", linewidth=1.4)
    # A faint cross through the set point, its lines named at their ends, without scales.
    ax.axhline(0, color="0.85", linewidth=0.7, zorder=0)
    ax.axvline(0, color="0.85", linewidth=0.7, zorder=0)
    ax.text(extent, 0, "forward", ha="right", va="top", fontsize=9, color="0.4")
    ax.text(0, extent, "sideways ", ha="right", va="bottom", fontsize=9, color="0.4", rotation=90, rotation_mode="anchor")
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title)
    return quiver, spread


def clouds(states: dict, extent: float) -> tuple[plt.Figure, list]:
    """The clouds of each named run, over its predicted and settled ellipses, and each cloud's arrows and
    predicted ellipse."""
    figure, row = plt.subplots(1, len(states), figsize=(5.5 * len(states), 5.5), layout="constrained")
    moving = [draw_cloud(ax, poses, predicted, limit, name, extent)
              for ax, (name, (poses, predicted, limit)) in zip(np.atleast_1d(row), states.items())]
    return figure, moving


def draw_settled(states: dict, extent: float) -> plt.Figure:
    """The clouds of each named run, over its predicted and settled ellipses."""
    figure, _ = clouds(states, extent)
    return figure


def animate_clouds(runs: dict, extent: float) -> Iterator[np.ndarray]:
    """One frame per yielded state: the clouds spreading while the predicted ellipse grows to the
    dashed settled one."""
    steps = zip(*runs.values())
    figure, moving = clouds(dict(zip(runs, next(steps))), extent)
    yield capture(figure)
    # The figure laid out once, by its first frame; only the arrows and the predicted ellipses move between frames.
    figure.set_layout_engine("none")
    for states in steps:
        for (quiver, spread), (poses, predicted, _) in zip(moving, states):
            here, ahead = arrows(poses, extent)
            quiver.set_offsets(here)
            quiver.set_UVC(ahead[:, 0], ahead[:, 1])
            spread.set_data(*ellipse(core.position_spread(predicted)))
        yield capture(figure)
    plt.close(figure)


def draw_envelope(gains: np.ndarray, spreads: core.Spread, tolerance: float) -> plt.Figure:
    """The 2σ half-widths of the position ellipse, along its long and its short axis, against the
    controller gain, with the tolerance on the long axis and the smallest gain that meets it."""
    variances = spreads.eigvalsh().to_array()                      # [gains, modes]
    half_widths = 2.0 * np.sqrt(np.sort(variances, axis=-1)[:, ::-1])
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
    ax.set_ylim(bottom=0)
    ax.legend()
    figure.tight_layout()
    return figure
