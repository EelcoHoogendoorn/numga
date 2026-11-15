"""Rendering of station-keeping vessel clouds, their two-standard-deviation position confidence quadrics, and the ellipsoid half-width envelope across controller gains."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from examples.estimation.pose_diffusion.core import Covariance, Motor, Quadric, Spread, mv, origin
from examples.util import charts
from examples.util.animation import capture
from examples.util.panels import overlay, panel
from numga import stack
from numga.render import Camera, Cap, drawing, quadrics, simplices
from numga.topology import Topology

ellipsoid_rgb = np.array(mcolors.to_rgb("#c0392b"))
heading_fraction = 0.08
sigmas = 2.0
point_px = 100 / 72
# Number of fleet vessels drawn as heading arrows, arrow styling in pixels, and cross-hair grey level.
n_shown = 120
arrow_rgb = np.array(mcolors.to_rgb("#4a6fa5"))
arrow_width, arrow_chevron = 1.5, 6.0
cross_rgb = np.full(3, 0.85)


# --- plumbing -------------------------------------------------------------------------
def cloud_layers(
    poses: Motor,
    predicted: Covariance,
    limit: Covariance,
    extent: float,
    ellipsoid_of: Callable[[Covariance], Quadric],
    camera: Camera,
) -> list[np.ndarray]:
    """Cross-hairs through the set point, shaded steady-state quadric, heading arrows for the first `n_shown`
    vessels in `poses`, and steady and integrated confidence curves form the RGBA layers on `camera`."""
    steady_quadric = ellipsoid_of(limit)[None]                                  # [1] Plane+ <- Point-
    predicted_quadric = ellipsoid_of(predicted)[None]                           # [1] Plane+ <- Point-
    cross = origin + mv.direction([[[-extent, 0.0], [extent, 0.0]], [[0.0, -extent], [0.0, extent]]])   # [2, 2] Point-
    # A bow offset ahead of the origin turns with each vessel motor to orient its heading arrow.
    bow = origin + mv.direction([heading_fraction * extent, 0.0])               # [] Point-
    shown = poses[:n_shown]                                                     # [...] Motor+
    positions = (shown >> origin).normalized()                                  # [...] Point-
    headings = (shown >> bow).normalized() - positions                          # [...] Point-
    arrows = stack((positions - headings / 2, positions + headings / 2), axis=-1)   # [..., 2] Point-
    return [
        simplices.strokes(cross.reshape(-1), Topology.chains([2, 2]), cross_rgb, charts.FRAME_PX, camera, cap=Cap.BUTT),
        quadrics.regions(steady_quadric[:, None], ellipsoid_rgb, camera, 0.14),
        simplices.strokes(arrows.reshape(-1), Topology.chains([2] * len(shown)), arrow_rgb, arrow_width, camera,
                          chevrons=np.ones(len(shown)), chevron=arrow_chevron),
        quadrics.curves(steady_quadric, 1.0 * point_px, ellipsoid_rgb, camera, 0.45),
        quadrics.curves(predicted_quadric, 1.4 * point_px, ellipsoid_rgb, camera),
    ]


def cloud_panel(
    ax: plt.Axes,
    poses: Motor,
    predicted: Covariance,
    limit: Covariance,
    extent: float,
    ellipsoid_of: Callable[[Covariance], Quadric],
) -> tuple:
    """Configures `ax` over `[-extent, extent]` with labelled forward and sideways axes and overlays the initial
    fleet cloud, returning `(image, camera)` for frame updates."""
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_axis_off()
    camera = panel(ax, mv.x, mv.y, mv.w)
    image = overlay(ax, cloud_layers(poses, predicted, limit, extent, ellipsoid_of, camera))
    ax.text(extent, 0, "forward", ha="right", va="top", color="0.4")
    ax.text(0, extent, "sideways ", ha="right", va="bottom", color="0.4", rotation=90, rotation_mode="anchor")
    return image, camera


@charts.style()
def cloud_layout(
    panel_states: list[tuple[Motor, Covariance, Covariance]],
    extent: float,
    ellipsoid_of: Callable[[Covariance], Quadric],
) -> tuple[plt.Figure, list]:
    """One or more `(poses, predicted, limit)` fleet states form side-by-side square panels (`400` px each)
    together with their `(image, camera)` handles."""
    n_panels = len(panel_states)
    figure, row = charts.subplots(400 * n_panels, 400, ncols=n_panels, layout="constrained")
    figure.canvas.draw()
    artists = [
        cloud_panel(ax, poses, predicted, limit, extent, ellipsoid_of)
        for ax, (poses, predicted, limit) in zip(np.atleast_1d(row), panel_states)
    ]
    return figure, artists


def settled_figure(
    panel_states: list[tuple[Motor, Covariance, Covariance]],
    extent: float,
    ellipsoid_of: Callable[[Covariance], Quadric],
) -> plt.Figure:
    """Side-by-side panels depict each settled fleet cloud in `panel_states` over its integrated and steady-state
    position confidence quadrics."""
    figure, _ = cloud_layout(panel_states, extent, ellipsoid_of)
    return figure


def clouds_animation(
    extent: float,
    ellipsoid_of: Callable[[Covariance], Quadric],
    states: Iterable[Iterable[tuple[Motor, Covariance, Covariance]]],
) -> Iterator[np.ndarray]:
    """Yields one RGB frame per multi-panel step in `states` as each fleet cloud spreads and its integrated
    position quadric approaches the steady-state ellipse."""
    iterator = iter(states)
    first = list(next(iterator))
    figure, artists = cloud_layout(first, extent, ellipsoid_of)
    yield capture(figure)
    # Lock the layout after the first frame so only the composited pixel overlays update.
    figure.set_layout_engine("none")
    for panel_states in iterator:
        for (image, camera), (poses, predicted, limit) in zip(artists, panel_states):
            image.set_data(drawing.composited(cloud_layers(poses, predicted, limit, extent, ellipsoid_of, camera)))
        yield capture(figure)
    plt.close(figure)


@charts.style()
def envelope_figure(gains: np.ndarray, spreads: Spread, tolerance: float) -> plt.Figure:
    """Plots the two-standard-deviation major and minor half-widths of `spreads` (`[n_gains] Scalar+ <- (Plane+, Plane+)`)
    against controller `gains` (`[n_gains]`) with dashed markers at `tolerance` and the minimum gain meeting it."""
    # Principal standard deviations scaled by `sigmas`, sorted major-axis first.
    half_widths = np.sort((sigmas * spreads.eigvalsh().square_root()).to_array(), axis=-1)[:, ::-1]   # [..., 2]
    figure, ax = charts.subplots(420, 280)
    ax.plot(gains, half_widths[:, 0], color="#c0392b")
    ax.plot(gains, half_widths[:, 1], color="#4a6fa5")
    ax.axhline(tolerance, color="0.4", linestyle="--", linewidth=charts.FRAME_WIDTH)
    # Ascending gains make the first entry within tolerance the minimum required gain.
    min_gain = float(gains[half_widths[:, 0] <= tolerance][0])
    ax.axvline(min_gain, color="0.4", linestyle=":", linewidth=charts.FRAME_WIDTH)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, gains[-1])
    ax.set_ylim(0, half_widths[:, 0].max() * 1.05)
    ax.set_xticks([0, round(min_gain, 3), gains[-1]])
    ax.set_yticks([0, tolerance])
    ax.set_xlabel("controller gain (1/s)")
    ax.set_ylabel("2σ half-width (m)")
    figure.tight_layout()
    return figure

