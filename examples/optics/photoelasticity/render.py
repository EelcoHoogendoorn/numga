"""Stress concentration and transmitted light around a hole in a loaded plate."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from numga import stack
from examples.util.animation import capture
from examples.optics.photoelasticity import core

animation_dpi = 85
animation_pixels = 200


# --- plumbing -------------------------------------------------------------------------
def window(positions: core.Planar) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Pixel-centre coordinates `[rows, columns, 2]` of an image grid, and its extent on the pixels' outer edges."""
    # Positions read by pairing with the planes across x and y, in the positions' own algebra.
    mv = positions.context.multivector
    coordinates = (stack((mv.x, mv.y)) & positions[..., None]).to_array().real
    horizontal, vertical = coordinates[0, :, 0], coordinates[:, 0, 1]
    half_x, half_y = (horizontal[1] - horizontal[0]) / 2, (vertical[1] - vertical[0]) / 2
    return coordinates, (horizontal[0] - half_x, horizontal[-1] + half_x, vertical[0] - half_y, vertical[-1] + half_y)


def plate(ax: plt.Axes, values: np.ndarray, extent: tuple[float, float, float, float], hole_radius: float,
          **style) -> plt.AxesImage:
    """An image over the plate's window, with the hole drawn over it."""
    image = ax.imshow(values, origin="lower", extent=extent, interpolation="bilinear", **style)
    ax.add_patch(Circle((0, 0), hole_radius, facecolor="white", edgecolor="0.6", linewidth=0.7))
    ax.set_axis_off()
    return image


def draw_stress(
    positions: core.Planar, difference: core.Scalar, hole_radius: float, far_stress: float,
) -> plt.Figure:
    """Principal stress difference over the applied stress, on the `[rows, columns]` pixel grid."""
    coordinates, extent = window(positions)
    values = difference.to_array().real / far_stress
    # The colour scale spans the plate only, not the pixels the hole covers.
    plate_pixels = np.linalg.norm(coordinates, axis=-1) >= hole_radius

    figure, ax = plt.subplots(figsize=(5.1, 4.3), layout="constrained")
    image = plate(ax, values, extent, hole_radius, cmap="cividis", vmin=0, vmax=values[plate_pixels].max())
    figure.colorbar(image, ax=ax, shrink=0.88)
    return figure


def draw_polariscope(positions: core.Planar, intensities: core.Scalar, hole_radius: float) -> plt.Figure:
    """Transmitted intensities `[views, rows, columns]` side by side, on a shared physical scale."""
    _, extent = window(positions)
    values = intensities.to_array().real

    figure, panels = plt.subplots(1, len(values), figsize=(10.8, 3.7), layout="constrained")
    for ax, intensity in zip(panels, values):
        # The pixel brightness is the light intensity, on the same scale in every view.
        plate(ax, intensity, extent, hole_radius, cmap="gray", vmin=0, vmax=1)
    return figure


def animate_polariscope(positions: core.Planar, frames: Iterable[core.Scalar], hole_radius: float) -> list[np.ndarray]:
    """One polariscope's transmitted intensity, `[rows, columns]` per frame, as the load changes."""
    _, extent = window(positions)
    figure = plt.figure(figsize=(animation_pixels / animation_dpi,) * 2, dpi=animation_dpi)
    ax = figure.add_axes((0, 0, 1, 1))
    image = plate(ax, np.zeros(positions.shape), extent, hole_radius, cmap="gray", vmin=0, vmax=1)
    rendered = []
    for intensity in frames:
        image.set_data(intensity.to_array().real)
        rendered.append(capture(figure))
    plt.close(figure)
    return rendered
