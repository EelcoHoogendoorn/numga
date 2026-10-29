"""Stress concentration and transmitted light around a hole in a loaded plate."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from numga import stack
from examples.util.animation import capture
from examples.util.wgsl import pinned
from examples.optics.photoelasticity import core

figure_dpi = 100
panel_gap = 4
rim_width = 0.7


# --- plumbing -------------------------------------------------------------------------
def window(positions: core.Planar) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Pixel-centre coordinates `[rows, columns, 2]` of an image grid, and its extent on the pixels' outer edges."""
    # Positions read by pairing with the planes across x and y, in the positions' own algebra.
    mv = positions.context.multivector
    coordinates = (stack((mv.x, mv.y)) & positions[..., None]).to_array().real
    horizontal, vertical = coordinates[0, :, 0], coordinates[:, 0, 1]
    half_x, half_y = (horizontal[1] - horizontal[0]) / 2, (vertical[1] - vertical[0]) / 2
    return coordinates, (horizontal[0] - half_x, horizontal[-1] + half_x, vertical[0] - half_y, vertical[-1] + half_y)


def panels(n_panels: int, pixels: tuple[int, ...]) -> tuple[plt.Figure, list[plt.Axes]]:
    """A figure of panels side by side, each the image's own pixels, a few pixels apart."""
    rows, columns = pixels
    width = n_panels * columns + (n_panels - 1) * panel_gap
    figure = plt.figure(figsize=(width / figure_dpi, rows / figure_dpi), dpi=figure_dpi)
    axes = figure.subplots(1, n_panels, squeeze=False)[0]
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=panel_gap / columns)
    return figure, list(axes)


def plate(ax: plt.Axes, values: np.ndarray, extent: tuple[float, float, float, float], hole_radius: float,
          cmap: str, vmin: float, vmax: float) -> plt.AxesImage:
    """An image over the plate's window, rows up from the bottom, pixel for pixel, with the hole drawn over it."""
    image = pinned(ax, values[::-1], extent)
    image.set_cmap(cmap)
    image.set_clim(vmin, vmax)
    ax.add_patch(Circle((0, 0), hole_radius, facecolor="white", edgecolor="0.6", linewidth=rim_width))
    return image


def draw_stress(
    positions: core.Planar, difference: core.Scalar, hole_radius: float, far_stress: float,
) -> plt.Figure:
    """Principal stress difference over the applied stress, on the `[rows, columns]` pixel grid."""
    coordinates, extent = window(positions)
    values = difference.to_array().real / far_stress
    # The colour scale spans the plate only, not the pixels the hole covers.
    plate_pixels = np.linalg.norm(coordinates, axis=-1) >= hole_radius

    figure, (ax,) = panels(1, values.shape)
    plate(ax, values, extent, hole_radius, "cividis", 0, values[plate_pixels].max())
    return figure


def draw_polariscope(positions: core.Planar, intensities: core.Scalar, hole_radius: float) -> plt.Figure:
    """Transmitted intensities `[views, rows, columns]` side by side, on a shared physical scale."""
    _, extent = window(positions)
    values = intensities.to_array().real

    figure, axes = panels(len(values), values.shape[1:])
    for ax, intensity in zip(axes, values):
        # The pixel brightness is the light intensity, on the same scale in every view.
        plate(ax, intensity, extent, hole_radius, "gray", 0, 1)
    return figure


def animate_polariscope(positions: core.Planar, frames: Iterable[core.Scalar],
                        hole_radius: float) -> Iterator[np.ndarray]:
    """One polariscope's transmitted intensity, `[rows, columns]` per frame, as the load changes."""
    _, extent = window(positions)
    figure, (ax,) = panels(1, positions.shape)
    image = plate(ax, np.zeros(positions.shape), extent, hole_radius, "gray", 0, 1)
    for intensity in frames:
        image.set_data(intensity.to_array().real[::-1])
        yield capture(figure)
    plt.close(figure)
