"""Pixel-exact panels render the principal stress difference and polariscope fringe patterns around a circular hole."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import NumpyContext, stack
from examples.util.animation import capture
from numga.render import simplices
from examples.util.panels import offset_panel, overlay, pinned
from examples.optics.photoelasticity import core

figure_dpi = 100
panel_gap = 4
# Rim stroke width in pixels and grey tone around the circular hole.
rim_width = 1.0
rim_colour = np.full(3, 0.6)
# Real-coefficient multivector factory in the plate's algebra for rendering the hole marker.
real = NumpyContext(core.ga).multivector


# --- plumbing -------------------------------------------------------------------------
def window(positions: core.Planar) -> tuple[float, float, float, float]:
    """Outer pixel-edge bounds `(x_min, x_max, y_min, y_max)` enclose a `[n_rows, n_columns] Planar-` grid of pixel
    centres."""
    n_rows, n_columns = positions.shape
    mv = positions.context.multivector
    readings = (stack((mv.x, mv.y))[:, None] & positions.reshape(-1)).real()    # [2, n_rows * n_columns] Scalar+
    lower, upper = readings.min(axis=-1), readings.max(axis=-1)                 # [2] Scalar+, [2] Scalar+
    half_step = (upper - lower) / (2 * (np.array([n_columns, n_rows]) - 1))     # [2] Scalar+
    return tuple(stack((lower - half_step, upper + half_step), axis=-1).reshape(-1).to_array())


def panels(n_panels: int, pixels: tuple[int, ...]) -> tuple[plt.Figure, list[plt.Axes]]:
    """Creates a row of `n_panels` pixel-exact axes of shape `[n_rows, n_columns]` separated by `panel_gap` pixels."""
    n_rows, n_columns = pixels
    width = n_panels * n_columns + (n_panels - 1) * panel_gap
    figure = plt.figure(figsize=(width / figure_dpi, n_rows / figure_dpi), dpi=figure_dpi)
    axes = figure.subplots(1, n_panels, squeeze=False)[0]
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=panel_gap / n_columns)
    return figure, list(axes)


def plate_image(ax: plt.Axes, values: np.ndarray, extent: tuple[float, float, float, float], hole_radius: float,
                cmap: str, vmin: float, vmax: float) -> plt.AxesImage:
    """Draws a pixel-aligned `[n_rows, n_columns]` scalar image on `ax` and overlays the grey-rimmed circular hole."""
    image = pinned(ax, values[::-1], extent)
    image.set_cmap(cmap)
    image.set_clim(vmin, vmax)
    camera = offset_panel(ax, real.x, real.y)
    middle = (0 * real.x.dual())[None]                                          # [1] Planar-
    radius = hole_radius / camera.pixel
    overlay(ax, [simplices.markers(middle, rim_colour, radius + rim_width / 2, camera),
                 simplices.markers(middle, np.ones(3), radius - rim_width / 2, camera)])
    return image


def stress_figure(
    positions: core.Planar, difference: core.Scalar, hole_radius: float, far_stress: float,
) -> plt.Figure:
    """Renders the normalized principal stress difference (`[n_rows, n_columns] Scalar+`) outside the circular hole."""
    extent = window(positions)
    values = difference.to_array().real / far_stress
    # Span the colour scale over exterior plate pixels only, excluding pixels covered by the hole.
    plate_pixels = positions.antinorm().real() >= hole_radius

    figure, (ax,) = panels(1, values.shape)
    plate_image(ax, values, extent, hole_radius, "cividis", 0, values[plate_pixels].max())
    return figure


def polariscope_figure(positions: core.Planar, intensities: core.Scalar, hole_radius: float) -> plt.Figure:
    """Renders `[n_views, n_rows, n_columns] Scalar+` polariscope intensities side by side on a shared `[0, 1]` scale."""
    extent = window(positions)
    values = intensities.to_array().real

    figure, axes = panels(len(values), values.shape[1:])
    for ax, intensity_panel in zip(axes, values):
        plate_image(ax, intensity_panel, extent, hole_radius, "gray", 0, 1)
    return figure


def polariscope_animation(positions: core.Planar, frames: Iterable[core.Scalar],
                          hole_radius: float) -> Iterator[np.ndarray]:
    """Yields RGB frames of the `[n_rows, n_columns] Scalar+` polariscope intensity across a loading sequence."""
    extent = window(positions)
    figure, (ax,) = panels(1, positions.shape)
    image = plate_image(ax, np.zeros(positions.shape), extent, hole_radius, "gray", 0, 1)
    try:
        for intensity_frame in frames:
            image.set_data(intensity_frame.to_array().real[::-1])
            yield capture(figure)
    finally:
        plt.close(figure)
