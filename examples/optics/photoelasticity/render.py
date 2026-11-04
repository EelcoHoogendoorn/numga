"""Stress concentration and transmitted light around a hole in a loaded plate."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import NumpyContext, stack
from examples.util.animation import capture
from examples.util.wgsl import pinned, simplices
from examples.util.wgsl.camera import offset_panel
from examples.optics.photoelasticity import core

figure_dpi = 100
panel_gap = 4
# The hole's rim, its width in pixels and its grey.
rim_width = 1.0
rim_colour = np.full(3, 0.6)
# The plate's algebra with real coefficients, for the picture: the hole's middle, the origin, which the camera reads.
real = NumpyContext(core.ga).multivector


# --- plumbing -------------------------------------------------------------------------
def window(positions: core.Planar) -> tuple[float, float, float, float]:
    """The extent of an image grid of pixel centres `[rows, columns]`, on the pixels' outer edges."""
    # Positions read by pairing with the planes across x and y, in the positions' own algebra; the outer edges half a
    # step between pixel centres past the outermost, columns along x and rows along y.
    mv = positions.context.multivector
    readings = (stack((mv.x, mv.y))[:, None] & positions.reshape(-1)).real()   # [2, pixels] Scalar+
    low, high = readings.min(axis=-1), readings.max(axis=-1)                    # [2] Scalar+, [2] Scalar+
    half_step = (high - low) / (2 * (np.array(positions.shape[::-1]) - 1))      # [2] Scalar+
    return tuple(stack((low - half_step, high + half_step), axis=-1).reshape(-1).to_array())


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
    """An image over the plate's window, rows up from the bottom, pixel for pixel, with the hole drawn over it, a white
    disc rimmed in grey about the origin, as a camera of the plate's planes x and y sees it."""
    image = pinned(ax, values[::-1], extent)
    image.set_cmap(cmap)
    image.set_clim(vmin, vmax)
    camera = offset_panel(ax, real.x, real.y)
    middle = (0 * real.x.dual())[None]                                          # [1] Planar-
    radius = hole_radius / camera.pixel
    simplices.overlay(ax, [simplices.markers(middle, rim_colour, radius + rim_width / 2, camera),
                           simplices.markers(middle, np.ones(3), radius - rim_width / 2, camera)])
    return image


def draw_stress(
    positions: core.Planar, difference: core.Scalar, hole_radius: float, far_stress: float,
) -> plt.Figure:
    """Principal stress difference over the applied stress, on the `[rows, columns]` pixel grid."""
    extent = window(positions)
    values = difference.to_array().real / far_stress
    # The colour scale spans the plate only, not the pixels the hole covers: those within the hole's radius of its
    # middle, by the size of their place's square, whichever sign the square has in the algebra.
    plate_pixels = positions.squared().norm() >= hole_radius**2

    figure, (ax,) = panels(1, values.shape)
    plate(ax, values, extent, hole_radius, "cividis", 0, values[plate_pixels].max())
    return figure


def draw_polariscope(positions: core.Planar, intensities: core.Scalar, hole_radius: float) -> plt.Figure:
    """Transmitted intensities `[views, rows, columns]` side by side, on a shared physical scale."""
    extent = window(positions)
    values = intensities.to_array().real

    figure, axes = panels(len(values), values.shape[1:])
    for ax, intensity in zip(axes, values):
        # The pixel brightness is the light intensity, on the same scale in every view.
        plate(ax, intensity, extent, hole_radius, "gray", 0, 1)
    return figure


def animate_polariscope(positions: core.Planar, frames: Iterable[core.Scalar],
                        hole_radius: float) -> Iterator[np.ndarray]:
    """One polariscope's transmitted intensity, `[rows, columns]` per frame, as the load changes."""
    extent = window(positions)
    figure, (ax,) = panels(1, positions.shape)
    image = plate(ax, np.zeros(positions.shape), extent, hole_radius, "gray", 0, 1)
    for intensity in frames:
        image.set_data(intensity.to_array().real[::-1])
        yield capture(figure)
    plt.close(figure)
