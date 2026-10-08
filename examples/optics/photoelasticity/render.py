"""Stress concentration and transmitted light around a hole in a loaded plate."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.ndimage import map_coordinates

from examples.animation import capture
from examples.optics.photoelasticity import core

if TYPE_CHECKING:
    from IPython.display import Image as Shown

ANIMATION_DPI = 85
IMAGE_SAMPLES = 200


# --- plumbing -------------------------------------------------------------------------
def draw_stress(
    positions: core.Planar, difference: core.Scalar, hole_radius: float, far_stress: float,
) -> plt.Figure:
    """Principal stress difference on the `[radii, angles + 1]` plate grid."""
    # Read coordinates only where the geometric field becomes a plotted image.
    coordinates = positions.cast(core.ga.subspace("x y")).kernel / hole_radius
    horizontal, vertical = np.moveaxis(coordinates, -1, 0)
    values = difference.cast(core.ga.subspace.scalar()).kernel[..., 0] / far_stress
    extent = np.max(np.abs(coordinates))

    figure, ax = plt.subplots(figsize=(5.1, 4.3), layout="constrained")
    image = ax.pcolormesh(horizontal, vertical, values, shading="gouraud",
                          cmap="cividis", vmin=0, rasterized=True)
    ax.add_patch(Circle((0, 0), 1, facecolor="white", edgecolor="0.6", linewidth=0.7))
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal",
           xlabel="x / hole radius", ylabel="y / hole radius")
    figure.colorbar(image, ax=ax, label="principal stress difference / applied stress", shrink=0.88)
    return figure


def draw_polariscope(positions: core.Planar, intensities: core.Scalar, hole_radius: float) -> plt.Figure:
    """Transmitted intensities `[views, radii, angles + 1]` side by side, on a shared physical scale."""
    coordinates = positions.cast(core.ga.subspace("x y")).kernel / hole_radius
    horizontal, vertical = np.moveaxis(coordinates, -1, 0)
    values = intensities.cast(core.ga.subspace.scalar()).kernel[..., 0]
    extent = np.max(np.abs(coordinates))

    figure, panels = plt.subplots(1, len(values), figsize=(10.8, 3.7), layout="constrained")
    for ax, intensity in zip(panels, values):
        # The pixel brightness is the light intensity, on the same scale in every view.
        ax.pcolormesh(horizontal, vertical, intensity, shading="gouraud",
                       cmap="gray", vmin=0, vmax=1, rasterized=True)
        ax.add_patch(Circle((0, 0), 1, facecolor="white", edgecolor="0.6", linewidth=0.7))
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
        ax.set_axis_off()
    return figure


def image_sampling(positions: core.Planar, hole_radius: float, samples: int) -> tuple[np.ndarray, float]:
    """Image-pixel locations in the plate's radial and angular sampling axes."""
    coordinates = positions.cast(core.ga.subspace("x y")).kernel / hole_radius
    radial_samples, angular_samples = coordinates.shape[:2]
    extent = np.max(np.abs(coordinates))
    angles = np.unwrap(np.arctan2(coordinates[0, :, 1], coordinates[0, :, 0]))
    outer_radii = np.hypot(coordinates[-1, :, 0], coordinates[-1, :, 1])

    pixel_centres = (np.arange(samples) + 0.5) * (2 * extent / samples) - extent
    horizontal, vertical = np.meshgrid(pixel_centres, pixel_centres)
    pixel_angles = (np.arctan2(vertical, horizontal) - angles[0]) % (2 * np.pi) + angles[0]
    angular_index = np.interp(pixel_angles, angles, np.arange(angular_samples))
    outer_radius = np.interp(pixel_angles, angles, outer_radii)
    radial_index = (np.hypot(horizontal, vertical) - 1) / (outer_radius - 1) * (radial_samples - 1)
    # These pixel-to-grid indices are fixed throughout loading; only the intensities change.
    return np.stack((radial_index, angular_index)), extent


def animate_polariscope(positions: core.Planar, frames: Iterable[core.Scalar], hole_radius: float) -> list[np.ndarray]:
    """One polariscope's transmitted intensity, `[radii, angles + 1]` per frame, as the load changes."""
    sampling, extent = image_sampling(positions, hole_radius, IMAGE_SAMPLES)
    figure = plt.figure(figsize=(IMAGE_SAMPLES / ANIMATION_DPI,) * 2, dpi=ANIMATION_DPI)
    ax = figure.add_axes((0, 0, 1, 1))
    image = ax.imshow(np.zeros((IMAGE_SAMPLES, IMAGE_SAMPLES)), origin="lower", cmap="gray",
                      vmin=0, vmax=1, extent=(-extent, extent, -extent, extent), interpolation="bilinear")
    ax.add_patch(Circle((0, 0), 1, facecolor="white", edgecolor="0.6", linewidth=0.7))
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
    ax.set_axis_off()
    rendered = []
    for intensity in frames:
        values = intensity.cast(core.ga.subspace.scalar()).kernel[..., 0]
        # Sample the same geometric field onto image pixels without rebuilding its mesh.
        # Pixels inside the hole are covered by the white circular patch.
        image.set_data(map_coordinates(values, sampling, order=1, mode="nearest"))
        rendered.append(capture(figure))
    plt.close(figure)
    return rendered


def inline(frames: list[np.ndarray], duration_ms: int) -> Shown:
    """Frames as a looping GIF to show in a notebook, kept in memory."""
    from io import BytesIO
    from IPython.display import Image as Shown
    from PIL import Image
    images = [Image.fromarray(pixels) for pixels in frames]
    buffer = BytesIO()
    images[0].save(buffer, format="GIF", save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
    return Shown(data=buffer.getvalue(), format="gif")
