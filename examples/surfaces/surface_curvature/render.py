"""Implicit renders of quadric surfaces shade each pixel by its Gaussian curvature and ink the lines of curvature as level sets of the confocal parameters."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

from numga import stack
from examples.surfaces.surface_curvature import core
from examples.util.panels import image_figure

family_colors = (np.array([0.10, 0.10, 0.10]), np.array([0.55, 0.08, 0.08]))
n_levels = 14
line_width = 1.2
# The lamp's angles in radians up from the line of sight toward the zenith, and then about the vertical.
lamp_rise, lamp_swing = np.radians(22.0), np.radians(22.0)


def view_rays(elevation: float, azimuth: float, extent: float, n_pixels: int) -> tuple[core.Point, core.Direction]:
    """An orthographic screen (`[n_pixels, n_pixels] Point-`) and view heading (`[] Direction-`) turn by `elevation` and `azimuth`."""
    mv = core.mv
    turn = (mv.xy * (-np.radians(azimuth) / 2)).exp() * (mv.xz * (-np.radians(elevation) / 2)).exp()  # [] Rotor+
    span = np.linspace(-extent, extent, n_pixels)
    # Place the screen behind the scene along x, across it along y, and up along z.
    screen = np.stack(np.broadcast_arrays(10 * extent, span[None, :], span[::-1, None]), axis=-1)
    origins = turn >> mv.point(screen)                                                  # [n_pixels, n_pixels] Point-
    heading = turn >> -mv.x.dual()                                                      # [] Direction-
    return origins, heading


def coordinates(points: core.Point) -> np.ndarray:
    """Pairing normalized points (`[...] Point-`) with the coordinate planes yields their Euclidean coordinates (`[..., 3]`)."""
    mv = core.mv
    return (stack((mv.x, mv.y, mv.z)) & points.normalized()[..., None]).to_array()


def level_lines(parameter: np.ndarray, width: float) -> np.ndarray:
    """Pixel coverage in `[0, 1]` marks evenly spaced level lines of `parameter` across the image."""
    finite = parameter[np.isfinite(parameter)]
    scaled = parameter / ((finite.max() - finite.min()) / n_levels)
    rate = np.hypot(*np.gradient(scaled)) + 1e-9
    return np.clip(1.0 - (np.abs(scaled - np.round(scaled)) / rate - 0.5 * width), 0.0, 1.0)


def surface_shading(
    hits: core.Point,
    discriminant: core.Scalar,
    tangents: core.Plane,
    heading: core.Direction,
    height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Visible pixels (`[n_pixels, n_pixels]`) lie below `height`, lit diffusely by how squarely their tangent planes face the lamp."""
    mv = core.mv
    covered = (discriminant >= 0) & ((mv.z & hits).abs() < height)
    toward_camera = -heading                                                            # [] Direction-
    # The horizontal line through the origin perpendicular to the view turns the lamp upward toward +z.
    level_hinge = (toward_camera.dual() ^ mv.z).normalized()                            # [] Line+
    rise_rotor = (level_hinge * (lamp_rise / 2)).exp()                                  # [] Rotor+
    swing_rotor = (mv.xy * (lamp_swing / 2)).exp()                                      # [] Rotor+
    lamp_rotor = swing_rotor * rise_rotor                                               # [] Rotor+
    lamp = lamp_rotor >> toward_camera                                                  # [] Direction-
    # Pairing a unit plane with a unit direction reads the cosine between the plane normal and the lamp direction.
    facing = (tangents.normalized() & lamp).abs()                                       # [...] Scalar+
    return covered, 0.4 + 0.6 * facing.to_array()


def gaussian_colors(curvatures: core.Scalar, scale: float) -> np.ndarray:
    """Maps Gaussian curvature, the product of the two principal curvatures, to red when positive and blue when negative."""
    gaussian = (curvatures[..., 0] * curvatures[..., 1]).to_array()
    return cm.RdBu_r(colors.Normalize(-scale, scale)(gaussian))[..., :3]


def line_overlay(rgb: np.ndarray, parameters: core.Scalar, covered: np.ndarray) -> np.ndarray:
    """Inks evenly spaced level lines of the two confocal parameters across the covered surface pixels."""
    values = parameters.to_array()
    for family in range(2):
        coverage = np.nan_to_num(level_lines(np.where(covered, values[..., family], np.nan), line_width))[..., None]
        rgb = rgb * (1 - coverage) + family_colors[family] * coverage
    return rgb


def panel_figure(images: list[tuple[np.ndarray, np.ndarray]]) -> plt.Figure:
    """Arranges rendered surface panels side by side on a white background at exact pixel dimensions."""
    panels = [np.where(covered[..., None], rgb, 1.0) for rgb, covered in images]
    return image_figure(np.concatenate(panels, axis=1))


def surfaces_figure(scenes: list[tuple[core.Point, core.Scalar, core.Plane, core.Direction]], heights: tuple[float, float]) -> plt.Figure:
    """Renders the lit quadric surfaces in neutral grey."""
    images = []
    for scene, height in zip(scenes, heights):
        covered, light = surface_shading(*scene, height)
        images.append((0.85 * light[..., None] * np.ones(3), covered))
    return panel_figure(images)


def gaussian_curvature_figure(
    scenes: list[tuple[core.Point, core.Scalar, core.Plane, core.Direction]],
    curvatures: list[core.Scalar],
    heights: tuple[float, float],
    scale: float,
) -> plt.Figure:
    """Renders the quadric surfaces coloured by Gaussian curvature."""
    images = []
    for scene, principal_pair, height in zip(scenes, curvatures, heights):
        covered, light = surface_shading(*scene, height)
        images.append((gaussian_colors(principal_pair, scale) * light[..., None], covered))
    return panel_figure(images)


def curvature_lines_figure(
    scenes: list[tuple[core.Point, core.Scalar, core.Plane, core.Direction]],
    curvatures: list[core.Scalar],
    parameters: list[core.Scalar],
    heights: tuple[float, float],
    scale: float,
) -> plt.Figure:
    """Renders the quadric surfaces coloured by Gaussian curvature with inked lines of curvature."""
    images = []
    for scene, principal_pair, confocal_pair, height in zip(scenes, curvatures, parameters, heights):
        covered, light = surface_shading(*scene, height)
        shaded = gaussian_colors(principal_pair, scale) * light[..., None]
        images.append((line_overlay(shaded, confocal_pair, covered), covered))
    return panel_figure(images)
