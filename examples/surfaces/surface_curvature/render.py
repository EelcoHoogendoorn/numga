"""Implicit renders of quadric surfaces shade each pixel by its Gaussian curvature and ink the lines of curvature as level sets of the confocal parameters."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

from numga import stack
from examples.surfaces.surface_curvature import core
from examples.util.wgsl import image_figure

FAMILY_COLORS = (np.array([0.10, 0.10, 0.10]), np.array([0.55, 0.08, 0.08]))
N_LEVELS = 14
LINE_WIDTH = 1.2


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """The Euclidean coordinates of finite points."""
    mv = core.mv
    return (stack((mv.x, mv.y, mv.z)) & points.normalized()[..., None]).to_array()


def components(directions: core.Direction) -> np.ndarray:
    """The Cartesian components of directions."""
    mv = core.mv
    return (stack((mv.x, mv.y, mv.z)) & directions[..., None]).to_array()


def level_lines(parameter: np.ndarray, width: float) -> np.ndarray:
    """Pixel coverage in `[0, 1]` of evenly spaced level lines of one parameter over the image."""
    finite = parameter[np.isfinite(parameter)]
    scaled = parameter / ((finite.max() - finite.min()) / N_LEVELS)
    rate = np.hypot(*np.gradient(scaled)) + 1e-9
    return np.clip(1.0 - (np.abs(scaled - np.round(scaled)) / rate - 0.5 * width), 0.0, 1.0)


def shading(hits: core.Point, discriminant: core.Scalar, tangents: core.Plane, heading: core.Direction,
            height: float) -> tuple[np.ndarray, np.ndarray]:
    """Which pixels see the surface below the given height, and how brightly it is lit there."""
    covered = (discriminant >= 0) & ((core.mv.z & hits).abs() < height)
    normal = components(tangents.dual().cast(core.Direction))
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    toward_camera = -components(heading)
    lamp = toward_camera + np.cross(toward_camera, [0, 0, 1.0]) * 0.4 + [0, 0, 0.6]
    return covered, 0.4 + 0.6 * np.abs(normal @ (lamp / np.linalg.norm(lamp)))


def gaussian_colors(curvatures: core.Scalar, scale: float) -> np.ndarray:
    """Gaussian curvature, the product of the principal curvatures, mapped to red where positive and blue where negative."""
    gaussian = (curvatures[..., 0] * curvatures[..., 1]).to_array()
    return cm.RdBu_r(colors.Normalize(-scale, scale)(gaussian))[..., :3]


def overlay_lines(rgb: np.ndarray, parameters: core.Scalar, covered: np.ndarray) -> np.ndarray:
    """Darken the pixels on level lines of the two confocal parameters."""
    values = parameters.to_array()
    for family in range(2):
        coverage = np.nan_to_num(level_lines(np.where(covered, values[..., family], np.nan), LINE_WIDTH))[..., None]
        rgb = rgb * (1 - coverage) + FAMILY_COLORS[family] * coverage
    return rgb


def figure(images: list[tuple[np.ndarray, np.ndarray]]) -> plt.Figure:
    """Arrange rendered surface panels side by side at their exact pixel dimensions."""
    panels = [np.where(covered[..., None], rgb, 1.0) for rgb, covered in images]
    return image_figure(np.concatenate(panels, axis=1))


def draw_surfaces(scenes, heights) -> plt.Figure:
    """Render the lit quadric surfaces in neutral grey."""
    images = []
    for scene, height in zip(scenes, heights):
        covered, light = shading(*scene, height)
        images.append((0.85 * light[..., None] * np.ones(3), covered))
    return figure(images)


def draw_gaussian_curvature(scenes, curvatures, heights, scale: float) -> plt.Figure:
    """Render the quadric surfaces coloured by Gaussian curvature."""
    images = []
    for scene, principal_pair, height in zip(scenes, curvatures, heights):
        covered, light = shading(*scene, height)
        images.append((gaussian_colors(principal_pair, scale) * light[..., None], covered))
    return figure(images)


def draw_curvature_lines(scenes, curvatures, parameters, heights, scale: float) -> plt.Figure:
    """Render the quadric surfaces coloured by Gaussian curvature with their lines of curvature."""
    images = []
    for scene, principal_pair, confocal_pair, height in zip(scenes, curvatures, parameters, heights):
        covered, light = shading(*scene, height)
        images.append((overlay_lines(gaussian_colors(principal_pair, scale) * light[..., None], confocal_pair, covered), covered))
    return figure(images)
