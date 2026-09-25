"""Implicit renders of quadric surfaces: Gaussian curvature as colour, curvature lines as bands.

The lines are level sets of the confocal parameters, drawn where a parameter crosses an evenly
spaced level, a fixed width in pixels.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

from examples.geometry.surface_curvature import core

FAMILY_COLORS = (np.array([0.10, 0.10, 0.10]), np.array([0.55, 0.08, 0.08]))
LEVELS = 14


def euclidean(points: core.Point) -> np.ndarray:
    homogeneous = points.dual().kernel
    return homogeneous[..., :3] / homogeneous[..., 3:]


def components(directions: core.Direction) -> np.ndarray:
    """Euclidean components: the inner products with the axis directions."""
    return core.metric(directions[..., None], core.direction(np.eye(3))).to_array()


def level_lines(parameter: np.ndarray, width: float) -> np.ndarray:
    """Coverage in [0, 1] of evenly spaced level lines of one parameter over the image."""
    finite = parameter[np.isfinite(parameter)]
    scaled = parameter / ((finite.max() - finite.min()) / LEVELS)
    rate = np.hypot(*np.gradient(scaled)) + 1e-9
    return np.clip(1.0 - (np.abs(scaled - np.round(scaled)) / rate - 0.5 * width), 0.0, 1.0)


def lit(hits, discriminant, tangents, heading, height: float) -> tuple[np.ndarray, np.ndarray]:
    """Which pixels see the surface below the given height, and how brightly it is lit there."""
    covered = (discriminant.to_array() >= 0) & (np.abs(euclidean(hits)[..., 2]) < height)
    normal = components(tangents.dual().cast(core.Direction))
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    back = -components(heading)
    lamp = back + np.cross(back, [0, 0, 1.0]) * 0.4 + [0, 0, 0.6]
    return covered, 0.4 + 0.6 * np.abs(normal @ (lamp / np.linalg.norm(lamp)))


def gaussian_colors(curvatures: core.Scalar, scale: float) -> np.ndarray:
    """Gaussian curvature, the product of the principal curvatures: red where positive, blue where negative."""
    k = curvatures.to_array()
    return cm.RdBu_r(colors.Normalize(-scale, scale)(k[..., 0] * k[..., 1]))[..., :3]


def with_lines(rgb: np.ndarray, parameters: core.Scalar, covered: np.ndarray) -> np.ndarray:
    """Darken the pixels on level lines of the two confocal parameters."""
    t = parameters.to_array()
    for family in range(2):
        coverage = np.nan_to_num(level_lines(np.where(covered, t[..., family], np.nan), 1.2))[..., None]
        rgb = rgb * (1 - coverage) + FAMILY_COLORS[family] * coverage
    return rgb


def figure(images) -> plt.Figure:
    result, axes = plt.subplots(1, len(images), figsize=(6 * len(images), 6))
    for ax, (rgb, covered) in zip(np.atleast_1d(axes), images):
        ax.imshow(np.where(covered[..., None], rgb, 1.0), interpolation="bilinear")
        ax.set_axis_off()
    result.tight_layout()
    return result


def draw_surfaces(scenes, heights) -> plt.Figure:
    """The surfaces, lit, in grey. A scene is the hits, the discriminants, the tangent planes and the view heading."""
    images = []
    for scene, height in zip(scenes, heights):
        covered, light = lit(*scene, height)
        images.append((0.85 * light[..., None] * np.ones(3), covered))
    return figure(images)


def draw_gaussian_curvature(scenes, curvatures, heights, scale: float) -> plt.Figure:
    """The surfaces coloured by Gaussian curvature."""
    images = []
    for scene, k, height in zip(scenes, curvatures, heights):
        covered, light = lit(*scene, height)
        images.append((gaussian_colors(k, scale) * light[..., None], covered))
    return figure(images)


def draw_curvature_lines(scenes, curvatures, parameters, heights, scale: float) -> plt.Figure:
    """The surfaces coloured by Gaussian curvature, with their lines of curvature."""
    images = []
    for scene, k, t, height in zip(scenes, curvatures, parameters, heights):
        covered, light = lit(*scene, height)
        images.append((with_lines(gaussian_colors(k, scale) * light[..., None], t, covered), covered))
    return figure(images)
