"""Orthographic views of a population and its carried circles on the unit sphere."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PathCollection
import numpy as np

from numga import stack
from examples.animation import capture
from examples.raster import view_frame
from examples.math.spherical_synchronization.core import Cut, Point

size = 500
view = view_frame(20, 45)
margin = 1.07
n_circle_samples = 192
particle_colors = np.array([
    [0.90, 0.36, 0.20], [0.91, 0.59, 0.16], [0.24, 0.55, 0.61],
    [0.32, 0.43, 0.68], [0.62, 0.39, 0.61],
])
circle_color = np.array([0.12, 0.42, 0.46])


def circle_vertices(circles: np.ndarray) -> np.ndarray:
    """Uniformly sample each plane's intersection with the unit sphere."""
    normals = circles[..., :3]
    lengths = np.linalg.norm(normals, axis=-1)
    normals = normals / lengths[..., None]
    distances = circles[..., 3] / lengths
    # A transverse frame samples the geometric circle, independently of how its
    # parameterisation crowds together during the motion.
    reference = np.eye(3)[np.argmin(np.abs(normals), axis=-1)]
    first = np.cross(normals, reference)
    first /= np.linalg.norm(first, axis=-1)[..., None]
    second = np.cross(normals, first)
    angles = np.linspace(0, 2 * np.pi, n_circle_samples + 1)
    radius = np.sqrt(np.maximum(0, 1 - distances ** 2))
    return (normals * distances[..., None])[..., None, :] + radius[..., None, None] * (
        first[..., None, :] * np.cos(angles)[..., None]
        + second[..., None, :] * np.sin(angles)[..., None]
    )


def canvas() -> tuple[plt.Figure, LineCollection, tuple[PathCollection, PathCollection]]:
    """A fixed, softly shaded sphere and its two dynamic drawing layers."""
    fig, ax = plt.subplots(figsize=(size / 100, size / 100), dpi=100)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(-margin, margin), ylim=(-margin, margin), aspect="equal")
    ax.axis("off")

    coordinates = np.linspace(-margin, margin, size)
    right, up = np.meshgrid(coordinates, coordinates)
    depth = np.sqrt(np.maximum(0, 1 - right ** 2 - up ** 2))
    light = np.clip(-0.25 * right + 0.35 * up + 0.90 * depth, 0, 1)
    shade = 0.92 + 0.07 * light
    surface = np.stack((shade, shade * 1.005, shade * 1.01), axis=-1)
    surface[right ** 2 + up ** 2 > 1] = 1
    ax.imshow(np.clip(surface, 0, 1), extent=(-margin, margin, -margin, margin), origin="lower")
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="#c2c7c8", linewidth=0.7))
    lines = LineCollection([], linewidths=1.05, zorder=2)
    ax.add_collection(lines)
    back = ax.scatter([], [], s=19, linewidths=0.35, edgecolors="white", zorder=1.5)
    front = ax.scatter([], [], s=19, linewidths=0.35, edgecolors="white", zorder=3)
    return fig, lines, (back, front)


def update(lines: LineCollection, particles: tuple[PathCollection, PathCollection],
           population: np.ndarray, circles: np.ndarray) -> None:
    """Project the surface geometry, fading its far side."""
    viewed_circles = circle_vertices(circles) @ view.T
    segments = np.stack((viewed_circles[:, :-1], viewed_circles[:, 1:]), axis=-2).reshape(-1, 2, 3)
    depths = segments[..., 2].mean(axis=-1)
    order = np.argsort(depths)
    visibility = 0.18 + 0.82 * np.clip(depths * 12 + 0.5, 0, 1)
    colors = 1 - visibility[:, None] * (1 - circle_color)
    lines.set_segments(segments[order, :, :2])
    lines.set_colors(colors[order])

    viewed = population @ view.T
    order = np.argsort(viewed[:, 2])
    colors = particle_colors[np.arange(len(population)) % len(particle_colors)]
    visibility = 0.25 + 0.75 * np.clip(viewed[:, 2] * 12 + 0.5, 0, 1)
    colors = 1 - visibility[:, None] * (1 - colors)
    for layer, visible in zip(particles, (viewed[:, 2] < 0, viewed[:, 2] >= 0)):
        indices = order[visible[order]]
        layer.set_offsets(viewed[indices, :2])
        layer.set_facecolors(colors[indices])
        layer.set_sizes(12 + 10 * visibility[indices])


def coordinates(points: Point) -> np.ndarray:
    """Positions of directions on the last axis: each paired with the planes across x, y and z."""
    planes = points.context.multivector
    return (stack((planes.x, planes.y, planes.z), axis=-1) & points[..., None]).to_array()


def draw(population: Point, circles: Cut) -> plt.Figure:
    """Draw one population and the circles carried alongside it."""
    positions = coordinates(population)
    cuts = circles.cast(circles.algebra.subspace("x y z t")).kernel
    fig, lines, particles = canvas()
    update(lines, particles, positions, cuts)
    return fig


def frames(population: Point, circles: Cut) -> list[np.ndarray]:
    """Animate a fixed camera while the common motion gathers the population."""
    positions = coordinates(population)
    cuts = circles.cast(circles.algebra.subspace("x y z t")).kernel
    fig, lines, particles = canvas()
    images = []
    for frame_positions, frame_cuts in zip(positions, cuts):
        update(lines, particles, frame_positions, frame_cuts)
        images.append(capture(fig))
    plt.close(fig)
    return images
