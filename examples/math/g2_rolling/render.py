"""A marked ball rolling over a sphere, with its contact route painted on the sphere."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from PIL import Image, ImageDraw

from examples.raster import view_frame
from numga import stack

from .core import Rotor, Vector, ga, mv


BACKGROUND = np.array([0.98, 0.98, 0.96])
SURFACE = np.array([0.85, 0.87, 0.86])
SEAMS = np.array([0.13, 0.15, 0.18])
OCTANTS = np.array([
    [0.19, 0.44, 0.66], [0.93, 0.48, 0.19],
    [0.94, 0.75, 0.32], [0.19, 0.64, 0.61],
    [0.81, 0.35, 0.39], [0.40, 0.68, 0.84],
    [0.61, 0.47, 0.69], [0.91, 0.89, 0.76],
])
ROUTE_COLORS = [(27, 111, 152), (213, 99, 36), (113, 67, 151)]
LIGHT = np.array([-0.40, 0.48, 0.78])
SUPERSAMPLE = 2


def coordinates(vectors: Vector) -> np.ndarray:
    """Spatial coefficients in the frame used by the drawing."""
    return vectors.cast(ga.subspace("x y z")).kernel


class SphereScene:
    """Cache the fixed sphere and route, then shade the moving ball in its own frame."""

    def __init__(self, paths: tuple[Vector, ...], fit_paths: tuple[Vector, ...],
                 sphere_radius: float, ball_radius: float,
                 elevation: float, azimuth: float, size: int) -> None:
        self.frame = view_frame(elevation, azimuth)
        self.sphere_radius = sphere_radius
        self.ball_radius = ball_radius
        self.size = size
        self.resolution = size * SUPERSAMPLE
        self.marks = stack([mv.x, mv.y, mv.z])
        route = [coordinates(path) @ self.frame.T for path in paths]

        # The moving ball's entire silhouette, together with the fixed sphere, sets the frame.
        centers = np.concatenate([coordinates(path) @ self.frame.T for path in fit_paths])[:, :2]
        centers = centers * (sphere_radius + ball_radius)
        low = np.minimum(centers.min(axis=0) - ball_radius, -sphere_radius)
        high = np.maximum(centers.max(axis=0) + ball_radius, sphere_radius)
        self.focus = (low + high) / 2
        self.reach = np.max(high - low) * 0.535
        self.scale = self.resolution / (2 * self.reach)
        pixel = (np.arange(self.resolution) + 0.5) / self.scale - self.reach
        horizontal, vertical = np.meshgrid(pixel + self.focus[0], -pixel + self.focus[1])
        self.horizontal = horizontal
        self.vertical = vertical
        radial_squared = horizontal ** 2 + vertical ** 2
        covered = radial_squared <= sphere_radius ** 2
        depth = np.sqrt(np.maximum(sphere_radius ** 2 - radial_squared, 0))
        self.depth = np.where(covered, depth, -np.inf)
        normals = np.stack([horizontal, vertical, depth], axis=-1) / sphere_radius
        shade = 0.58 + 0.42 * np.maximum(normals @ LIGHT, 0)
        image = np.broadcast_to(BACKGROUND, (*covered.shape, 3)).copy()
        image[covered] = (SURFACE * shade[..., None])[covered]
        image = Image.fromarray(np.uint8(np.clip(image * 255, 0, 255)))
        draw = ImageDraw.Draw(image)

        # Curves on the far side remain hidden; the depth test also hides the ball there.
        for route_index, directions in enumerate(route):
            projected = self.project(directions * sphere_radius)
            color = ROUTE_COLORS[route_index]
            for first, second, visible in zip(projected[:-1], projected[1:],
                                               (directions[:-1, 2] + directions[1:, 2]) > 0):
                if visible:
                    draw.line([tuple(first), tuple(second)], fill=color, width=3 * SUPERSAMPLE)
            for endpoint in (0, -1):
                if directions[endpoint, 2] > 0:
                    x, y = projected[endpoint]
                    radius = 4 * SUPERSAMPLE
                    draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                                 fill=color, outline=(245, 245, 238), width=SUPERSAMPLE)
        self.background = np.asarray(image)

    def project(self, points: np.ndarray) -> np.ndarray:
        """Camera-space points to image coordinates."""
        pixels = (points[..., :2] - self.focus) * self.scale
        return pixels * [1, -1] + self.resolution / 2

    def draw(self, contact: Vector, attitude: Rotor) -> np.ndarray:
        """Shade the visible sphere intersection under each pixel of the moving ball."""
        center = coordinates(contact) @ self.frame.T * (self.sphere_radius + self.ball_radius)
        center_pixel = self.project(center)
        pixel_radius = self.ball_radius * self.scale
        left, top = np.maximum(np.floor(center_pixel - pixel_radius - 1), 0).astype(int)
        right, bottom = np.minimum(np.ceil(center_pixel + pixel_radius + 1), self.resolution).astype(int)
        horizontal = (self.horizontal[top:bottom, left:right] - center[0]) / self.ball_radius
        vertical = (self.vertical[top:bottom, left:right] - center[1]) / self.ball_radius
        radial_squared = horizontal ** 2 + vertical ** 2
        depth = np.sqrt(np.maximum(1 - radial_squared, 0))
        normals = np.stack([horizontal, vertical, depth], axis=-1)
        visible = ((radial_squared <= 1)
                   & (center[2] + self.ball_radius * depth >= self.depth[top:bottom, left:right]))

        # The sphere's paint is fixed to its material, so it follows the supplied attitude.
        axes = coordinates(attitude >> self.marks) @ self.frame.T
        body = normals @ axes.T
        octant = (body[..., 0] >= 0) + 2 * (body[..., 1] >= 0) + 4 * (body[..., 2] >= 0)
        colors = OCTANTS[octant]
        seam = np.clip(1 - np.min(np.abs(body), axis=-1) / 0.035, 0, 1)
        colors = colors + seam[..., None] * (SEAMS - colors)
        shade = 0.47 + 0.53 * np.maximum(normals @ LIGHT, 0)
        colors = np.uint8(np.clip(colors * shade[..., None] * 255, 0, 255))
        image = self.background.copy()
        image[top:bottom, left:right][visible] = colors[visible]
        return np.asarray(Image.fromarray(image).resize((self.size, self.size), Image.Resampling.LANCZOS))


def frames(paths: tuple[Vector, ...], rolling: Iterable[tuple[Vector, Rotor]],
           sphere_radius: float, ball_radius: float, elevation: float, azimuth: float,
           size: int) -> list[np.ndarray]:
    """Render successive contact directions and ball attitudes with one fixed camera."""
    scene = SphereScene(paths, paths, sphere_radius, ball_radius, elevation, azimuth, size)
    return [scene.draw(contact, attitude) for contact, attitude in rolling]


def comparison_frames(paths: tuple[Vector, ...], transformed_paths: tuple[Vector, ...],
                      rolling: Iterable[tuple[Vector, Rotor]],
                      transformed_rolling: Iterable[tuple[Vector, Rotor]],
                      sphere_radius: float, ball_radius: float, elevation: float,
                      azimuth: float, size: int) -> list[np.ndarray]:
    """Two rolling routes in identical camera frames, synchronized by their supplied states."""
    fit_paths = paths + transformed_paths
    scene = SphereScene(paths, fit_paths, sphere_radius, ball_radius, elevation, azimuth, size)
    transformed_scene = SphereScene(transformed_paths, fit_paths, sphere_radius, ball_radius,
                                    elevation, azimuth, size)
    return [np.concatenate([scene.draw(*state), transformed_scene.draw(*transformed_state)], axis=1)
            for state, transformed_state in zip(rolling, transformed_rolling)]
