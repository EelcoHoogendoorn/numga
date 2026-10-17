"""A unit circle reflected in its own plane."""

from __future__ import annotations

import numpy as np

from numga import stack

from examples.orientation.conformal_circle import core
from examples.orientation.conformal_circle.core import Circle, Path, Point, Sphere, mv


def circle() -> tuple[Point, Point, Point, Sphere, Sphere]:
    """Ordered circle points stay fixed while one cutting surface changes side."""
    points: Point = stack([(mv.x + mv.e).dual(), (-mv.y + mv.e).dual(), (-mv.x + mv.e).dual()])
    plane: Sphere = mv.z
    sphere: Sphere = mv.w
    mirror: Sphere = plane
    samples = 96
    meridians = 16
    parallels = 9
    angles = np.linspace(0, 2 * np.pi, samples + 1)
    longitudes = np.linspace(0, 2 * np.pi, meridians + 1)
    latitudes = np.linspace(-np.pi / 2, np.pi / 2, parallels)

    curve = core.circle_samples(points[0], mv.xy, angles)
    surface = core.sphere_samples(points[0], mv.xy, mv.xz, longitudes, latitudes)
    return core.reflected_scene(points, curve, surface, plane, sphere, mirror)


def circle_motions() -> tuple[Point, Point]:
    """Rigid circulation and a hyperbolic glide reflection preserving the same circle."""
    points: Point = stack([(mv.x + mv.e).dual(), (-mv.y + mv.e).dual(), (-mv.x + mv.e).dual()])
    infinity: Sphere = mv.e + mv.w
    origin: Sphere = (mv.e - mv.w) / 2
    reflector: Sphere = mv.x
    frames = 64
    radial_samples = 40
    angular_samples = 96
    glide_amplitude = 0.9
    phase = np.linspace(0, 2 * np.pi, frames, endpoint=False)
    radii = np.linspace(0, 1, radial_samples + 1)
    angles = np.linspace(0, 2 * np.pi, angular_samples + 1)

    circle: Path = (points[0] & points[1] & points[2]).normalized()
    # Infinity selects the Euclidean rotation about the circle's axis.
    axis: Circle = (circle.dual() | infinity).normalized()
    rotations = (axis * (-phase / 2)).exp()
    # The reflection commutes with this boost; its two fixed circle points are at ±y.
    boost: Circle = reflector | circle.dual()
    glides = reflector * (boost * (glide_amplitude * np.sin(phase))).exp()

    grid: Point = core.disk_samples(mv.x, origin, infinity, axis, radii, angles)
    moving_grid: Point = stack([rotations[:, None, None] >> grid,
                               glides[:, None, None] >> grid], axis=1)
    fixed_points: Point = stack([(mv.y + mv.e).dual(), (-mv.y + mv.e).dual()])
    return moving_grid, fixed_points


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.orientation.conformal_circle import render

    save_figure(render.draw_circle(*circle()), "orientation_conformal_circle")
    duration_ms = 70
    save_animation(render.animate_circle_motions(*circle_motions()),
                   "orientation_circle_motions", duration_ms)
