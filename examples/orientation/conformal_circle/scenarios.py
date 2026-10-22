"""A unit circle reflected in its own plane."""

from __future__ import annotations

import numpy as np

from numga import stack

from examples.orientation.conformal_circle import core
from examples.orientation.conformal_circle.core import Circle, Path, Point, Sphere, mv

# The duration of each animation frame in milliseconds.
DURATION_MS = 70


def circle() -> tuple[Point, Point, Point, Sphere, Sphere]:
    """Ordered circle points stay fixed while one cutting surface changes side."""
    points = stack([(mv.x + mv.e).dual(), (-mv.y + mv.e).dual(), (-mv.x + mv.e).dual()])  # Point
    plane = mv.z                                                                # Sphere
    sphere = mv.w                                                               # Sphere
    mirror = plane                                                              # Sphere
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
    points = stack([(mv.x + mv.e).dual(), (-mv.y + mv.e).dual(), (-mv.x + mv.e).dual()])  # Point
    infinity = mv.e + mv.w                                                      # Sphere
    origin = (mv.e - mv.w) / 2                                                  # Sphere
    reflector = mv.x                                                            # Sphere
    frames = 64
    radial_samples = 40
    angular_samples = 96
    glide_amplitude = 0.9
    phase = np.linspace(0, 2 * np.pi, frames, endpoint=False)
    radii = np.linspace(0, 1, radial_samples + 1)
    angles = np.linspace(0, 2 * np.pi, angular_samples + 1)

    circle = (points[0] & points[1] & points[2]).normalized()                   # Path
    # Infinity selects the Euclidean rotation about the circle's axis.
    axis = (circle.dual() | infinity).normalized()                              # Circle
    rotations = (axis * (-phase / 2)).exp()
    # The reflection commutes with this boost; its two fixed circle points are at ±y.
    boost = reflector | circle.dual()                                           # Circle
    glides = reflector * (boost * (glide_amplitude * np.sin(phase))).exp()

    grid = core.disk_samples(mv.x, origin, infinity, axis, radii, angles)       # Point
    moving_grid = stack([rotations[:, None, None] >> grid,                      # Point
                               glides[:, None, None] >> grid], axis=1)
    fixed_points = stack([(mv.y + mv.e).dual(), (-mv.y + mv.e).dual()])         # Point
    return moving_grid, fixed_points


def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.orientation.conformal_circle import render

    save_figure(render.draw_circle(*circle()), "orientation_conformal_circle")
    save_animation(render.animate_circle_motions(*circle_motions()),
                   "orientation_circle_motions", DURATION_MS)


if __name__ == "__main__":
    main()
