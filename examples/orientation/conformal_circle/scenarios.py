"""A unit circle reflected in its own plane, and two motions that carry it into itself."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack

from examples.orientation.conformal_circle import core
from examples.orientation.conformal_circle.core import Point, Sphere, mv

# The duration of each animation frame in milliseconds.
DURATION_MS = 70


def circle() -> tuple[Point, Point, Point, Sphere, Sphere]:
    """Ordered circle points stay fixed while one cutting surface changes side, beside their reflections."""
    # Three points of the unit circle in the plane z = 0, at x, at -y and at -x.
    points = stack([mv.x.dual() + mv.e.dual(), -mv.y.dual() + mv.e.dual(),
                    -mv.x.dual() + mv.e.dual()])                                # [3] Point-
    plane = mv.z                                                                # [] Sphere+
    sphere = mv.w                                                               # [] Sphere+
    mirror = plane                                                              # [] Sphere+
    n_samples, n_meridians, n_parallels = 96, 16, 9
    angles = np.linspace(0, 2 * np.pi, n_samples + 1)
    longitudes = np.linspace(0, 2 * np.pi, n_meridians + 1)
    latitudes = np.linspace(-np.pi / 2, np.pi / 2, n_parallels)

    turns = (mv.xy * (angles / 2)).exp()                                        # [samples + 1] Rotor+
    curve = turns >> points[0]                                                  # [samples + 1] Point-
    surface = core.sphere_samples(points[0], mv.xy, mv.xz, longitudes, latitudes)   # [parallels, meridians + 1] Point-
    # Each part of the scene beside its reflection.
    markers = stack([points, mirror >> points])                                 # [sides, 3] Point-
    curves = stack([curve, mirror >> curve])                                    # [sides, samples + 1] Point-
    surfaces = stack([surface, mirror >> surface])                              # [sides, parallels, meridians + 1] Point-
    planes = stack([plane, mirror >> plane])                                    # [sides] Sphere+
    spheres = stack([sphere, mirror >> sphere])                                 # [sides] Sphere+
    return markers, curves, surfaces, planes, spheres


def circle_motions() -> tuple[Iterator[Point], Point]:
    """Rigid circulation and a hyperbolic glide reflection preserving the same circle."""
    points = stack([mv.x.dual() + mv.e.dual(), -mv.y.dual() + mv.e.dual(),
                    -mv.x.dual() + mv.e.dual()])                                # [3] Point-
    origin = (mv.e.dual() - mv.w.dual()) / 2                                    # [] Point-
    infinity = mv.e.dual() + mv.w.dual()                                        # [] Point-
    # The sphere at infinity.
    horizon = mv.e + mv.w                                                       # [] Sphere+
    reflector = mv.x                                                            # [] Sphere+
    n_frames, n_radii, n_angles = 64, 40, 96
    glide_amplitude = 0.9
    phases = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    radii = np.linspace(0, 1, n_radii + 1)
    angles = np.linspace(0, 2 * np.pi, n_angles + 1)

    circle = (points[0] & points[1] & points[2]).normalized()                   # [] Path-
    # The sphere at infinity selects the Euclidean rotation about the circle's axis.
    axis = (circle.dual() | horizon).normalized()                               # [] Circle+
    rotations = (axis * (-phases / 2)).exp()                                    # [frames] Rotor+
    # The reflection commutes with this boost; its two fixed circle points are at ±y.
    boost = reflector | circle.dual()                                           # [] Circle+
    glides = reflector * (boost * (glide_amplitude * np.sin(phases))).exp()     # [frames] Odd+

    grid = core.disk_samples(mv.x.dual(), origin, infinity, axis, radii, angles)   # [radii + 1, angles + 1] Point-
    # Each frame, the grid carried by the rotation beside the grid carried by the glide.
    frames = (stack([rotations[frame] >> grid, glides[frame] >> grid]) for frame in range(n_frames))
    fixed_points = stack([mv.y.dual() + mv.e.dual(), -mv.y.dual() + mv.e.dual()])   # [2] Point-
    return frames, fixed_points


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.orientation.conformal_circle import render

    save_figure(render.draw_circle(*circle()), "orientation_conformal_circle")
    save_animation(render.animate_circle_motions(*circle_motions()),
                   "orientation_circle_motions", DURATION_MS)


if __name__ == "__main__":
    main()
