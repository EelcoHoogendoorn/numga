"""Scene setup and checks for four identical quadric ellipsoids along a geodesic on the 3-sphere."""

from __future__ import annotations

import numpy as np

from examples.quadrics.s3_raytracer.core import (
    direction,
    mv,
    origin,
    outlines,
    pixel_chart,
    placed_ellipsoids,
    project,
    reproject,
    walk_frames,
)

# Principal angular half-widths, geodesic distances along the line of sight, lateral and vertical offsets, spin angle, and RGB colours.
n_bodies = 4
half_widths = np.array([0.2, 0.28, 0.15])
distances = np.array([0.7, 1.4, 2.1, 2.6])
sideways = np.array([-0.4, 0.4, -0.4, 0.4])
upward = np.array([-0.3, -0.3, 0.3, 0.3])
turn = 0.8
colours = np.array([[0.9, 0.3, 0.3], [0.3, 0.8, 0.4], [0.3, 0.5, 0.95], [0.95, 0.8, 0.3]])

# Light direction coordinates, vertical field of view, and total arc walked along the forward geodesic.
light_coordinates = np.array([-0.4, 0.6, 0.7])
field_of_view = np.radians(80.0)
walk_distance = 1.2

# Animation frame count, pixel grid dimensions, frame duration, and coarse pixel grid for projection checks.
n_frames = 96
image_shape = (360, 480)
frame_duration_ms = 40
coarse_shape = (90, 120)
check_stride = 10


def scene_setup():
    """Four placed ellipsoid quadrics (`[4] Plane+ <- Point-`), RGB colours (`[4, 3]`), light point (`[] Point-`), and field of view form the static scene."""
    surfaces = placed_ellipsoids(half_widths, distances, sideways, upward, turn)  # [4] Plane+ <- Point-
    light = direction(light_coordinates)                                       # [] Point-
    return surfaces, colours, light, field_of_view


def main() -> None:
    """Renders the initial view and geodesic walk animation to `plots/` and verifies the conic projection."""
    from examples.quadrics.s3_raytracer import render
    from examples.util.animation import save_animation, save_figure

    surfaces, body_colours, light, fov = scene_setup()
    recorded_frames: list = []

    def streaming_walk():
        for frame in walk_frames(walk_distance, n_frames):
            recorded_frames.append(frame)
            yield frame

    save_animation(
        render.walk_animation(surfaces, body_colours, light, fov, image_shape, streaming_walk()),
        "s3_raytracer",
        frame_duration_ms,
    )
    save_figure(
        render.walk_figure(recorded_frames[0], surfaces, body_colours, light, fov, image_shape),
        "s3_raytracer",
    )

    # checks
    # A pixel lies inside a body's tangent outline cone whenever its great circle `origin + chart * t` meets
    # the surface with positive quadratic discriminant; the screen conic matches that normalized discriminant
    # and every reprojected hit lies on its quadric surface.
    n_coarse_pixels = coarse_shape[0] * coarse_shape[1]
    chart = pixel_chart(fov, coarse_shape)                                     # [n_coarse_pixels] ScreenPoint-
    cones = outlines(recorded_frames[0], surfaces).reshape(-1, 1)              # [n_bodies, 1] Plane+ <- Point-
    covered = (chart & cones(chart)) >= 0                                      # [n_bodies, n_coarse_pixels]
    eye_polars = surfaces.reshape(-1, 1)(origin)                               # [n_bodies, 1] Plane+
    ray_polars = surfaces.reshape(-1, 1)(chart)                                # [n_bodies, n_coarse_pixels] Plane+
    quad_a = chart & ray_polars                                                # [n_bodies, n_coarse_pixels] Scalar+
    quad_b = chart & eye_polars                                                # [n_bodies, n_coarse_pixels] Scalar+
    quad_c = origin & eye_polars                                               # [n_bodies, 1] Scalar+
    discriminant = quad_b * quad_b - quad_a * quad_c                           # [n_bodies, n_coarse_pixels] Scalar+
    clear = discriminant.abs() > 1e-3 * discriminant.abs().max(axis=1, keepdims=True)
    assert np.array_equal(covered[clear], (discriminant >= 0)[clear])
    conics, polars = project(recorded_frames[0], surfaces)                     # [n_bodies] Scalar+ <- (ScreenPoint-, ScreenPoint-), Scalar+ <- ScreenPoint-
    conic_residual = conics.reshape(-1, 1)(chart, chart) + discriminant / (quad_c * quad_c)  # [n_bodies, n_coarse_pixels] Scalar+
    assert (conic_residual.abs() < 1e-6).all()
    for body_index in range(n_bodies):
        hit_indices = np.nonzero(covered[body_index])[0][::check_stride]
        depths = reproject(conics[body_index], polars[body_index], chart[hit_indices])  # [...] Scalar+
        hits = (origin * depths + chart[hit_indices]).normalized()             # [...] Point-
        assert ((hits & surfaces[body_index](hits)).abs() < 1e-6).all()


if __name__ == "__main__":
    main()
