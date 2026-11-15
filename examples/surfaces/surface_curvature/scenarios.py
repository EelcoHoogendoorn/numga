"""Scenes for the curvature of quadric surfaces shade an ellipsoid and a hyperboloid of one sheet by their Gaussian curvature and trace their lines of curvature as level sets of their confocal parameters."""

from __future__ import annotations

import numpy as np

from examples.surfaces.surface_curvature import core, render

# The ellipsoid's semi-axes, longest to shortest, and the hyperboloid's, with one negative sign for a saddle.
semi_axes = np.array([3.0, 2.0, 1.2])
hyperboloid_axes = np.array([1.6, 1.0, 1.0])
hyperboloid_signs = np.array([1.0, 1.0, -1.0])
# Each surface's camera elevation and azimuth in degrees, half the screen width, and the pixels across each panel.
views = (((25.0, -60.0), 3.4), ((18.0, -60.0), 3.6))
n_pixels = 520
# The hyperboloid runs to infinity and is clipped to |z| = 2; the curvature scale sets where the colours saturate.
heights = (np.inf, 2.0)
curvature_scale = 1.2


# --- math -----------------------------------------------------------------------------
def quadric_pair() -> tuple[core.Quadric, core.Quadric]:
    """The triaxial ellipsoid and the hyperboloid of one sheet are central polarities (`[] Plane+ <- Point-` each)."""
    ellipsoid = core.central_quadric(1 / semi_axes**2)                                  # [] Plane+ <- Point-
    hyperboloid = core.central_quadric(hyperboloid_signs / hyperboloid_axes**2)         # [] Plane+ <- Point-
    return ellipsoid, hyperboloid


def curvature_scenes(n_pixels: int):
    """Camera intersections, tangent planes, principal curvatures, and confocal parameters span `n_pixels` by `n_pixels` panels."""
    camera_rays = [(surface, render.view_rays(*view, extent, n_pixels)) for surface, (view, extent) in zip(quadric_pair(), views)]
    hits = [(surface, heading, *core.ray_intersections(surface, origins, heading)) for surface, (origins, heading) in camera_rays]
    scenes = [(points, discriminant, surface(points), heading) for surface, heading, points, discriminant in hits]
    curvatures = [core.principal_curvatures(surface, points) for surface, _, points, _ in hits]  # [n_pixels, n_pixels, 2] Scalar+
    parameters = [core.confocal_parameters(surface, points) for surface, _, points, _ in hits]   # [n_pixels, n_pixels, 2] Scalar+
    return scenes, curvatures, parameters


def ellipsoid_umbilics() -> core.Point:
    """The ellipsoid's four umbilic points (`[4] Point-`) lie in the plane of its longest and shortest semi-axes."""
    long_axis, middle_axis, short_axis = semi_axes
    x_coord = long_axis * np.sqrt((long_axis**2 - middle_axis**2) / (long_axis**2 - short_axis**2))
    z_coord = short_axis * np.sqrt((middle_axis**2 - short_axis**2) / (long_axis**2 - short_axis**2))
    return core.mv.point([
        [x_coord, 0.0, z_coord],
        [-x_coord, 0.0, z_coord],
        [x_coord, 0.0, -z_coord],
        [-x_coord, 0.0, -z_coord],
    ])                                                                                  # [4] Point-


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    """Saves the combined Gaussian curvature and curvature-line figure and verifies analytic ellipsoid landmarks."""
    from examples.util.animation import save_figure

    scenes, curvatures, parameters = curvature_scenes(n_pixels)
    save_figure(render.curvature_lines_figure(scenes, curvatures, parameters, heights, curvature_scale), "surface_curvature")

    # checks
    ellipsoid, _ = quadric_pair()
    # At the tip of the long axis the principal curvatures equal the long semi-axis divided by each other semi-axis squared.
    tip_curvatures = core.principal_curvatures(ellipsoid, core.mv.point([semi_axes[0], 0.0, 0.0]))  # [2] Scalar+
    expected_tip = semi_axes[0] / semi_axes[1:]**2
    assert ((tip_curvatures - expected_tip).abs() < 1e-8).all()
    # At the four umbilics the ellipsoid curves equally in every direction, and the two confocal parameters meet.
    umbilics = ellipsoid_umbilics()                                                     # [4] Point-
    umbilic_curvatures = core.principal_curvatures(ellipsoid, umbilics)                 # [4, 2] Scalar+
    umbilic_parameters = core.confocal_parameters(ellipsoid, umbilics)                  # [4, 2] Scalar+
    assert ((umbilic_curvatures[:, 0] - umbilic_curvatures[:, 1]).abs() < 1e-8).all()
    assert ((umbilic_parameters[:, 0] - umbilic_parameters[:, 1]).abs() < 1e-6).all()


if __name__ == "__main__":
    main()
