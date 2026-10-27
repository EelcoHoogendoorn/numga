"""Scenes for the curvature of quadric surfaces shade an ellipsoid and a hyperboloid of one sheet by their Gaussian curvature and trace their lines of curvature as level sets of their confocal parameters."""

from __future__ import annotations

import numpy as np

from examples.surfaces.surface_curvature import core

# The ellipsoid's semi-axes, longest to shortest, and the hyperboloid's, with one negative sign for a saddle.
SEMI_AXES = np.array([3.0, 2.0, 1.2])
HYPERBOLOID_AXES = np.array([1.6, 1.0, 1.0])
HYPERBOLOID_SIGNS = np.array([1.0, 1.0, -1.0])
# Each surface's camera elevation and azimuth in degrees, half the screen width, and the pixels across the screen.
VIEWS = (((25, -60), 3.4), ((18, -60), 3.6))
N_PIXELS = 520
# The hyperboloid runs to infinity and is clipped to |z| = 2; the curvature scale sets where the colours saturate.
HEIGHTS = (np.inf, 2.0)
CURVATURE_SCALE = 1.2


# --- math -----------------------------------------------------------------------------
def surfaces() -> tuple[core.Quadric, core.Quadric]:
    """The ellipsoid and the hyperboloid of one sheet."""
    return core.quadric(1 / SEMI_AXES**2), core.quadric(HYPERBOLOID_SIGNS / HYPERBOLOID_AXES**2)  # [] Plane+ <- Point-, Plane+ <- Point-


def curvature_lines(n_pixels: int):
    """For each surface, what its camera sees: the hits, the discriminants, the tangent planes and the
    heading, together with the principal curvatures and the confocal parameters at every hit."""
    camera_rays = [(surface, core.view_rays(*view, extent, n_pixels)) for surface, (view, extent) in zip(surfaces(), VIEWS)]
    hits = [(surface, heading, *core.hit(surface, origins, heading)) for surface, (origins, heading) in camera_rays]
    scenes = [(points, discriminant, surface(points), heading) for surface, heading, points, discriminant in hits]
    curvatures = [core.principal(surface, points) for surface, _, points, _ in hits]    # [pixels, pixels, curvatures] Scalar+, Scalar+
    parameters = [core.confocal(surface, points) for surface, _, points, _ in hits]     # [pixels, pixels, members] Scalar+, Scalar+
    return scenes, curvatures, parameters


def umbilics() -> core.Point:
    """The ellipsoid's four umbilics in the plane of its longest and shortest axes."""
    long_axis, middle_axis, short_axis = SEMI_AXES
    x_coord = long_axis * np.sqrt((long_axis**2 - middle_axis**2) / (long_axis**2 - short_axis**2))
    z_coord = short_axis * np.sqrt((middle_axis**2 - short_axis**2) / (long_axis**2 - short_axis**2))
    return core.mv.point([[x_coord, 0, z_coord], [-x_coord, 0, z_coord], [x_coord, 0, -z_coord], [-x_coord, 0, -z_coord]])  # [umbilics] Point-


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.surfaces.surface_curvature import render

    save_figure(render.draw_curvature_lines(*curvature_lines(N_PIXELS), HEIGHTS, CURVATURE_SCALE), "surface_curvature")

    # checks
    # At the tip of the long axis the principal curvatures equal the long semi-axis divided by each other semi-axis squared.
    ellipsoid, _ = surfaces()
    tip_curvatures = core.principal(ellipsoid, core.mv.point([SEMI_AXES[0], 0.0, 0.0]))
    np.testing.assert_allclose(tip_curvatures.kernel[:, 0], SEMI_AXES[0] / SEMI_AXES[1:]**2, rtol=1e-8)
    # At the umbilics the ellipsoid curves equally in every direction, and the two confocal parameters meet.
    curvatures, parameters = core.principal(ellipsoid, umbilics()), core.confocal(ellipsoid, umbilics())
    np.testing.assert_allclose(curvatures[:, 0].kernel, curvatures[:, 1].kernel, rtol=1e-8)
    np.testing.assert_allclose(parameters[:, 0].kernel, parameters[:, 1].kernel, rtol=1e-6)


if __name__ == "__main__":
    main()
