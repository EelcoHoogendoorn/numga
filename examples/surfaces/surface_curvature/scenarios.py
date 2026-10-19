"""Scenes for the curvature of quadric surfaces: an ellipsoid and a hyperboloid of one sheet, each pixel
shaded by its principal curvatures and placed in the net of curvature lines by its confocal parameters."""

from __future__ import annotations

import numpy as np

from examples.surfaces.surface_curvature import core

# The ellipsoid's semi-axes, longest to shortest, and the hyperboloid's, the last one imaginary: a saddle.
SEMI_AXES = np.array([3.0, 2.0, 1.2])
HYPERBOLOID_AXES = np.array([1.6, 1.0, 1.0])
HYPERBOLOID_SIGNS = np.array([1.0, 1.0, -1.0])
# Each surface's camera: its elevation and azimuth in degrees, and half the image's width; the pixels
# across the image.
VIEWS = (((25, -60), 3.4), ((18, -60), 3.6))
N_PIXELS = 520
# The hyperboloid runs to infinity: drawn up to |z| = 2. The curvature at which the colours saturate.
HEIGHTS = (np.inf, 2.0)
CURVATURE_SCALE = 1.2


# --- math -----------------------------------------------------------------------------
def surfaces() -> tuple[core.Quadric, core.Quadric]:
    """The ellipsoid and the hyperboloid of one sheet."""
    return core.quadric(1 / SEMI_AXES**2), core.quadric(HYPERBOLOID_SIGNS / HYPERBOLOID_AXES**2)


def curvature_lines(n_pixels: int):
    """For each surface, what its camera sees: the hits, which pixels hit, the tangent planes and the
    heading; the principal curvatures `[pixels, pixels, principal] Scalar` and the confocal parameters
    `[pixels, pixels, members] Scalar` at every hit."""
    seen = [(surface, core.view_rays(*view, extent, n_pixels)) for surface, (view, extent) in zip(surfaces(), VIEWS)]
    hits = [(surface, heading, *core.hit(surface, origins, heading)) for surface, (origins, heading) in seen]
    scenes = [(points, discriminant, surface(points), heading) for surface, heading, points, discriminant in hits]
    curvatures = [core.principal(surface, points) for surface, _, points, _ in hits]
    parameters = [core.confocal(surface, points) for surface, _, points, _ in hits]
    return scenes, curvatures, parameters


def umbilics() -> core.Point:
    """The ellipsoid's four umbilics, in the plane of its longest and shortest axes `[umbilics] Point`."""
    long, middle, short = SEMI_AXES
    along = long * np.sqrt((long**2 - middle**2) / (long**2 - short**2))
    up = short * np.sqrt((middle**2 - short**2) / (long**2 - short**2))
    return core.mv.point(np.array([[along, 0, up], [-along, 0, up], [along, 0, -up], [-along, 0, -up]]))


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_figure
    from examples.surfaces.surface_curvature import render

    save_figure(render.draw_curvature_lines(*curvature_lines(N_PIXELS), HEIGHTS, CURVATURE_SCALE), "surface_curvature")

    # checks
    # At the tip of the long axis the principal curvatures are the long semi-axis over each other one squared.
    ellipsoid, _ = surfaces()
    tip = core.principal(ellipsoid, core.mv.point(np.array([SEMI_AXES[0], 0.0, 0.0])))
    np.testing.assert_allclose(tip.to_array(), SEMI_AXES[0] / SEMI_AXES[1:]**2, rtol=1e-8)
    # At the umbilics the ellipsoid curves equally in every direction, and the two confocal parameters
    # meet: the net of lines closes up there.
    curvatures, parameters = core.principal(ellipsoid, umbilics()), core.confocal(ellipsoid, umbilics())
    np.testing.assert_allclose(curvatures[:, 0].to_array(), curvatures[:, 1].to_array(), rtol=1e-8)
    np.testing.assert_allclose(parameters[:, 0].to_array(), parameters[:, 1].to_array(), rtol=1e-6)


if __name__ == "__main__":
    main()
