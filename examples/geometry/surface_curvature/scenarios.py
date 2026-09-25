"""Scenes for the curvature of quadric surfaces.

One function per figure. Builds the surfaces and the camera, calls the mathematics in `core`, and
returns the geometry for `render`.
"""

from __future__ import annotations

import numpy as np

from examples.geometry.surface_curvature import core


def curvature_lines(pixels: int):
    """An ellipsoid and a hyperboloid of one sheet, each pixel shaded by its principal curvatures
    and placed in the net of curvature lines by its confocal parameters."""
    a, b, c = 3.0, 2.0, 1.2
    ellipsoid = core.quadric(1 / np.array([a**2, b**2, c**2]))
    hyperboloid = core.quadric(1 / np.array([1.6**2, 1.0**2, -(1.0**2)]))

    scenes, curvatures, parameters = [], [], []
    for surface, view, extent in ((ellipsoid, (25, -60), 3.4), (hyperboloid, (18, -60), 3.6)):
        origins, heading = core.view_rays(*view, extent, pixels)
        hits, discriminant = core.hit(surface, origins, heading)     # [pixels, pixels] Point, Scalar
        scenes.append((hits, discriminant, surface(hits), heading))
        curvatures.append(core.principal(surface, hits))             # [pixels, pixels, principal] Scalar
        parameters.append(core.confocal(surface, hits))              # [pixels, pixels, members] Scalar

    # --- checks
    # At the tip of the long axis the principal curvatures are a / b**2 and a / c**2.
    tip = core.principal(ellipsoid, core.point(np.array([a, 0.0, 0.0])))
    np.testing.assert_allclose(tip.to_array(), [a / b**2, a / c**2], rtol=1e-8)
    # The four umbilics, where the ellipsoid curves equally in every direction, lie in the plane
    # of its longest and shortest axes; there the two curvatures agree.
    x, z = a * np.sqrt((a**2 - b**2) / (a**2 - c**2)), c * np.sqrt((b**2 - c**2) / (a**2 - c**2))
    umbilics = core.principal(ellipsoid, core.point(np.array([[x, 0, z], [-x, 0, z], [x, 0, -z], [-x, 0, -z]])))
    np.testing.assert_allclose(umbilics[:, 0].to_array(), umbilics[:, 1].to_array(), rtol=1e-8)
    # At the umbilics the two confocal parameters meet as well: the net of lines closes up there.
    meeting = core.confocal(ellipsoid, core.point(np.array([[x, 0, z], [-x, 0, z], [x, 0, -z], [-x, 0, -z]])))
    np.testing.assert_allclose(meeting[:, 0].to_array(), meeting[:, 1].to_array(), rtol=1e-6)

    return scenes, curvatures, parameters


if __name__ == "__main__":
    from examples.animation import save_figure
    from examples.geometry.surface_curvature import render

    save_figure(render.draw_curvature_lines(*curvature_lines(520), (np.inf, 2.0), 1.2), "surface_curvature")
