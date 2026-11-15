"""A displaced and rotated normal cloud tests homogeneous Gaussian fitting."""

from __future__ import annotations

from functools import partial

import numpy as np

from examples.quadrics.gaussian.core import Point, gaussian_density, gaussian_fit, mv

n_points = 400
spread = np.array([1.5, 0.5])
seed = 4
shift = (0.6, 0.3)
turn = 0.3


# --- math -----------------------------------------------------------------------------
def point_cloud() -> Point:
    """An anisotropic normal sample placed by a rigid motor forms the test cloud (`[n_points] Point-`)."""
    rng = np.random.default_rng(seed)
    rest_points = mv.point(rng.normal(size=(n_points, 2)) * spread)            # [n_points] Point-
    placement = (mv.xw * shift[0] - mv.yw * shift[1]).exp() * (mv.xy * turn).exp()  # [] Motor+
    return placement >> rest_points


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    """Fits the displaced point cloud, writes the figure, and checks quadric and density identities."""
    from examples.quadrics.gaussian import render
    from examples.util.animation import save_figure

    points = point_cloud()                                                     # [n_points] Point-
    precision, ellipsoid = gaussian_fit(points)                                # [] Plane+ <- Point-, Plane+ <- Point-
    save_figure(render.gaussian_figure(points, partial(gaussian_density, precision), ellipsoid), "gaussian")

    # checks
    mean = points.mean(axis=0)                                                 # [] Point-
    levels = ellipsoid(points) & points                                        # [n_points] Scalar+
    assert (gaussian_density(precision, mean) - 1).abs() < 1e-8
    assert ((ellipsoid(mean) & mean) + 1).abs() < 1e-8
    assert (levels.mean(axis=0) - 1).abs() < 1e-8
    assert ((levels + 2 * gaussian_density(precision, points).log() + 1).abs() < 1e-8).all()


if __name__ == "__main__":
    main()
