"""A stretched, moved point cloud and the Gaussian fitted to it."""

from __future__ import annotations

import numpy as np

from functools import partial

from examples.quadrics.gaussian.core import density, fit_gaussian, mv

# The cloud: its size, its spread along x and y before it is moved, and the seed of its samples.
N_POINTS = 400
SPREAD = np.array([1.5, 0.5])
SEED = 4
# How it is moved: a translation along x and against y, then a turn, as generator coefficients.
SHIFT = (0.6, 0.3)
TURN = 0.3


# --- math -----------------------------------------------------------------------------
def gaussian():
    """A Gaussian fits a stretched cloud, moved as a batch: the cloud, its precision and its 1σ ellipse."""
    rng = np.random.default_rng(SEED)
    placement = (mv.xw * SHIFT[0] - mv.yw * SHIFT[1]).exp() * (mv.xy * TURN).exp()   # [] Motor+
    points = placement >> mv.point(rng.normal(size=(N_POINTS, 2)) * SPREAD)    # [points] Point-
    precision, ellipse = fit_gaussian(points)                                  # [] Line+ <- Point-, Line+ <- Point-
    return points, precision, ellipse


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.quadrics.gaussian import render

    points, precision, ellipse = gaussian()
    save_figure(render.draw_gaussian(points, partial(density, precision), ellipse), "gaussian")

    # checks
    # On unit-weight points the level is the squared Mahalanobis distance less one and the density
    # `(-0.5 * squared_distance).exp()`: the 1σ quadric is exactly the contour where the density falls to
    # `np.exp(-0.5)`.
    level = ellipse(points) & points                                           # [points] Scalar+
    np.testing.assert_allclose(level.to_array(), -2.0 * np.log(density(precision, points).to_array()) - 1.0, atol=1e-8)


if __name__ == "__main__":
    main()
