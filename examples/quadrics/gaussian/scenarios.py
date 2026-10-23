"""A stretched, moved point cloud and the Gaussian fitted to it."""

from __future__ import annotations

import numpy as np

from examples.quadrics.gaussian.core import fit_gaussian, mv

# The cloud: its size, its spread along x and y before it is moved, and the seed of its samples.
N_POINTS = 400
SPREAD = np.array([1.5, 0.5])
SEED = 4
# How it is moved: a translation along x and against y, then a turn, as generator coefficients.
SHIFT = (0.6, 0.3)
TURN = 0.3
# The pixels the fit is evaluated on: their count across, and the box they cover along x and y.
N_ACROSS = 320
BOX = np.array([[-5.0, -5.0], [6.0, 4.0]])


# --- math -----------------------------------------------------------------------------
def gaussian():
    """Fit a Gaussian to a stretched cloud, moved as a batch, on points covering the plot."""
    rng = np.random.default_rng(SEED)
    placement = (mv.xw * SHIFT[0] - mv.yw * SHIFT[1]).exp() * (mv.xy * TURN).exp()
    points = placement >> mv.point(rng.normal(size=(N_POINTS, 2)) * SPREAD)    # [points] Point
    pixels = mv.point(np.stack(np.meshgrid(np.linspace(BOX[0, 0], BOX[1, 0], N_ACROSS),
                                           np.linspace(BOX[0, 1], BOX[1, 1], N_ACROSS)), axis=-1))   # [rows, columns] Point
    density, level = fit_gaussian(points, pixels)                              # [rows, columns] Scalar each
    return points, pixels, density, level


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.quadrics.gaussian import render

    points, pixels, density, level = gaussian()
    save_figure(render.draw_gaussian(points, pixels, density, level), "gaussian")

    # checks
    # On unit-weight points the level is the squared Mahalanobis distance less one and the density
    # `(-0.5 * squared_distance).exp()`: the 1σ quadric is exactly the contour where the density falls to
    # `np.exp(-0.5)`.
    np.testing.assert_allclose(level.to_array(), -2.0 * np.log(density.to_array()) - 1.0, atol=1e-8)


if __name__ == "__main__":
    main()
