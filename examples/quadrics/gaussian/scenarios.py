"""A stretched, moved point cloud and the Gaussian fitted to it."""

from __future__ import annotations

import numpy as np

from examples.quadrics.gaussian.core import fit_gaussian, mv, point


def gaussian():
    """Fit a Gaussian to a stretched cloud, moved as a batch, on points covering the plot."""
    rng = np.random.default_rng(4)
    points = point(rng.normal(size=(400, 2)) * [1.5, 0.5])
    placement = (mv.xw * 0.6 - mv.yw * 0.3).exp() * (mv.xy * 0.3).exp()
    points = placement >> points
    across = 320
    pixels = point(np.stack(np.meshgrid(np.linspace(-5, 6, across),
                                        np.linspace(-5, 4, across)), axis=-1))

    density, level = fit_gaussian(points, pixels)

    # --- checks ---------------------------------------------------------------------------
    # On unit-weight points the level is squared_distance - 1 and the density
    # (-0.5 * squared_distance).exp(): the 1σ quadric is exactly the contour where the
    # density falls to np.exp(-0.5).
    np.testing.assert_allclose(level.to_array(), -2.0 * np.log(density.to_array()) - 1.0, atol=1e-8)
    return points, pixels, density, level


if __name__ == "__main__":
    from examples.animation import save_figure
    from examples.quadrics.gaussian import render

    save_figure(render.draw_gaussian(*gaussian()), "gaussian")
