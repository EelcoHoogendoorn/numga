"""Scenes for point-cloud registration compare centered sandwich alignment and the one-sided motor residual on the same noisy point correspondences."""

from __future__ import annotations

import numpy as np

from examples.geometry.registration.core import Point, cloud, fit_motor, fit_motor_alignment, jitter, mv


def correspondences() -> tuple[Point, Point]:
    """The same seeded, noisy rigid correspondences for both fitting methods."""
    rng = np.random.default_rng(0)
    source = cloud(60, rng)
    truth = (mv.xw * 0.75 - mv.yw * 0.25 + mv.zw).exp() * (mv.xy * 0.4 - mv.yz * 0.3 + mv.zx * 0.7).exp()
    target = jitter(truth >> source, 0.02, rng)
    return source, target


def sandwich_alignment():
    """Cartesian least squares: center, fit rotation, then match centroids."""
    source, target = correspondences()
    estimate = fit_motor_alignment(source, target)
    return source, target, estimate >> source


def one_sided_residual():
    """Coefficient least squares: fit and normalize a motor in one eigenproblem."""
    source, target = correspondences()
    estimate = fit_motor(source, target)
    return source, target, estimate >> source


if __name__ == "__main__":
    from examples.animation import save_figure
    from examples.geometry.registration import render

    save_figure(render.draw_registration(*sandwich_alignment(), "Centered sandwich alignment"),
                "registration_sandwich_alignment")
    save_figure(render.draw_registration(*one_sided_residual(), "One-sided motor residual"),
                "registration_one_sided_residual")
