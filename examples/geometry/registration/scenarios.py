"""Scenes for point-cloud registration compare centered sandwich alignment and the one-sided motor residual on the same noisy point correspondences."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.geometry.registration.core import Point, cloud, fit_motor, fit_motor_alignment, jitter, mv

# The source cloud: its points, its spreads along x, y and z, and the seed of its noise; the true motion, a
# shift after a turn; and the noise that moves each target point.
N_POINTS = 60
CLOUD_SPREAD = np.array([2.0, 1.0, 0.5])
SEED = 0
SHIFT = np.array([1.5, -0.5, 2.0])
TURN = np.array([0.8, -0.6, 1.4])
NOISE = 0.02


# --- math -----------------------------------------------------------------------------
def true_motor():
    """The true motion: a shift after a turn, each the exponential of half its generator."""
    shift = (stack((mv.xw, mv.yw, mv.zw)) * SHIFT).sum(axis=0) / 2
    turn = (stack((mv.xy, mv.yz, mv.zx)) * TURN).sum(axis=0) / 2
    return shift.exp() * turn.exp()


def registration() -> tuple[Point, Point, Point, Point]:
    """The same noisy rigid correspondences fitted both ways: the source, the target, and the source moved
    by the centered sandwich alignment and by the one-sided motor residual."""
    rng = np.random.default_rng(SEED)
    source = cloud(N_POINTS, CLOUD_SPREAD, rng)                                # [points] Point
    target = jitter(true_motor() >> source, NOISE, rng)                        # [points] Point
    return source, target, fit_motor_alignment(source, target) >> source, fit_motor(source, target) >> source


def squared_distances(points: Point, targets: Point):
    """The summed squared distances of points of unit weight from their targets: the squared norms of their
    differences' complements."""
    return ((points.normalized() - targets.normalized()).dual().norm_squared()).sum(axis=0)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.geometry.registration import render

    source, target, sandwich, one_sided = registration()
    save_figure(render.draw_registration(source, target, sandwich), "registration_sandwich_alignment")
    save_figure(render.draw_registration(source, target, one_sided), "registration_one_sided_residual")

    # checks
    # Centering then aligning minimizes the summed squared point distances, so the one-sided fit cannot beat
    # it on that measure; both come within the noise of the target, a few centimetres per point.
    cartesian, coefficient = squared_distances(sandwich, target), squared_distances(one_sided, target)
    assert cartesian.to_array() <= coefficient.to_array() * (1 + 1e-12)
    assert np.sqrt(coefficient.to_array() / N_POINTS) < 3 * NOISE * np.sqrt(3)


if __name__ == "__main__":
    main()
