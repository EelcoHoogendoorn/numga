"""Scenes for epipolar geometry and two-view 3D reconstruction.

One function per figure. Builds the synthetic scene, injects sensor noise, calls the
mathematics in `core`, and returns the resulting geometry for `render`.
"""

from __future__ import annotations

import numpy as np

from examples.geometry.epipolar.core import Camera, Line, Point, direction, house_landmarks, mv, reconstruct


def epipolar():
    """Two cameras view a house; from their noisy images recover the relative pose and the house."""
    # The house, in camera 1's frame:
    landmarks = house_landmarks()                     # [n] Point
    n_points = landmarks.shape[0]

    # Camera 1 is at the world origin, looking along +z at the screen z == 1:
    origin = mv.zyx
    screen = mv.z - mv.w
    camera: Camera = (origin & Point) ^ screen

    # Camera 2 ground truth pose: rotated 14 degrees around y, translated along [0.65, 0.08, 0.18]:
    theta = np.radians(14.0)
    true_rot = (mv.zx * (theta * 0.5)).exp()
    true_trans = (mv.xw * 0.65 + mv.yw * 0.08 + mv.zw * 0.18) * 0.5
    true_motor = (true_trans.exp() * true_rot).normalized()

    # Each camera images the landmarks in its own frame, onto its own screen. Dividing by
    # the weight, the pairing with the plane at infinity, gives the image points unit weight:
    projected_1 = camera(landmarks)
    projected_2 = camera(true_motor << landmarks)
    image_1 = projected_1 / (mv.w & projected_1)      # [n] Point
    image_2 = projected_2 / (mv.w & projected_2)

    # Realistic sensor measurement noise (standard deviation 0.0015, ~1.5 pixels on a 1000px sensor),
    # a displacement within the screen:
    rng = np.random.default_rng(42)
    noise_sigma = 0.0015
    in_screen = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    noisy_1 = image_1 + direction(rng.normal(0.0, noise_sigma, size=(n_points, 2)) @ in_screen)
    noisy_2 = image_2 + direction(rng.normal(0.0, noise_sigma, size=(n_points, 2)) @ in_screen)

    # Measured sight rays, each in its own camera's frame:
    rays_1 = (origin & noisy_1).normalized()
    rays_2 = (origin & noisy_2).normalized()

    # Jointly solve for relative camera pose and 3D world points from the noisy measurements,
    # starting from the right baseline without any rotation:
    est_motor, reconstructed = reconstruct(rays_1, rays_2, true_trans.exp().normalized(), 10)

    # The epipolar lines on camera 2's screen are the images of camera 1's rays: the line
    # camera is the same join-then-meet with a line in the open slot.
    line_camera = (origin & Line) ^ screen
    epipolar_lines_2 = line_camera(est_motor << rays_1)

    # --- checks
    # The noise limits the recovery: the rotation comes back within half a degree, each
    # rotated axis plane agreeing with the truth.
    axes = mv("x y z", np.eye(3))
    turned = (est_motor >> axes).normalized() | (true_motor >> axes).normalized()
    assert np.all(turned.to_array() > np.cos(np.radians(0.5)))
    # The baseline direction comes back within two degrees; its length is not observable.
    true_baseline = origin & (true_motor >> origin)
    est_baseline = origin & (est_motor >> origin)
    assert (-(true_baseline.normalized() | est_baseline.normalized())).to_array() > np.cos(np.radians(2.0))
    # Scaled about camera 1 to the true baseline, the house comes back within 10 cm RMS.
    scale = true_baseline.norm() / est_baseline.norm()
    # The difference of two unit points is a direction; its length is the norm of its complement.
    unit = reconstructed / (mv.w & reconstructed)
    error = ((unit * scale + origin * (1.0 - scale)) - landmarks).dual()
    assert error.norm_squared().mean(axis=0).square_root().to_array() < 0.1

    return landmarks, image_1, image_2, noisy_1, noisy_2, epipolar_lines_2, reconstructed, true_motor, est_motor


if __name__ == "__main__":
    from examples.animation import save_figure
    from examples.geometry.epipolar import render

    save_figure(render.draw_epipolar(*epipolar()), "epipolar_reconstruction")
