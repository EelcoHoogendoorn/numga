"""Scenes for epipolar geometry and two-view 3D reconstruction build a synthetic two-camera view of a wireframe house, inject sensor noise, and recover the relative pose and 3D landmarks."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import numpy as np

from numga import Extensor, stack
from examples.estimation.epipolar.core import Camera, Line, Motor, Point, mv, reconstruction, triangulate

# The house, in camera 1's frame, x to the right, y down and z ahead: the corners of its walls, from the ground up to
# the eaves, front then back, and its roof's ridge over the front and the back; and a grid of markers on the ground in
# front of it, by their counts across and ahead, half their width, their depths and the ground's level below the camera.
WALLS = [[-0.5, 0.6, 3.0], [0.5, 0.6, 3.0], [0.5, 0.0, 3.0], [-0.5, 0.0, 3.0],
         [-0.5, 0.6, 4.0], [0.5, 0.6, 4.0], [0.5, 0.0, 4.0], [-0.5, 0.0, 4.0]]
ROOF = [[0.0, -0.4, 3.0], [0.0, -0.4, 4.0]]
# The house's edges, by the landmarks they join: the front and back walls' rings and the four edges between them, and
# the roof's ridge and the rafters from its ends down to the eaves.
HOUSE_EDGES = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7],
               [8, 9], [8, 2], [8, 3], [9, 6], [9, 7]]
N_GROUND = (4, 3)
GROUND_HALF_WIDTH = 0.8
GROUND_DEPTHS = (2.0, 2.8)
GROUND_LEVEL = 0.6
# Camera 2's true pose: turned about y, then shifted.
TURN = np.radians(14.0)
SHIFT = np.array([0.65, 0.08, 0.18])
# Sensor noise on the screen at unit distance, about 1.5 pixels on a 1000 pixel sensor, and its seed.
NOISE = 0.0015
SEED = 42
# The Gauss-Newton steps; the animation's frames from one step to the next and held at the end, and the
# duration of a frame in milliseconds.
N_ITERATIONS = 4
N_EASING, N_HOLD = 10, 16
FRAME_DURATION = 50


# --- plumbing --------------------------------------------------------------
def house() -> Point:
    """The landmarks: the house's corners and the ground markers."""
    across = np.linspace(-GROUND_HALF_WIDTH, GROUND_HALF_WIDTH, N_GROUND[0])
    ahead = np.linspace(*GROUND_DEPTHS, N_GROUND[1])
    grid_x, grid_z = np.meshgrid(across, ahead)
    ground = np.stack([grid_x.ravel(), np.full(grid_x.size, GROUND_LEVEL), grid_z.ravel()], axis=-1)
    return mv.point(np.concatenate([WALLS, ROOF, ground]))                    # [landmarks] Point


# --- math ------------------------------------------------------------------
def epipolar(noise: float):
    """Two cameras view a house; from their images, with sensor noise of the given size, recover the
    relative pose and the house. Returns the landmarks, the images and their measurements in both views,
    the measured sight rays, over the iterations of the search the reconstructed landmarks, camera 2's true
    pose, and over the iterations its found pose."""
    landmarks = house()                                                       # [landmarks] Point

    # Camera 1 is at the world origin, looking along +z at the screen z == 1: a point's image is where the
    # sight ray through it, its join with the eye, meets the screen.
    origin = mv.w.dual()
    screen = mv.z - mv.w
    camera = (origin & Point) ^ screen                                        # [] Point <- Point

    true_shift = (stack((mv.xw, mv.yw, mv.zw)) * SHIFT).sum(axis=0) / 2       # [] Twist
    true_motor = (true_shift.exp() * (mv.zx * (TURN / 2)).exp()).normalized()   # Motor

    # Each camera images the landmarks in its own frame, onto its own screen. Dividing by the weight, the
    # pairing with the plane at infinity, gives the image points unit weight.
    projected = camera(Extensor.stack([landmarks, true_motor << landmarks]))  # [views, landmarks] Point
    images = projected / (mv.w & projected)                                   # [views, landmarks] Point

    # Sensor noise, a displacement within the screen.
    in_screen = mv.direction([[1, 0, 0], [0, 1, 0]])                          # [2] Point
    offsets = np.random.default_rng(SEED).normal(0.0, noise, size=images.shape + (2,))
    measured = images + (in_screen * offsets).sum(axis=-1)                    # [views, landmarks] Point

    # Measured sight rays, each in its own camera's frame. Jointly solve for the relative pose and the
    # world points, starting from the right baseline without any rotation.
    rays = (origin & measured).normalized()                                   # [views, landmarks] Line
    iterates = islice(reconstruction(rays[0], rays[1], true_shift.exp().normalized()), N_ITERATIONS + 1)
    est_motors, reconstructed = (Extensor.stack(path) for path in zip(*iterates))   # [iterations] Motor, [iterations, landmarks] Point

    return landmarks, images, measured, rays, reconstructed, true_motor, est_motors


def easing(rays: Line, est_motors: Motor, n_easing: int, n_hold: int) -> Iterator[tuple[Motor, Point]]:
    """The search drawn as a motion: camera 2's pose eased from one Gauss-Newton iterate to the next along
    the screw between them, then held, with the world points the pose triangulates."""
    fractions = np.arange(n_easing) / n_easing
    for start, end in zip(est_motors[:-1], est_motors[1:]):
        # The screw from one pose to the next, its log, taken a fraction of the way.
        screw = (end * start.inverse()).log()                                 # [] Twist
        for fraction in fractions:
            motor = ((screw * fraction).exp() * start).normalized()
            yield motor, triangulate(rays[0], rays[1], motor)
    for _ in range(n_hold):
        yield est_motors[-1], triangulate(rays[0], rays[1], est_motors[-1])


def errors(landmarks: Point, reconstructed: Point, true_motor: Motor, est_motor: Motor):
    """How far the recovery is from the truth: the cosines between the true and found turns of the axis
    planes and between the baselines, and the house's RMS error once scaled about camera 1 to the true
    baseline, the baseline's length being unobservable."""
    axes = stack((mv.x, mv.y, mv.z))                                          # [3] Plane
    alignment = (est_motor >> axes).normalized() | (true_motor >> axes).normalized()   # [3] Scalar
    origin = mv.w.dual()
    true_baseline = origin & (true_motor >> origin)                           # [] Line
    est_baseline = origin & (est_motor >> origin)                             # [] Line
    agreement = true_baseline.normalized().dot(est_baseline.normalized())     # [] Scalar
    scale = true_baseline.norm() / est_baseline.norm()                        # [] Scalar
    unit = reconstructed / (mv.w & reconstructed)                             # [landmarks] Point
    # The difference of two unit points is a direction; its length is the norm of its complement.
    error = ((unit * scale + origin * (1.0 - scale)) - landmarks).dual()      # [landmarks] Plane
    return alignment, agreement, error.norm_squared().mean(axis=0).square_root()


def main() -> None:
    from examples.animation import save_animation
    from examples.estimation.epipolar import render

    landmarks, images, measured, rays, reconstructed, true_motor, est_motors = epipolar(NOISE)
    frames = render.animate_reconstruction(landmarks, np.array(HOUSE_EDGES), true_motor, easing(rays, est_motors, N_EASING, N_HOLD))
    save_animation(frames, "epipolar_reconstruction", FRAME_DURATION)

    # checks
    # The noise limits the recovery: each axis plane turns within half a degree of the truth, the baseline
    # direction comes back within two degrees, and the house within 10 cm RMS.
    alignment, agreement, rms = errors(landmarks, reconstructed[-1], true_motor, est_motors[-1])
    assert np.all(alignment.to_array() > np.cos(np.radians(0.5)))
    assert agreement.to_array() > np.cos(np.radians(2.0))
    assert rms.to_array() < 0.1


if __name__ == "__main__":
    main()
