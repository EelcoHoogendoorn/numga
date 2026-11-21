"""Synthetic two-camera scenes image a wireframe house under sensor noise and recover camera 2's relative pose and the 3D landmarks."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import numpy as np

from numga import dim, stack
from examples.estimation.epipolar.core import Line, Motor, Point, Scalar, camera_scale, mv, reconstruction_states, triangulated_points

# House wall corners, roof ridge corners, wireframe edge pairs, and ground-marker grid in camera 1's frame (+x right, +y down, +z forward).
walls = [
    [-0.5, 0.6, 3.0], [0.5, 0.6, 3.0], [0.5, 0.0, 3.0], [-0.5, 0.0, 3.0],
    [-0.5, 0.6, 4.0], [0.5, 0.6, 4.0], [0.5, 0.0, 4.0], [-0.5, 0.0, 4.0],
]
roof = [[0.0, -0.4, 3.0], [0.0, -0.4, 4.0]]
house_edges = np.array([
    [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7], [8, 9], [8, 2], [8, 3], [9, 6], [9, 7],
])
n_ground_across, n_ground_ahead = 4, 3
ground_half_width = 0.8
ground_depths = (2.0, 2.8)
ground_level = 0.6
# Camera 2's true rotation about y and baseline translation, screen noise standard deviation (~1.5 px on 1000 px), and seed.
turn_angle = np.radians(14.0)
shift_vector = np.array([0.65, 0.08, 0.18])
sensor_noise = 0.0015
noise_seed = 42
# Gauss-Newton step count, screw-easing and final-hold frame counts, and smooth frame duration in milliseconds.
n_iterations = 4
n_easing, n_hold = 12, 22
frame_duration_ms = 40


# --- plumbing --------------------------------------------------------------
def house_landmarks() -> Point:
    """Wall corners, roof ridge corners, and ground markers form the 22 scene landmarks (`[22] Point-`)."""
    across = np.linspace(-ground_half_width, ground_half_width, n_ground_across)
    ahead = np.linspace(*ground_depths, n_ground_ahead)
    grid_x, grid_z = np.meshgrid(across, ahead)
    ground = np.stack([grid_x.ravel(), np.full(grid_x.size, ground_level), grid_z.ravel()], axis=-1)
    return mv.point(np.concatenate([walls, roof, ground]), unit=dim.length)   # [22] Point-, 1 | length


def eased_states(rays: Line, estimated_motors: Motor, n_easing: int, n_hold: int) -> Iterator[tuple[Motor, Point]]:
    """Yields camera 2's motor (`[] Motor+`) and triangulated landmarks (`[...] Point-`) interpolated along the screw between iterates and held at the final pose."""
    fractions = np.arange(n_easing) / n_easing
    for start, end in zip(estimated_motors[:-1], estimated_motors[1:]):
        screw = (end * start.inverse()).log()                                 # [] Twist+
        for fraction in fractions:
            step_motor = (screw * fraction).exp()                             # [] Motor+
            motor = (step_motor * start).normalized()                         # [] Motor+
            yield motor, triangulated_points(rays[0], rays[1], motor)
    for _ in range(n_hold):
        yield estimated_motors[-1], triangulated_points(rays[0], rays[1], estimated_motors[-1])


# --- math ------------------------------------------------------------------
def epipolar_scene(
    noise: float,
    n_steps: int,
    seed: int,
) -> tuple[Point, Point, Point, Line, Point, Motor, Motor]:
    """Two cameras image `house_landmarks()` under screen noise `noise`, and `n_steps` Gauss–Newton updates produce the landmarks, clean and noisy images, sight rays (`[2, 22] Line-`), reconstructed landmarks (`[n_steps + 1, 22] Point-`), true motor, and estimated motors (`[n_steps + 1] Motor+`)."""
    landmarks = house_landmarks()                                             # [22] Point-
    n_landmarks = landmarks.shape[0]

    # Camera 1 sits at the origin looking along +z onto the image plane z == camera_scale.
    origin = mv.w.dual()                                                      # [] Point-
    screen = mv.z - camera_scale * mv.w                                       # [] Plane+, 1 | length
    camera = (origin & Point) ^ screen                                        # [] Point- <- Point-

    shift_generator = ((stack((mv.xw, mv.yw, mv.zw)) * shift_vector).sum(axis=0) / 2) * dim.length  # [] Twist+, length
    shift_motor = shift_generator.exp()                                       # [] Motor+, 1 | length
    turn_motor = (mv.zx * (turn_angle / 2)).exp()                             # [] Motor+
    true_motor = (shift_motor * turn_motor).normalized()                      # [] Motor+, 1 | length

    # Dividing each projected point by its projective weight places all image points at unit weight on z == camera_scale.
    raw_images = camera(stack([landmarks, true_motor << landmarks]))          # [2, n_landmarks] Point-
    images = raw_images / (mv.w & raw_images)                                 # [2, n_landmarks] Point-, 1 | length

    screen_axes = mv.direction([[1, 0, 0], [0, 1, 0]], unit=dim.length)       # [2] Point-, length
    offsets = np.random.default_rng(seed).normal(0.0, noise, size=images.shape + (2,))
    measured_images = images + (screen_axes * offsets).sum(axis=-1)           # [2, n_landmarks] Point-, 1 | length

    rays = (origin & measured_images).normalized()                            # [2, n_landmarks] Line-
    initial_motor = shift_motor.normalized()                                  # [] Motor+
    iterates = islice(reconstruction_states(rays[0], rays[1], initial_motor), n_steps + 1)
    estimated_motors, reconstructed_landmarks = (stack(path) for path in zip(*iterates))  # [n_steps + 1] Motor+, [n_steps + 1, n_landmarks] Point-

    return landmarks, images, measured_images, rays, reconstructed_landmarks, true_motor, estimated_motors


def pose_and_landmark_errors(
    landmarks: Point,
    reconstructed_landmarks: Point,
    true_motor: Motor,
    estimated_motor: Motor,
) -> tuple[Scalar, Scalar, Scalar]:
    """Axis-plane cosines (`[3] Scalar+`), baseline-direction cosine (`[] Scalar+`), and baseline-scaled landmark RMS error (`[] Scalar+`) compare `estimated_motor` and `reconstructed_landmarks` (`[...] Point-`) against ground truth."""
    axes = stack((mv.x, mv.y, mv.z))                                          # [3] Plane+
    axis_cosines = (estimated_motor >> axes).normalized() | (true_motor >> axes).normalized()  # [3] Scalar+
    origin = mv.w.dual()                                                      # [] Point-
    # Subtracting two unit-weight points yields the ideal displacement direction along each camera baseline.
    true_baseline = (true_motor >> origin) - origin                           # [] Point-
    estimated_baseline = (estimated_motor >> origin) - origin                 # [] Point-
    baseline_cosine = true_baseline.dot(estimated_baseline) / (true_baseline.antinorm() * estimated_baseline.antinorm())  # [] Scalar+
    scale = true_baseline.antinorm() / estimated_baseline.antinorm()          # [] Scalar+
    unit_landmarks = reconstructed_landmarks / (mv.w & reconstructed_landmarks)  # [...] Point-
    displacements = (unit_landmarks * scale + origin * (1.0 - scale)) - landmarks  # [...] Point-
    landmark_rms = displacements.dot(displacements).mean(axis=0).square_root()  # [] Scalar+
    return axis_cosines, baseline_cosine, landmark_rms


# --- plumbing --------------------------------------------------------------
def main() -> None:
    """Writes the ray-traced reconstruction animation to `plots/` and checks pose and landmark recovery."""
    from examples.util.animation import save_animation
    from examples.estimation.epipolar import render

    landmarks, _, _, rays, reconstructed_landmarks, true_motor, estimated_motors = epipolar_scene(
        sensor_noise, n_iterations, noise_seed,
    )
    frames = render.reconstruction_animation(
        landmarks, house_edges, true_motor, eased_states(rays, estimated_motors, n_easing, n_hold),
    )
    save_animation(frames, "epipolar_reconstruction", frame_duration_ms)

    # checks
    axis_cosines, baseline_cosine, landmark_rms = pose_and_landmark_errors(
        landmarks, reconstructed_landmarks[-1], true_motor, estimated_motors[-1],
    )
    assert (axis_cosines > np.cos(np.radians(0.5))).all()
    assert (baseline_cosine > np.cos(np.radians(2.0))).all()
    assert (landmark_rms < mv.scalar(0.1, unit=dim.length)).all()


if __name__ == "__main__":
    main()
