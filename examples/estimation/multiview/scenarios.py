"""Planar multi-view reconstruction scenarios place three convergent cameras in front of six landmarks and recover misplaced camera poses by alternating or joint Gauss-Newton alignment.

One camera on the left and one on the right turn inward while a third camera between them looks straight ahead. Each scenario misplaces one or more cameras and walks them back into alignment, yielding geometric extensors per step for `render`.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from numga.algebras import PGA2D

from examples import instantiate

core = instantiate("examples.estimation.multiview.core", PGA2D)
Camera, Information, Motor, Point, Quadric = core.Camera, core.Information, core.Motor, core.Point, core.Quadric
mv = core.mv

# Six scene landmarks in front of the three-camera rig.
landmark_coordinates = [[0.15, 0.85], [-0.43, 1.15], [0.50, 1.50], [0.03, 1.85], [0.65, 2.20], [-0.60, 2.55]]
# Horizontal offsets and inward turn angles for the left, right, and centre cameras.
camera_offsets = np.array([-0.75, 0.75, 0.0])
camera_turns = np.radians([18.0, -18.0, 0.0])
n_steps = 10
n_free_steps = 30
frame_duration_ms = 333
free_frame_duration_ms = 40
# Frame count, final hold count, and characteristic landmark depth for the evenly paced three-camera gallery animation.
n_even_frames, n_hold_frames = 108, 18
rotation_lever = 1.5


# --- math -----------------------------------------------------------------------------
def rig_poses(offsets: np.ndarray, turns: np.ndarray) -> Motor:
    """Camera poses (`[...] Motor+`) translate along `x` by `offsets` and rotate in the `xy` plane by `turns`."""
    translations = (mv.xw * (offsets / 2)).exp()                                # [...] Motor+
    rotations = (mv.xy * (turns / 2)).exp()                                     # [...] Motor+
    return translations * rotations                                             # [...] Motor+


def scene_observations(poses: Motor) -> tuple[Point, Camera, Quadric]:
    """Scene landmarks (`[6] Point-`), pinhole cameras (`[n_cameras] Point- <- Point-`), and local sight cones (`[6, n_cameras] Plane+ <- Point-`) observed from `poses`."""
    n_cameras = poses.shape[0]
    landmarks = mv.point(landmark_coordinates)                                  # [6] Point-
    # Place the canonical pinhole at the origin and the sensor plane one unit ahead along y.
    pinhole = mv.point([0, 0])                                                  # [] Point-
    sensor = mv.y - mv.w                                                        # [] Plane+
    cameras = ((pinhole & Point) ^ sensor).broadcast_to((n_cameras,))           # [n_cameras] Point- <- Point-
    images = cameras(poses << landmarks[:, None])                               # [6, n_cameras] Point-
    pixels = images / (mv.w & images)                                           # [6, n_cameras] Point-
    balls = core.sensor_balls(pixels)                                           # [6, n_cameras] Plane+ <- Point-
    return landmarks, cameras, cameras.adjugate()(balls(cameras))


def alternating_trajectory(
    poses: Motor, initial_poses: Motor, damping: float, free_cameras: np.ndarray,
) -> Iterator[tuple[Motor, Quadric, Quadric, Point]]:
    """Yields camera poses, world sight cones, fused splats, and triangulated landmarks at `initial_poses` and after each damped alternating Gauss-Newton step."""
    _, _, cones = scene_observations(poses)
    current_poses = initial_poses                                               # [...] Motor+
    while True:
        centres, splats = core.triangulation(current_poses, cones)
        yield current_poses, current_poses >> cones(current_poses << Point), splats, centres
        current_poses, _, _ = core.alternating_alignment(current_poses, cones, 1, damping, free_cameras)


def joint_trajectory(
    poses: Motor, initial_poses: Motor, damping: float, free_cameras: np.ndarray,
) -> Iterator[tuple[Motor, Quadric, Quadric, Point, Information]]:
    """Yields camera poses, world sight cones, fused splats, triangulated landmarks, and the joint pose information form at `initial_poses` and after each coupled Gauss-Newton step."""
    _, cameras, cones = scene_observations(poses)
    current_poses = initial_poses                                               # [...] Motor+
    while True:
        centres, splats = core.triangulation(current_poses, cones)
        next_poses, _, _, information = core.joint_alignment(cameras, current_poses, cones, 1, damping, free_cameras)
        yield current_poses, current_poses >> cones(current_poses << Point), splats, centres, information
        current_poses = next_poses


def aligned_pair() -> tuple[Motor, Quadric, Quadric, Point]:
    """Aligned poses, world sight cones, fused splats, and landmarks for a two-camera rig after eight alternating steps from a 5% overturn of the right camera."""
    overturns = [1.0, 1.05]
    n_pair_steps, damping, free_cameras = 8, 0.8, np.array([0.0, 1.0])
    poses = rig_poses(camera_offsets[:2], camera_turns[:2])                     # [2] Motor+
    _, _, cones = scene_observations(poses)
    initial_poses = rig_poses(camera_offsets[:2], camera_turns[:2] * overturns)  # [2] Motor+
    aligned_poses, centres, splats = core.alternating_alignment(
        initial_poses, cones, n_pair_steps, damping, free_cameras,
    )
    return aligned_poses, aligned_poses >> cones(aligned_poses << Point), splats, centres


def one_camera() -> Iterator[tuple[Motor, Quadric, Quadric, Point]]:
    """Yields states of the three-camera rig aligning from a 25% overturn of the right camera with the other two cameras anchored."""
    overturns = [1.0, 1.25, 1.0]
    damping, free_cameras = 0.38, np.array([0.0, 1.0, 0.0])
    return alternating_trajectory(
        rig_poses(camera_offsets, camera_turns),
        rig_poses(camera_offsets, camera_turns * overturns),
        damping,
        free_cameras,
    )


def two_cameras() -> Iterator[tuple[Motor, Quadric, Quadric, Point]]:
    """Yields states of the three-camera rig aligning when the right camera is overturned and the middle camera is shifted and turned, with the left camera anchored."""
    offset_errors, overturns, turn_errors = [0.0, 0.0, -0.1], [1.0, 1.25, 1.0], [0.0, 0.0, 0.05]
    damping, free_cameras = 0.35, np.array([0.0, 1.0, 1.0])
    initial_poses = rig_poses(camera_offsets + offset_errors, camera_turns * overturns + turn_errors)  # [3] Motor+
    return alternating_trajectory(rig_poses(camera_offsets, camera_turns), initial_poses, damping, free_cameras)


def three_cameras() -> Iterator[tuple[Motor, Quadric, Quadric, Point]]:
    """Yields states of the three-camera rig aligning when all three cameras start misplaced and move freely."""
    stretches, offset_errors = [0.7, 1.2, 1.0], [0.0, 0.0, -0.45]
    overturns, turn_errors = [1.6, 2.2, 1.0], [0.0, 0.0, 0.35]
    damping, free_cameras = 0.1, np.ones(3)
    initial_poses = rig_poses(
        camera_offsets * stretches + offset_errors, camera_turns * overturns + turn_errors,
    )                                                                           # [3] Motor+
    return alternating_trajectory(rig_poses(camera_offsets, camera_turns), initial_poses, damping, free_cameras)


def paced_trajectory(states: list, n_frames: int, n_hold: int) -> Iterator[tuple[Motor, Quadric, Quadric, Point]]:
    """Yields `n_frames` states interpolated at equal increments of combined pinhole travel and rotation along `states`, followed by `n_hold` copies of the final state."""
    n_states = len(states)
    _, _, cones = scene_observations(rig_poses(camera_offsets, camera_turns))
    poses = stack([state_poses for state_poses, _, _, _ in states])             # [n_states, 3] Motor+
    screws = (poses[1:] * poses[:-1].inverse()).log()                           # [n_states - 1, 3] Twist+
    # Weight camera rotation at landmark depth alongside pinhole travel along each step's screw.
    pinholes = poses >> mv.point([0, 0])                                        # [n_states, 3] Point-
    step_distances = (pinholes[1:] - pinholes[:-1]).antinorm()                  # [n_states - 1, 3] Scalar+
    increments = (step_distances + 2 * rotation_lever * screws.norm()).sum(axis=-1)  # [n_states - 1] Scalar+
    pace = (np.tri(n_states, n_states - 1, -1) * increments).sum(axis=-1)       # [n_states] Scalar+
    for fraction in np.linspace(0.0, 1.0, n_frames):
        target = fraction * pace[-1]                                            # [] Scalar+
        step = min(int((pace <= target).sum()) - 1, n_states - 2)
        weight = (target - pace[step]) / (pace[step + 1] - pace[step])          # [] Scalar+
        eased_poses = (screws[step] * weight).exp() * poses[step]               # [3] Motor+
        centres, splats = core.triangulation(eased_poses, cones)
        yield eased_poses, eased_poses >> cones(eased_poses << Point), splats, centres
    yield from [states[-1]] * n_hold


def joint_one_camera() -> Iterator[tuple[Motor, Quadric, Quadric, Point, Information]]:
    """Yields states of the overturned right camera aligning via coupled camera-and-landmark Gauss-Newton steps."""
    overturns = [1.0, 1.25, 1.0]
    damping, free_cameras = 0.7, np.array([0.0, 1.0, 0.0])
    return joint_trajectory(
        rig_poses(camera_offsets, camera_turns),
        rig_poses(camera_offsets, camera_turns * overturns),
        damping,
        free_cameras,
    )


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from itertools import islice

    from examples.util.animation import save_animation, save_figure
    from examples.estimation.multiview import render

    pair = aligned_pair()
    save_figure(render.reconstruction_figure(*pair), "multiview_aligned_pair")
    one_camera_states = list(islice(one_camera(), n_steps + 1))
    save_animation(
        render.convergence_animation(one_camera_states), "multiview_one_camera", frame_duration_ms, stepped=True,
    )
    two_camera_states = list(islice(two_cameras(), n_steps + 1))
    save_animation(
        render.convergence_animation(two_camera_states), "multiview_two_cameras", frame_duration_ms, stepped=True,
    )
    joint_states = list(islice(joint_one_camera(), n_steps + 1))
    save_animation(
        render.convergence_animation(joint_states), "multiview_jointly", frame_duration_ms, stepped=True,
    )

    # checks
    # The aligned pair places every landmark on both cameras' sight lines, every trajectory cuts
    # sight-cone cost across its ten steps, and the coupled Schur-complement step converges far further
    # than alternating steps from the same overturned camera.
    def trajectory_cost(state: tuple) -> core.Scalar:
        _, state_cones, _, state_centres, *_ = state
        return (state_cones(state_centres[:, None]) & state_centres[:, None]).sum()  # [] Scalar+

    assert trajectory_cost(pair).abs() < 1e-10
    assert trajectory_cost(one_camera_states[-1]) < 0.5 * trajectory_cost(one_camera_states[0])
    assert trajectory_cost(two_camera_states[-1]) < 0.5 * trajectory_cost(two_camera_states[0])
    assert trajectory_cost(joint_states[-1]) < 1e-2 * trajectory_cost(one_camera_states[-1])


def pinned() -> None:
    """Gallery animation tracking the three misplaced cameras aligning at an even pace."""
    from itertools import islice

    from examples.util.animation import pin_animation
    from examples.estimation.multiview import render

    pin_animation(
        render.convergence_animation(
            paced_trajectory(list(islice(three_cameras(), n_free_steps + 1)), n_even_frames, n_hold_frames),
        ),
        "multiview_convergence",
        free_frame_duration_ms,
    )


if __name__ == "__main__":
    main()
