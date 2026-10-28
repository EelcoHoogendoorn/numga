"""Planar multi-view reconstruction scenes place three convergent cameras in front of six landmarks and recover misplaced camera poses by alternating or joint Gauss-Newton alignment.

Three planar cameras, one on the left and one on the right turned inwards and one between them looking
straight ahead, view six landmarks. In each scene some cameras are misplaced and walk back into place, either by
the alternating steps of `align`, yielding one state per step for an animation, or by the joint steps of
`align_jointly`, which also return the pose information form. All scenes return geometric extensors for `render`.
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

# The landmarks, in front of the cameras.
LANDMARKS = [[0.15, 0.85], [-0.43, 1.15], [0.50, 1.50], [0.03, 1.85], [0.65, 2.20], [-0.60, 2.55]]
# Each camera's offset along x and how far it is turned inwards: left, right, and in the middle.
OFFSETS = np.array([-0.75, 0.75, 0.0])
TURNS = np.radians([18.0, -18.0, 0.0])
# The steps of each animation, and the durations of their frames in milliseconds.
N_STEPS = 10
N_FREE_STEPS = 30
FRAME_DURATION = 333
FREE_FRAME_DURATION = 40
# The three cameras' alignment shown at an even pace: its frames, the frames the aligned rig is held, and the depth
# at which a camera's turn counts as much as its move, about where the landmarks are. This animation is the README's
# (plots/multiview_convergence.gif): its misplacements in `three_cameras`, its damping and these frames are tuned as a
# picture, a rig that visibly starts far off and settles near the end, not as a demonstration of the alignment's
# speed; keep them, and check the picture after any change.
N_EVEN_FRAMES, N_HOLD = 108, 18
LEVER = 1.5


# --- math -----------------------------------------------------------------------------
def rig(offsets: np.ndarray, turns: np.ndarray) -> Motor:
    """Camera poses placed by moving each along x and then turning it in the xy plane."""
    shifts = (mv.xw * (offsets / 2)).exp()                                      # [cameras] Motor+
    rotations = (mv.xy * (turns / 2)).exp()                                     # [cameras] Motor+
    return shifts * rotations                                                   # [cameras] Motor+


def observe(poses: Motor) -> tuple[Point, Camera, Quadric]:
    """The landmarks, the pinhole cameras at `poses`, and the sight cones pulled back from their pixel balls."""
    landmarks = mv.point(LANDMARKS)                                             # [landmarks] Point-
    # The pinhole at the origin and the sensor, the plane one unit in front of it.
    pinhole = mv.point([0, 0])                                                  # [] Point-
    sensor = mv.y - mv.w                                                        # [] Plane+
    cameras = ((pinhole & Point) ^ sensor).broadcast_to(poses.shape)            # [cameras] Point- <- Point-
    images = cameras(poses << landmarks[:, None])                               # [landmarks, cameras] Point-
    pixels = images / (core.w & images)                                         # [landmarks, cameras] Point-
    balls = core.sensor_ball(pixels)                                            # [landmarks, cameras] Plane+ <- Point-
    return landmarks, cameras, cameras.adjugate()(balls(cameras))


def alignment(poses: Motor, guess: Motor, damping: float, free_cameras: np.ndarray) -> Iterator[tuple[Motor, Quadric, Quadric, Point]]:
    """The camera poses, world sight cones, fused splats, and triangulated landmarks at the initial guess and after each alternating Gauss-Newton step."""
    _, _, cones = observe(poses)
    while True:
        centres, splats = core.triangulate(guess, cones)
        yield guess, guess >> cones(guess << Point), splats, centres
        guess, _, _ = core.align(guess, cones, 1, damping, free_cameras)


def joint_alignment(
    poses: Motor, guess: Motor, damping: float, free_cameras: np.ndarray,
) -> Iterator[tuple[Motor, Quadric, Quadric, Point, Information]]:
    """The camera poses, world sight cones, fused splats, triangulated landmarks, and pose information form at the initial guess and after each joint Gauss-Newton step."""
    _, cameras, cones = observe(poses)
    while True:
        centres, splats = core.triangulate(guess, cones)
        aligned_poses, _, _, information = core.align_jointly(cameras, guess, cones, 1, damping, free_cameras)
        yield guess, guess >> cones(guess << Point), splats, centres, information
        guess = aligned_poses


def aligned_pair() -> tuple[Motor, Quadric, Quadric, Point]:
    """Two cameras aligned over eight alternating steps from the right camera turned 5% too far with the left camera held still."""
    # How far each camera is turned, against its true turn; the steps, the share of each step taken, and
    # the camera that moves.
    overturns = [1.0, 1.05]
    n_steps, damping, free_cameras = 8, 0.8, np.array([0.0, 1.0])
    poses = rig(OFFSETS[:2], TURNS[:2])                                         # [cameras] Motor+
    _, _, cones = observe(poses)
    aligned_poses, centres, splats = core.align(rig(OFFSETS[:2], TURNS[:2] * overturns), cones, n_steps, damping, free_cameras)
    return aligned_poses, aligned_poses >> cones(aligned_poses << Point), splats, centres


def one_camera() -> Iterator[tuple[Motor, Quadric, Quadric, Point]]:
    """The three-camera rig aligning from the right camera turned 25% too far with the other two held still."""
    overturns = [1.0, 1.25, 1.0]
    damping, free_cameras = 0.38, np.array([0.0, 1.0, 0.0])
    return alignment(rig(OFFSETS, TURNS), rig(OFFSETS, TURNS * overturns), damping, free_cameras)


def two_cameras() -> Iterator[tuple[Motor, Quadric, Quadric, Point]]:
    """The three-camera rig aligning from the right camera turned 25% too far and the middle camera shifted and turned, with the left camera held still."""
    # Each camera's misplacement along x, and its overturn and added turn in radians.
    shifts, overturns, added_turns = [0.0, 0.0, -0.1], [1.0, 1.25, 1.0], [0.0, 0.0, 0.05]
    damping, free_cameras = 0.35, np.array([0.0, 1.0, 1.0])
    guess = rig(OFFSETS + shifts, TURNS * overturns + added_turns)              # [cameras] Motor+
    return alignment(rig(OFFSETS, TURNS), guess, damping, free_cameras)


# README picture: this makes plots/multiview_convergence.gif, tuned as a visual for the README. Don't change it; to
# experiment, copy it.
def three_cameras() -> Iterator[tuple[Motor, Quadric, Quadric, Point]]:
    """The three-camera rig aligning when every camera is misplaced and all three cameras move freely."""
    # Each camera's offset scaled and shifted along x, and its turn scaled and added to, in radians.
    stretches, shifts = [0.7, 1.2, 1.0], [0.0, 0.0, -0.45]
    overturns, added_turns = [1.6, 2.2, 1.0], [0.0, 0.0, 0.35]
    damping, free_cameras = 0.1, np.ones(3)
    guess = rig(OFFSETS * stretches + shifts, TURNS * overturns + added_turns)  # [cameras] Motor+
    return alignment(rig(OFFSETS, TURNS), guess, damping, free_cameras)


# README picture: this makes plots/multiview_convergence.gif, tuned as a visual for the README. Don't change it; to
# experiment, copy it.
def evenly(states: list, n_frames: int, n_hold: int) -> Iterator[tuple[Motor, Quadric, Quadric, Point]]:
    """A converging alignment sequence resampled at equal increments of total pinhole travel and camera rotation along the interpolating screw, then held at the final rig."""
    _, _, cones = observe(rig(OFFSETS, TURNS))
    poses = stack([guess for guess, _, _, _ in states])                         # [steps, cameras] Motor+
    screws = (poses[1:] * poses[:-1].inverse()).log()                           # [steps - 1, cameras] Twist+
    # How far each step moves the pinholes, and turns the cameras, counted at the depth of the landmarks.
    pinholes = poses >> mv.point([0, 0])                                        # [steps, cameras] Point-
    step_distances = (pinholes[1:] - pinholes[:-1]).dual().norm()               # [steps - 1, cameras] Scalar+
    pace = np.cumsum([0.0] + list((step_distances + 2 * LEVER * screws.norm()).sum(axis=-1).to_array()))   # [steps]
    for share in np.linspace(0.0, pace[-1], n_frames):
        step = min(np.searchsorted(pace, share, side="right") - 1, len(screws) - 1)
        eased_poses = (screws[step] * ((share - pace[step]) / (pace[step + 1] - pace[step]))).exp() * poses[step]   # [cameras] Motor+
        centres, splats = core.triangulate(eased_poses, cones)
        yield eased_poses, eased_poses >> cones(eased_poses << Point), splats, centres
    yield from [states[-1]] * n_hold


def jointly() -> Iterator[tuple[Motor, Quadric, Quadric, Point, Information]]:
    """The misplaced right camera of `one_camera` aligning by joint camera-and-landmark Gauss-Newton steps."""
    overturns = [1.0, 1.25, 1.0]
    damping, free_cameras = 0.7, np.array([0.0, 1.0, 0.0])
    return joint_alignment(rig(OFFSETS, TURNS), rig(OFFSETS, TURNS * overturns), damping, free_cameras)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from itertools import islice

    from examples.util.animation import save_animation, save_figure
    from examples.estimation.multiview import render

    pair = aligned_pair()
    save_figure(render.draw_reconstruction(*pair), "multiview_aligned_pair")
    save_animation(render.animate_convergence(islice(one_camera(), N_STEPS + 1)), "multiview_one_camera", FRAME_DURATION,
                   stepped=True)
    save_animation(render.animate_convergence(islice(two_cameras(), N_STEPS + 1)), "multiview_two_cameras", FRAME_DURATION,
                   stepped=True)
    # README picture: this makes plots/multiview_convergence.gif, tuned as a visual for the README. Don't change it; to
    # experiment, copy it.
    save_animation(render.animate_convergence(evenly(list(islice(three_cameras(), N_FREE_STEPS + 1)), N_EVEN_FRAMES, N_HOLD)),
                   "multiview_three_cameras", FREE_FRAME_DURATION)
    save_animation(render.animate_convergence(state[:4] for state in islice(jointly(), N_STEPS + 1)), "multiview_jointly",
                   FRAME_DURATION, stepped=True)

    # checks
    # The aligned pair explains every pixel: every landmark's cones vanish at it. Two cameras with one held
    # still leave the scene's scale free, so the landmarks are found only up to that scale.
    aligned_poses, _, _, centres = pair
    _, _, cones = observe(rig(OFFSETS[:2], TURNS[:2]))
    local_centres = aligned_poses << centres[:, None]                           # [landmarks, cameras] Point-
    np.testing.assert_allclose((cones(local_centres) & local_centres).sum().to_array(), 0.0, atol=1e-12)


if __name__ == "__main__":
    main()
