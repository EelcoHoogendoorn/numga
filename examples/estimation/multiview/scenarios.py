"""Scenes of multi-view reconstruction and camera alignment in the plane.

Three planar cameras, one on the left and one on the right turned inwards and one between them looking
ahead, view six landmarks. In each scene some cameras are misplaced and walk back into place: by the
alternating steps of `align`, one state per step for an animation, or by the joint steps of
`align_jointly`, which also give the information on every pose. All scenes return geometry for `render`.
"""

from __future__ import annotations

import numpy as np

from numga import stack
from numga.algebras import PGA2D

from examples import instantiate

core = instantiate("examples.estimation.multiview.core", PGA2D)
Motor, Point, Quadric = core.Motor, core.Point, core.Quadric
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
FREE_FRAME_DURATION = 120
# The three cameras' alignment shown at an even pace: its frames, the frames the aligned rig is held, and the depth
# at which a camera's turn counts as much as its move, about where the landmarks are.
N_EVEN_FRAMES, N_HOLD = 36, 6
LEVER = 1.5


# --- math -----------------------------------------------------------------------------
def rig(offsets: np.ndarray, turns: np.ndarray) -> Motor:
    """Each camera moved along x, then turned."""
    return (mv.xw * (offsets / 2)).exp() * (mv.xy * (turns / 2)).exp()


def observe(poses: Motor) -> tuple[Point, core.Camera, Quadric]:
    """The landmarks, the pinhole camera at each pose, and the sight cones of the pixels where each camera
    sees each landmark."""
    landmarks = mv.point(LANDMARKS)                                             # [landmarks] Point
    # The pinhole at the origin and the sensor, the line one unit in front of it.
    pinhole = mv.point([0.0, 0.0])                                              # [] Point
    sensor = mv.y - mv.w                                                        # [] Plane
    cameras = ((pinhole & Point) ^ sensor).broadcast_to(poses.shape)            # [cameras] Point <- Point
    images = cameras(poses << landmarks[:, None])                               # [landmarks, cameras] Point
    pixels = images / (core.w & images)                                         # [landmarks, cameras] Point
    return landmarks, cameras, core.sight_cones(cameras, core.sensor_disc(pixels))


def alignment(poses: Motor, guess: Motor, damping: float, moving: np.ndarray):
    """The rig, its sight cones in the world, the splats and the landmarks, from the guess and after each
    alternating step, without end."""
    _, _, cones = observe(poses)
    while True:
        centres, splats = core.triangulate(guess, cones)
        yield guess, guess >> cones(guess << Point), splats, centres
        guess, _, _ = core.align(guess, cones, 1, damping, moving)


def joint_alignment(poses: Motor, guess: Motor, damping: float, moving: np.ndarray):
    """As `alignment`, by joint steps of cameras and landmarks, each state with the information on the
    poses from the step taken there."""
    _, cameras, cones = observe(poses)
    while True:
        centres, splats = core.triangulate(guess, cones)
        aligned, _, _, information = core.align_jointly(cameras, guess, cones, 1, damping, moving)
        yield guess, guess >> cones(guess << Point), splats, centres, information
        guess = aligned


def aligned_pair():
    """Two cameras, the right one turned 5% too far, aligned by eight alternating steps with the left one
    held still: the rig, its sight cones in the world, the splats and the landmarks."""
    # How far each camera is turned, against its true turn; the steps, the share of each step taken, and
    # the camera that moves.
    overturns = [1.0, 1.05]
    n_steps, damping, moving = 8, 0.8, np.array([0.0, 1.0])
    poses = rig(OFFSETS[:2], TURNS[:2])                                         # [cameras] Motor
    _, _, cones = observe(poses)
    aligned, centres, splats = core.align(rig(OFFSETS[:2], TURNS[:2] * overturns), cones, n_steps, damping, moving)
    return aligned, aligned >> cones(aligned << Point), splats, centres


def one_camera():
    """The right camera turned 25% too far, the other two held still."""
    overturns = [1.0, 1.25, 1.0]
    damping, moving = 0.38, np.array([0.0, 1.0, 0.0])
    return alignment(rig(OFFSETS, TURNS), rig(OFFSETS, TURNS * overturns), damping, moving)


def two_cameras():
    """The right camera turned 25% too far and the middle one moved and turned, the left one held still."""
    # Each camera's misplacement along x, and its overturn and added turn in radians.
    shifts, overturns, added_turns = [0.0, 0.0, -0.1], [1.0, 1.25, 1.0], [0.0, 0.0, 0.05]
    damping, moving = 0.35, np.array([0.0, 1.0, 1.0])
    guess = rig(OFFSETS + shifts, TURNS * overturns + added_turns)              # [cameras] Motor
    return alignment(rig(OFFSETS, TURNS), guess, damping, moving)


def three_cameras():
    """Every camera misplaced, and every camera moving with none held still."""
    # Each camera's offset scaled and shifted along x, and its turn scaled and added to, in radians.
    stretches, shifts = [0.7, 1.2, 1.0], [0.0, 0.0, -0.45]
    overturns, added_turns = [1.6, 2.2, 1.0], [0.0, 0.0, 0.35]
    damping, moving = 0.1, np.ones(3)
    guess = rig(OFFSETS * stretches + shifts, TURNS * overturns + added_turns)  # [cameras] Motor
    return alignment(rig(OFFSETS, TURNS), guess, damping, moving)


def evenly(states: list, n_frames: int, n_hold: int):
    """A converging alignment `[steps]` at an even pace: frames at equal shares of how far the cameras move and turn
    in all, each rig eased along the screw from the step before to the step after, with its cones, splats and
    landmarks; then the aligned rig held."""
    _, _, cones = observe(rig(OFFSETS, TURNS))
    poses = stack([guess for guess, _, _, _ in states])                   # [steps, cameras] Motor
    screws = (poses[1:] * poses[:-1].inverse()).log()                           # [steps - 1, cameras] Twist
    # How far each step moves the pinholes, and turns the cameras, counted at the depth of the landmarks.
    pinholes = poses >> mv.point([0.0, 0.0])                                    # [steps, cameras] Point
    moved = (pinholes[1:] - pinholes[:-1]).dual().norm()                        # [steps - 1, cameras] Scalar
    pace = np.cumsum([0.0] + list((moved + 2 * LEVER * screws.norm()).sum(axis=-1).to_array()))   # [steps]
    for share in np.linspace(0.0, pace[-1], n_frames):
        step = min(np.searchsorted(pace, share, side="right") - 1, len(screws) - 1)
        eased = ((screws[step] * ((share - pace[step]) / (pace[step + 1] - pace[step]))).exp() * poses[step])   # [cameras] Motor
        centres, splats = core.triangulate(eased, cones)
        yield eased, eased >> cones(eased << Point), splats, centres
    yield from [states[-1]] * n_hold


def jointly():
    """The misplaced right camera of `one_camera`, aligned by joint steps."""
    overturns = [1.0, 1.25, 1.0]
    damping, moving = 0.7, np.array([0.0, 1.0, 0.0])
    return joint_alignment(rig(OFFSETS, TURNS), rig(OFFSETS, TURNS * overturns), damping, moving)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from itertools import islice

    from examples.animation import save_animation, save_figure
    from examples.estimation.multiview import render

    pair = aligned_pair()
    save_figure(render.draw_reconstruction(*pair), "multiview_aligned_pair")
    save_animation(render.animate_convergence(islice(one_camera(), N_STEPS + 1)), "multiview_one_camera", FRAME_DURATION)
    save_animation(render.animate_convergence(islice(two_cameras(), N_STEPS + 1)), "multiview_two_cameras", FRAME_DURATION)
    save_animation(render.animate_convergence(evenly(list(islice(three_cameras(), N_FREE_STEPS + 1)), N_EVEN_FRAMES, N_HOLD)),
                   "multiview_three_cameras", FREE_FRAME_DURATION)
    save_animation(render.animate_convergence_with_covariance(islice(jointly(), N_STEPS + 1)), "multiview_jointly",
                   FRAME_DURATION)

    # checks
    # The aligned pair explains every pixel: every landmark's cones vanish at it. Two cameras with one held
    # still leave the scene's scale free, so the landmarks are found only up to that scale.
    aligned, _, _, centres = pair
    _, _, cones = observe(rig(OFFSETS[:2], TURNS[:2]))
    seen = aligned << centres[:, None]                                          # [landmarks, cameras] Point
    np.testing.assert_allclose((cones(seen) & seen).sum().to_array(), 0.0, atol=1e-12)


if __name__ == "__main__":
    main()
