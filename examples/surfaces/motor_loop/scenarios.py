"""A ribbon wound up by twisting, refined as points and as motors."""

from __future__ import annotations

from functools import reduce

import numpy as np

from numga import stack
from examples.surfaces.motor_loop import core

# The ribbon's control mesh: its vertices along and across, its length and width, and its four corners,
# pinned.
N_ALONG, N_ACROSS = 5, 3
LENGTH, WIDTH = 4.0, 1.2
CORNERS = [0, N_ACROSS - 1, N_ALONG * N_ACROSS - N_ACROSS, N_ALONG * N_ACROSS - 1]
# The turn of the far end at the top of the wind-up, the near end held, the twist building towards the far
# end as the square of the distance; the frames of one wind-up and release, and the duration of each.
WOUND = 3 * np.pi
N_FRAMES = 48
DURATION_MS = 60
# How far each frame turns either way about its own axis as the ribbon lies flat, and the seed of the axes
# and of the phases of the turns.
TURNING = np.pi / 3
TURNING_SEED = 2
# Loop steps taken, and the length of the drawn frame axes.
N_LOOP_STEPS = 3
AXIS = 0.3


# --- math -----------------------------------------------------------------------------
def winding() -> tuple[core.Point, core.Point, np.ndarray, np.ndarray]:
    """At every frame of the wind-up, which the animation plays forward and back: the control frames, and the
    surfaces refined as points and as motors."""
    along, across = grid()
    wound = WOUND * (1 - np.cos(np.linspace(0, np.pi, N_FRAMES // 2 + 1))) / 2   # [frames]
    held = ((along + LENGTH / 2) / LENGTH) ** 2                                # [vertices]
    # Frames on the ribbon down the z axis: slid along it and out across it along x, then turned about the axis.
    motors = ((core.mv.xy * (-wound[:, None] * held / 2)).exp()
              * (core.mv.z * core.mv.w * (along / 2)).exp() * (core.mv.x * core.mv.w * (across / 2)).exp())
    return refined(motors.field())                                             # [frames] Motor[vertices]


def turning() -> tuple[core.Point, core.Point, np.ndarray, np.ndarray]:
    """At every frame, the ribbon held flat with every frame turning back and forth about its own axis
    through its origin, each with its own phase: the control frames, and the surfaces refined as points and
    as motors."""
    along, across = grid()
    rng = np.random.default_rng(TURNING_SEED)
    axes = core.mv(core.Line.derive.nondegenerate, rng.normal(size=(len(along), 3))).normalized()   # [vertices] Line
    phases = rng.uniform(0, 2 * np.pi, len(along))
    angles = TURNING * np.sin(np.linspace(0, 2 * np.pi, N_FRAMES, endpoint=False)[:, None] + phases)   # [frames, vertices]
    flat = (core.mv.z * core.mv.w * (along / 2)).exp() * (core.mv.x * core.mv.w * (across / 2)).exp()
    return refined((flat * (axes * (-angles / 2)).exp()).field())              # [frames] Motor[vertices]


def refined(motors: core.Motor) -> tuple[core.Point, core.Point, np.ndarray, np.ndarray]:
    """The control frames of motors at the vertices of the ribbon, each its origin and the ends of its x, y
    and z axes, and the surface refined two ways: as the points the motors carry, through the steps composed
    into one map, and as the motors themselves, normalized after every step; with the faces of the control
    mesh and of the surfaces."""
    faces = core.grid_faces(N_ALONG, N_ACROSS)
    pinned = np.isin(np.arange(N_ALONG * N_ACROSS), CORNERS)
    markers = stack([core.mv.w.dual(), core.mv.point([AXIS, 0.0, 0.0]), core.mv.point([0.0, AXIS, 0.0]),
                     core.mv.point([0.0, 0.0, AXIS])])                         # [axes + 1] Point
    origin = markers[0]                                                        # [] Point
    steps, meshes, refined_motors = zip(*core.refinements(core.Mesh(N_ALONG * N_ACROSS, faces), motors, pinned, N_LOOP_STEPS))
    points = reduce(lambda total, step: step * total, steps) * (motors >> origin)   # [frames] Point[refined]
    surfaces = stack([points, refined_motors[-1] >> origin], axis=1)           # [frames, schemes] Point[refined]
    return motors[:, None] >> markers, surfaces, faces, meshes[-1].faces       # [frames, axes + 1] Point[vertices]


# --- plumbing -------------------------------------------------------------------------
def grid() -> tuple[np.ndarray, np.ndarray]:
    """How far along and across the ribbon each vertex of its control mesh lies, row by row."""
    along, across = np.meshgrid(np.linspace(-LENGTH / 2, LENGTH / 2, N_ALONG),
                                np.linspace(-WIDTH / 2, WIDTH / 2, N_ACROSS), indexing="ij")
    return along.reshape(-1), across.reshape(-1)


def main() -> None:
    from examples.util.animation import save_animation
    from examples.surfaces.motor_loop import render

    wound = winding()
    turned = turning()
    save_animation(render.animate(*wound), "motor_loop_ribbon", DURATION_MS, forward_then_back=True)
    save_animation(render.animate(*turned), "motor_loop_turning", DURATION_MS)

    # checks
    # Untwisted the two agree; wound up, the points wring the ribbon towards its axis and the motors do not.
    # How far each surface reaches from the z axis on average, at rest and wound up: a point joined with the line
    # through the origin and z is a plane as large as their distance.
    centre_line = core.mv.w.dual() & core.mv.point([0.0, 0.0, 1.0])           # [] Line
    reach = (centre_line & wound[1][[0, N_FRAMES // 2]].batch()).norm().mean(axis=-1)   # [cases, schemes] Scalar
    (points_rest, motors_rest), (points_peak, motors_peak) = reach.to_array()
    assert abs(points_rest - motors_rest) < 1e-12
    assert points_peak < 0.7 * points_rest and motors_peak >= motors_rest
    # Turning the frames about their origins leaves the points alone and bends the motors: two points joined are
    # a line as long as their distance.
    points, motors = turned[1][:, 0].batch(), turned[1][:, 1].batch()          # [frames, refined] Point
    assert (points & points[:1]).norm().to_array().max() < 1e-12
    assert (motors & points).norm().to_array().max() > 0.05
    # A turn common to every frame changes nothing.
    along, across = grid()
    flat = (core.mv.z * core.mv.w * (along / 2)).exp() * (core.mv.x * core.mv.w * (across / 2)).exp()   # [vertices] Motor
    cases = refined(stack([flat, flat * (core.mv.yz * (-0.7 / 2)).exp()]).field())[1][:, 1].batch()   # [cases, refined] Point
    assert (cases[0] & cases[1]).norm().to_array().max() < 1e-12


if __name__ == "__main__":
    main()
