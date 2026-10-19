"""A ribbon wound up by twisting, refined as points and as motors."""

from __future__ import annotations

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
    """At every frame of the wind-up and release: the control frames, and the surfaces refined as points and
    as motors."""
    along, across = grid()
    wound = WOUND * (1 - np.cos(np.linspace(0, 2 * np.pi, N_FRAMES, endpoint=False))) / 2   # [frames]
    held = ((along + LENGTH / 2) / LENGTH) ** 2                                # [vertices]
    return refined(core.ribbon(along, across, wound[:, None] * held).field())   # [frames] Motor[vertices]


def turning() -> tuple[core.Point, core.Point, np.ndarray, np.ndarray]:
    """At every frame, the ribbon held flat with every frame turning back and forth about its own axis
    through its origin, each with its own phase: the control frames, and the surfaces refined as points and
    as motors."""
    along, across = grid()
    rng = np.random.default_rng(TURNING_SEED)
    axes = core.mv(core.Line.derive.nondegenerate, rng.normal(size=(len(along), 3))).normalized()   # [vertices] Line
    phases = rng.uniform(0, 2 * np.pi, len(along))
    angles = TURNING * np.sin(np.linspace(0, 2 * np.pi, N_FRAMES, endpoint=False)[:, None] + phases)   # [frames, vertices]
    return refined((core.ribbon(along, across, 0.0) * core.turn(axes, angles)).field())   # [frames] Motor[vertices]


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
    steps, meshes = zip(*core.refinements(core.Mesh(motors, faces), pinned, N_LOOP_STEPS))
    points = core.composed(steps) * (motors >> origin)                         # [frames] Point[refined]
    surfaces = stack([points, meshes[-1].vertices >> origin], axis=1)          # [frames, schemes] Point[refined]
    return motors[:, None] >> markers, surfaces, faces, meshes[-1].faces       # [frames, axes + 1] Point[vertices]


# --- plumbing -------------------------------------------------------------------------
def grid() -> tuple[np.ndarray, np.ndarray]:
    """How far along and across the ribbon each vertex of its control mesh lies, row by row."""
    along, across = np.meshgrid(np.linspace(-LENGTH / 2, LENGTH / 2, N_ALONG),
                                np.linspace(-WIDTH / 2, WIDTH / 2, N_ACROSS), indexing="ij")
    return along.reshape(-1), across.reshape(-1)


def main() -> None:
    from examples.animation import save_animation
    from examples.surfaces.motor_loop import render

    wound = winding()
    turned = turning()
    save_animation(render.animate(*wound), "motor_loop_ribbon", DURATION_MS)
    save_animation(render.animate(*turned), "motor_loop_turning", DURATION_MS)

    # checks
    # Untwisted the two agree; wound up, the points wring the ribbon towards its axis and the motors do not.
    (points_rest, motors_rest), (points_peak, motors_peak) = render.reach(wound[1][[0, N_FRAMES // 2]])
    assert abs(points_rest - motors_rest) < 1e-12
    assert points_peak < 0.7 * points_rest and motors_peak >= motors_rest
    # Turning the frames about their origins leaves the points alone and bends the motors.
    points, motors = turned[1][:, 0], turned[1][:, 1]                          # [frames] Point[refined]
    assert render.separation(points, points[:1]) < 1e-12
    assert render.separation(motors, points) > 0.05
    # A turn common to every frame changes nothing.
    flat = core.ribbon(*grid(), 0.0)                                           # [vertices] Motor
    cases = refined(stack([flat, flat * core.turn(core.mv.yz, 0.7)]).field())[1][:, 1]   # [cases] Point[refined]
    assert render.separation(cases[0], cases[1]) < 1e-12


if __name__ == "__main__":
    main()
