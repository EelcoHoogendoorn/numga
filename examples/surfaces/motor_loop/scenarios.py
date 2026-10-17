"""A ribbon wound up by twisting, refined as points and as motors."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.surfaces.motor_loop import core

# The ribbon's control mesh: its vertices along and across, its length and width, and its four corners,
# pinned.
ALONG_COUNT, ACROSS_COUNT = 5, 3
LENGTH, WIDTH = 4.0, 1.2
CORNERS = [0, ACROSS_COUNT - 1, ALONG_COUNT * ACROSS_COUNT - ACROSS_COUNT, ALONG_COUNT * ACROSS_COUNT - 1]
# The turn of the far end at the top of the wind-up, the near end held, the twist building towards the far
# end as the square of the distance; the frames of one wind-up and release, and the duration of each.
WOUND = 3 * np.pi
FRAMES = 48
DURATION_MS = 60
# How far each frame turns either way about its own axis as the ribbon lies flat, and the seed of the axes
# and of the phases of the turns.
TURNING = np.pi / 3
TURNING_SEED = 2
# Loop steps taken, and the length of the drawn frame axes.
DEPTH = 3
AXIS = 0.3


# --- math -----------------------------------------------------------------------------
def winding() -> tuple[core.Point, core.Point, np.ndarray, np.ndarray]:
    """At every frame of the wind-up and release: the control frames, and the surfaces refined as points and
    as motors."""
    along, across = grid()
    wound = WOUND * (1 - np.cos(np.linspace(0, 2 * np.pi, FRAMES, endpoint=False))) / 2   # [frames]
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
    angles = TURNING * np.sin(np.linspace(0, 2 * np.pi, FRAMES, endpoint=False)[:, None] + phases)   # [frames, vertices]
    return refined((core.ribbon(along, across, 0.0) * core.turn(axes, angles)).field())   # [frames] Motor[vertices]


def refined(motors: core.Motor) -> tuple[core.Point, core.Point, np.ndarray, np.ndarray]:
    """The control frames of motors at the vertices of the ribbon, each its origin and the ends of its x, y
    and z axes, and the surface refined two ways: as the points the motors carry, through the steps composed
    into one map, and as the motors themselves, normalized after every step; with the faces of the control
    mesh and of the surfaces."""
    faces = core.grid_faces(ALONG_COUNT, ACROSS_COUNT)
    pinned = np.isin(np.arange(ALONG_COUNT * ACROSS_COUNT), CORNERS)
    markers = stack([core.mv.w.dual(), core.mv.point([AXIS, 0.0, 0.0]), core.mv.point([0.0, AXIS, 0.0]),
                     core.mv.point([0.0, 0.0, AXIS])])                         # [axes + 1] Point
    origin = markers[0]                                                        # [] Point
    steps, meshes = zip(*core.refinements(core.Mesh(motors, faces), pinned, DEPTH))
    points = core.composed(steps) * (motors >> origin)                         # [frames] Point[refined]
    surfaces = stack([points, meshes[-1].vertices >> origin], axis=1)          # [frames, schemes] Point[refined]
    return motors[:, None] >> markers, surfaces, faces, meshes[-1].faces       # [frames, axes + 1] Point[vertices]


# --- plumbing -------------------------------------------------------------------------
def grid() -> tuple[np.ndarray, np.ndarray]:
    """How far along and across the ribbon each vertex of its control mesh lies, row by row."""
    along, across = np.meshgrid(np.linspace(-LENGTH / 2, LENGTH / 2, ALONG_COUNT),
                                np.linspace(-WIDTH / 2, WIDTH / 2, ACROSS_COUNT), indexing="ij")
    return along.reshape(-1), across.reshape(-1)


def main() -> None:
    from examples.animation import save_animation
    from examples.surfaces.motor_loop import render

    save_animation(render.animate(*winding()), "motor_loop_ribbon", DURATION_MS)
    save_animation(render.animate(*turning()), "motor_loop_turning", DURATION_MS)


if __name__ == "__main__":
    main()
