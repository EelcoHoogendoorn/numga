"""Scenes for the tricycle: the same drive, its steering held to the left and swung about that, in the
plane and in space, by the same code; only the algebra differs."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga.algebras import PGA2D, PGA3D
from examples import instantiate

WHEELBASE = 1.2
TRACK = 0.8
# The distance the front wheel rolls per step, over the wheelbase, and the number of steps.
STEP = 0.04
N_STEPS = 600
# The steering angle held, how far it swings about that, and how many times it swings over the drive.
BIAS = 0.35
SWING = 0.3
N_SWINGS = 3
# The frames of each animation, and the duration of each in milliseconds.
N_FRAMES = 150
FRAME_DURATION = 50
# The same module, instantiated for the plane and for space.
SCENES = {name: instantiate("examples.mechanics.tricycle.core", ga) for name, ga in (("tricycle_plane", PGA2D), ("tricycle_space", PGA3D))}


# --- math -----------------------------------------------------------------------------
def run(core):
    """The drive in the core's algebra: per step the wheels' contacts with the ground and the points ahead
    of them, the turning centre, and the middle of the rear axle, all in the ground's frame."""
    steering = BIAS + SWING * np.sin(2 * np.pi * N_SWINGS * np.arange(N_STEPS) / N_STEPS)   # [steps]
    rear, front, steer = core.axles(WHEELBASE, steering)                        # [], [steps], [steps]
    poses = core.drive(rear, front, STEP)                                       # [steps] Motor
    contacts, forward = core.wheels(WHEELBASE, TRACK, steer)                    # [wheels], [steps, wheels] Point
    contacts = poses[:, None] >> contacts                                       # [steps, wheels] Point
    forward = poses[:, None] >> forward                                         # [steps, wheels] Point
    centres = poses >> (rear ^ front)                                           # [steps] Meet
    middle = poses >> core.point([0.0, 0.0])                                    # [steps] Point
    return contacts, forward, centres, middle


def moments(contacts, forward, centres, middle, n_frames: int) -> Iterator[tuple]:
    """Per frame, the wheels, the points ahead of them and the turning centre at that moment, and the trail
    of the middle of the rear axle so far, from the first step on, so that every trail is a line."""
    for step in np.linspace(1, N_STEPS - 1, n_frames).astype(int):
        yield contacts[step], forward[step], centres[step], middle[:step + 1]


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation
    from examples.mechanics.tricycle import render

    drives = {name: run(core) for name, core in SCENES.items()}
    for (name, drive), draw in zip(drives.items(), (render.plane, render.space)):
        save_animation(render.animate(draw, moments(*drive, N_FRAMES)), name, FRAME_DURATION)

    # checks
    # No wheel slides sideways: each wheel's contact, the point ahead of it and its next contact join to
    # nothing, but for the step's curvature, in the plane and in space alike; a step's chord leaves its
    # tangent by its length squared over the turning radius. The front wheel rolls the same distance every
    # step. With the front wheel straight the axles meet at infinity. Both drives leave the same track.
    for core, (contacts, forward, _, middle) in zip(SCENES.values(), drives.values()):
        rolling = (contacts[:-1].normalized() & forward[:-1].normalized() & contacts[1:].normalized()).norm().to_array()
        assert rolling.max() < 0.1 * WHEELBASE * (STEP * WHEELBASE) ** 2
        front_steps = np.linalg.norm(np.diff(render.ground(contacts[:, 2]), axis=0), axis=-1)
        np.testing.assert_allclose(front_steps, STEP * WHEELBASE, rtol=1e-3)
        rear, front, _ = core.axles(WHEELBASE, np.zeros(1))
        np.testing.assert_allclose((core.mv.w ^ (rear ^ front)).kernel, 0.0, atol=1e-12)
    plane_track, space_track = (render.ground(drive[3]) for drive in drives.values())
    np.testing.assert_allclose(plane_track, space_track, atol=1e-9)


if __name__ == "__main__":
    main()
