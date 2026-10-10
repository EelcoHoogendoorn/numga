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
STEPS = 600
# The steering angle held, how far it swings about that, and how many times it swings over the drive.
BIAS = 0.35
SWING = 0.3
SWINGS = 3
# The same module, instantiated for the plane and for space.
SCENES = {name: instantiate("examples.mechanics.tricycle.core", ga) for name, ga in (("tricycle_plane", PGA2D), ("tricycle_space", PGA3D))}


# --- math -----------------------------------------------------------------------------
def run(core):
    """The drive in the core's algebra: per step the wheels' contacts with the ground and the points ahead
    of them, the turning centre, and the middle of the rear axle, all in the ground's frame."""
    mv = core.mv
    steering = BIAS + SWING * np.sin(2 * np.pi * SWINGS * np.arange(STEPS) / STEPS)   # [steps]
    rear, front, steer = core.axles(WHEELBASE, steering)                    # [], [steps], [steps]
    poses = core.drive(rear, front, STEP)                                       # [steps] Motor
    contacts, forward = core.wheels(WHEELBASE, TRACK, steer)                # [wheels], [steps, wheels] Point
    contacts = poses[:, None] >> contacts                                       # [steps, wheels] Point
    forward = poses[:, None] >> forward                                         # [steps, wheels] Point
    centres = poses >> (rear ^ front)                                           # [steps] turning centre
    middle = poses >> core.point([0.0, 0.0])                                # [steps] Point

    # --- checks
    # No wheel slides sideways: each wheel's contact, the point ahead of it and its next contact join to
    # nothing, but for the step's curvature, in the plane and in space alike. The front wheel rolls the
    # same distance every step. With the front wheel straight the axles meet at infinity.
    rolling = (contacts[:-1].normalized() & forward[:-1].normalized() & contacts[1:].normalized()).norm().to_array()
    # A step's chord leaves its tangent by its length squared over the turning radius, under the step itself.
    assert rolling.max() < 0.1 * WHEELBASE * (STEP * WHEELBASE) ** 2
    front_steps = np.linalg.norm(np.diff(coordinates(mv, contacts[:, 2]), axis=0), axis=-1)
    np.testing.assert_allclose(front_steps, STEP * WHEELBASE, rtol=1e-3)
    straight = core.axles(WHEELBASE, np.zeros(1))
    np.testing.assert_allclose((mv.w ^ (straight[0] ^ straight[1])).kernel, 0.0, atol=1e-12)
    return contacts, forward, centres, middle


def moments(contacts, forward, centres, middle, frames: int) -> Iterator[tuple]:
    """Per frame, the wheels, the points ahead of them and the turning centre at that moment, and the trail
    of the middle of the rear axle so far."""
    for step in np.linspace(0, STEPS - 1, frames).astype(int):
        yield contacts[step], forward[step], centres[step], middle[:step + 1]


# --- plumbing -------------------------------------------------------------------------
def coordinates(mv, points) -> np.ndarray:
    """The coordinates along x and y of points on the ground, for the checks."""
    return np.stack([((plane & points) / (mv.w & points)).to_array() for plane in (mv.x, mv.y)], axis=-1)


if __name__ == "__main__":
    from examples.animation import save_animation
    from examples.mechanics.tricycle import render

    for (name, core), draw in zip(SCENES.items(), (render.plane, render.space)):
        save_animation(render.animate(draw, moments(*run(core), 150)), name, 50)
