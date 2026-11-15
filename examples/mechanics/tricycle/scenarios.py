"""Planar and spatial tricycle scenes drive the same sinusoidal steering profile in `PGA2D` and `PGA3D` through one instantiated core module."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from numga.algebras import PGA2D, PGA3D
from examples import instantiate

wheelbase = 1.2
track = 0.8
# Front-wheel travel per step divided by wheelbase, and total step count.
step = 0.016
n_steps = 1500
# Mean left-turn steering angle, oscillation amplitude around it, and full cycles across the drive.
bias = 0.35
swing = 0.3
n_swings = 3
# Sampled frame count and per-frame delay in milliseconds.
n_frames = 188
frame_duration_ms = 40
# Identical core module instantiated once for the plane and once for space.
scenes = {
    name: instantiate("examples.mechanics.tricycle.core", ga)
    for name, ga in (("tricycle_plane", PGA2D), ("tricycle_space", PGA3D))
}


# --- math -----------------------------------------------------------------------------
def run(core, n_steps: int):
    """Sinusoidally modulated steering drives `core` for `n_steps` steps, returning world-frame wheel contacts (`[n_steps, 3] Point-`), rolling-heading points (`[n_steps, 3] Point-`), turning centres (`[n_steps] Meet+`), and rear-axle midpoints (`[n_steps] Point-`)."""
    steering = bias + swing * np.sin(2 * np.pi * n_swings * np.arange(n_steps) / n_steps)  # [n_steps]
    rear, front, steer = core.axles(wheelbase, steering)                        # [] Plane+, [n_steps] Plane+, [n_steps] Motor+
    poses = core.drive(rear, front, step)                                       # [n_steps] Motor+
    contacts, forward = core.wheels(wheelbase, track, steer)                    # [3] Point-, [n_steps, 3] Point-
    contacts = poses[:, None] >> contacts                                       # [n_steps, 3] Point-
    forward = poses[:, None] >> forward                                         # [n_steps, 3] Point-
    centres = poses >> (rear ^ front)                                           # [n_steps] Meet+
    middle = poses >> core.ground_point([0, 0])                                 # [n_steps] Point-
    return contacts, forward, centres, middle


def moments(contacts, forward, centres, middle, n_frames: int) -> Iterator[tuple]:
    """Yields `n_frames` evenly spaced snapshots of wheel contacts, forward heading points, turning centre, and the rear-axle trail accumulated from step zero."""
    for index in np.linspace(1, len(contacts) - 1, n_frames).astype(int):
        yield contacts[index], forward[index], centres[index], middle[:index + 1]


# --- plumbing -------------------------------------------------------------------------
def ground_coordinates(points) -> np.ndarray:
    """Pairing `points` with the planes `x = 0` and `y = 0` over their weights gives their Euclidean ground coordinates (`[..., 2]`), infinite where a point lies at infinity."""
    mv = points.context.multivector
    with np.errstate(divide="ignore", invalid="ignore"):
        coordinates = (stack((mv.x, mv.y)) & points[..., None]) / (mv.w & points)[..., None]  # [..., 2] Scalar+
    return coordinates.to_array()


def main() -> None:
    """Saves the planar and spatial tricycle animations to `PLOT_DIR` and checks rolling, front-wheel speed, straight-ahead parallel axles, and planar-spatial track agreement."""
    from examples.util.animation import save_animation
    from examples.mechanics.tricycle import render

    drives = {name: run(core, n_steps) for name, core in scenes.items()}
    for (name, trajectory), draw in zip(drives.items(), (render.plane, render.space)):
        save_animation(render.animate(draw, moments(*trajectory, n_frames)), name, frame_duration_ms)

    # checks
    # No wheel slides sideways: each contact, its forward heading point, and the next contact join to zero up to
    # chord curvature, the front wheel rolls the same distance every step, straight-ahead axles meet at infinity,
    # and the planar and spatial drives trace identical ground tracks.
    for core, (contacts, forward, _, _) in zip(scenes.values(), drives.values()):
        rolling = (contacts[:-1].normalized() & forward[:-1].normalized() & contacts[1:].normalized()).norm()  # [n_steps - 1, 3] Scalar+
        assert bool((rolling < 0.1 * wheelbase * (step * wheelbase) ** 2).all())
        front_chords = (contacts[1:, 2].normalized() & contacts[:-1, 2].normalized()).norm()  # [n_steps - 1] Scalar+
        assert bool(((front_chords - step * wheelbase).abs() < 1e-3 * step * wheelbase).all())
        rear, front, _ = core.axles(wheelbase, np.zeros(1))                     # [] Plane+, [1] Plane+, [1] Motor+
        assert bool(((core.mv.w ^ (rear ^ front)).scalar_norm_squared() < 1e-24).all())
    plane_track, space_track = (ground_coordinates(trajectory[3]) for trajectory in drives.values())
    assert bool((np.abs(plane_track - space_track) < 1e-9).all())


if __name__ == "__main__":
    main()
