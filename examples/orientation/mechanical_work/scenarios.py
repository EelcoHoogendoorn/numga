"""A tangential push drives a crank beside its reflected counterpart."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.orientation.mechanical_work import core
from examples.orientation.mechanical_work.core import Point, Scalar, mv

# The frame drawn as the still figure.
still_frame = 4
# The duration of each animation frame in milliseconds.
duration_ms = 85


def mirrored_cranks() -> tuple[Point, Point, Point, Point, Scalar]:
    """Two reflected cranks turn with the same positive mechanical power."""
    separation = 1.65
    radius = 0.9
    speed = 1
    force_strength = 1.5
    n_steps = 36
    times = np.linspace(0, 2 * np.pi / speed, n_steps, endpoint=False)
    mirror = mv.x                                                               # [] Plane+
    pivot = mv.point([-separation, 0])                                          # [] Point-
    handle = mv.point([-separation + radius, 0])                                # [] Point-
    force = mv.direction([0, force_strength])                                   # [] Direction-
    # These two planes meet at the pivot and generate its rotation.
    rate = ((mv.x + separation * mv.w) ^ mv.y) * speed                          # [] Rate+

    pivots = stack([pivot, mirror >> pivot])                                    # [cranks] Point-
    handles = stack([handle, mirror >> handle])                                 # [cranks] Point-
    forces = stack([force, mirror >> force])                                    # [cranks] Direction-
    rates = stack([rate, mirror >> rate])                                       # [cranks] Rate+
    return pivots, *core.turn(handles, forces, rates, times)


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.orientation.mechanical_work import render

    pivots, handles, velocities, forces, powers = mirrored_cranks()
    save_figure(render.draw_work(pivots, handles[still_frame], velocities[still_frame], forces[still_frame], handles),
                "mechanical_work")
    save_animation(render.animate(pivots, handles, velocities, forces), "mechanical_work", duration_ms)


if __name__ == "__main__":
    main()
