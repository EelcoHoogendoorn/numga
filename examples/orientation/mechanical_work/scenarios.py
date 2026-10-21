"""A tangential push drives a crank beside its reflected counterpart."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.orientation.mechanical_work import core
from examples.orientation.mechanical_work.core import Plane, Point, Rate, Scalar, mv

# The frame drawn as the still figure.
FRAME = 4
# The duration of each animation frame in milliseconds.
DURATION_MS = 85


def mirrored_cranks() -> tuple[Point, Point, Point, Point, Scalar]:
    """Two reflected cranks turn with the same positive mechanical power."""
    separation = 1.65
    radius = 0.9
    speed = 1
    force_strength = 1.5
    steps = 36
    times = np.linspace(0, 2 * np.pi / speed, steps, endpoint=False)
    mirror: Plane = mv.x
    pivot: Point = mv.point([-separation, 0])
    handle: Point = mv.point([-separation + radius, 0])
    force: Point = mv.direction([0, force_strength])
    rate: Rate = ((mv.x + separation * mv.w) ^ mv.y) * speed

    pivots: Point = stack([pivot, mirror >> pivot])
    handles: Point = stack([handle, mirror >> handle])
    forces: Point = stack([force, mirror >> force])
    rates: Rate = stack([rate, mirror >> rate])
    return pivots, *core.turn(handles, forces, rates, times)


def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.orientation.mechanical_work import render

    pivots, handles, velocities, forces, powers = mirrored_cranks()
    save_figure(render.draw_work(pivots, handles[FRAME], velocities[FRAME], forces[FRAME], handles), "mechanical_work")
    save_animation(render.animate(pivots, handles, velocities, forces), "mechanical_work", DURATION_MS)


if __name__ == "__main__":
    main()
