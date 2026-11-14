"""A tangential force drives a turning crank beside its mirror reflection."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.orientation.mechanical_work import core
from examples.orientation.mechanical_work.core import Point, Scalar, mv

# Frame index rendered as the static preview figure.
still_frame = 8
# Duration of each animation frame in milliseconds.
duration_ms = 40


def mirrored_cranks() -> tuple[Point, Point, Point, Point, Scalar]:
    """A crank driven by a tangential force and its reflection across the plane `x = 0` turn over `72` steps with equal
    mechanical power (`[72, 2] Scalar+`)."""
    separation = 1.65
    radius = 0.9
    speed = 1
    force_strength = 1.5
    n_steps = 72
    times = np.linspace(0, 2 * np.pi / speed, n_steps, endpoint=False)
    mirror = mv.x                                                               # [] Plane+
    pivot = mv.point([-separation, 0])                                          # [] Point-
    initial_handle = mv.point([-separation + radius, 0])                        # [] Point-
    initial_force = mv.direction([0, force_strength])                           # [] Point-
    # Two orthogonal planes through the pivot meet along its rotation axis.
    rate = ((mv.x + separation * mv.w) ^ mv.y) * speed                          # [] Rate+

    pivots = stack([pivot, mirror >> pivot])                                    # [2] Point-
    initial_handles = stack([initial_handle, mirror >> initial_handle])         # [2] Point-
    initial_forces = stack([initial_force, mirror >> initial_force])            # [2] Point-
    rates = stack([rate, mirror >> rate])                                       # [2] Rate+
    return pivots, *core.crank_motion(initial_handles, initial_forces, rates, times)


def main() -> None:
    import matplotlib.pyplot as plt
    from examples.util.animation import save_animation, save_figure
    from examples.orientation.mechanical_work import render

    pivots, handles, velocities, forces, powers = mirrored_cranks()
    figure = render.work_figure(pivots, handles[still_frame], velocities[still_frame], forces[still_frame], handles)
    save_figure(figure, "mechanical_work")
    plt.close(figure)
    save_animation(render.work_animation(pivots, handles, velocities, forces), "mechanical_work", duration_ms)

    # checks
    mirror = mv.x                                                               # [] Plane+
    forques = handles & forces                                                  # [72, 2] Forque-
    forque_difference = forques[:, 1] - (mirror >> forques[:, 0])               # [72] Forque-
    assert (forque_difference.norm() + forque_difference.antinorm() < 1e-12).all()
    assert ((velocities[:, 1] - (mirror >> velocities[:, 0])).antinorm() < 1e-12).all()
    assert ((powers[:, 0] - powers[:, 1]).abs() < 1e-12).all()
    assert (((mirror >> powers[:, 0]) - powers[:, 1]).abs() < 1e-12).all()
    assert (powers > 0).all()


if __name__ == "__main__":
    main()
