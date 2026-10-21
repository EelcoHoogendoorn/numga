"""Three masses turn on a sphere beside their reflected motion."""

from __future__ import annotations

import numpy as np

from examples.orientation.mirror_energy import core
from examples.orientation.mirror_energy.core import Plane, Point, Rate, Scalar, mv

# The duration of each animation frame in milliseconds.
DURATION_MS = 90


def rotating_masses() -> tuple[Point, Point, np.ndarray, Scalar]:
    """A prescribed turn and its mirror have the same kinetic energy."""
    points: Point = mv.antivector([[1, 0, 0], [0, 0.8, 0.6], [-0.6, 0, -0.8]])
    masses = np.array([1, 1.5, 0.5])
    rate: Rate = mv.xy
    mirror: Plane = mv.x
    frames = 36
    times = np.linspace(0, 2 * np.pi, frames, endpoint=False)
    positions, velocities = core.motion(points, rate, mirror, times)
    return positions, velocities, masses, core.energies(points, masses, rate, mirror)


def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.orientation.mirror_energy import render

    positions, velocities, masses, energies = rotating_masses()
    for label, energy in zip(("Energy", "Reflected", "Rebuilt", "Without orientation"), energies):
        print(label)
        print(energy.coefficient_table())
    save_figure(render.draw_motion(positions, velocities, masses), "mirror_energy")
    save_animation(render.animate_motion(positions, velocities, masses), "mirror_energy", DURATION_MS)


if __name__ == "__main__":
    main()
