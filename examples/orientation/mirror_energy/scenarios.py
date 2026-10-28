"""Three masses turn on a sphere beside their reflected motion."""

from __future__ import annotations

import numpy as np

from examples.orientation.mirror_energy import core
from examples.orientation.mirror_energy.core import Point, Scalar, mv

# The duration of each animation frame in milliseconds.
frame_duration_ms = 90


def rotating_masses() -> tuple[Point, Point, np.ndarray, Scalar]:
    """Three masses on the unit sphere turning in the plane `mv.xy`, and their mirror image in the plane `mv.x`: their
    positions and velocities over one turn, the masses, and the kinetic energies of the turning body, reflected, rebuilt
    and sandwiched."""
    points = mv.point([[1, 0, 0], [0, 0.8, 0.6], [-0.6, 0, -0.8]])             # [masses] Point-
    masses = np.array([1, 1.5, 0.5])
    rate = mv.xy                                                                # [] Rate+
    mirror = mv.x                                                               # [] Plane+
    n_frames = 36
    times = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    positions, velocities = core.motion(points, rate, mirror, times)
    return positions, velocities, masses, core.energies(points, masses, rate, mirror)


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.orientation.mirror_energy import render

    positions, velocities, masses, energies = rotating_masses()
    for label, energy in zip(("Energy", "Reflected", "Rebuilt", "Without orientation"), energies):
        print(label)
        print(energy.coefficient_table())
    save_figure(render.draw_motion(positions, velocities, masses), "mirror_energy")
    save_animation(render.animate_motion(positions, velocities, masses), "mirror_energy", frame_duration_ms)


if __name__ == "__main__":
    main()
