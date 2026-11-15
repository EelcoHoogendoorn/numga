"""Three masses turn on a unit sphere beside their mirror reflection."""

from __future__ import annotations

import numpy as np

from examples.orientation.mirror_energy import core
from examples.orientation.mirror_energy.core import Point, Scalar, mv

# Duration of each animation frame in milliseconds.
frame_duration_ms = 40


def rotating_masses() -> tuple[Point, Point, np.ndarray, Scalar]:
    """Three masses (`[3] Point-`) turn on the unit sphere in the `xy` plane alongside their reflection in the `x` plane
    over `81` frames, giving `[81, 2, 3] Point-` positions and velocities and `[4] Scalar+` kinetic energies."""
    points = mv.vector([[1, 0, 0], [0, 0.8, 0.6], [-0.6, 0, -0.8]]).dual()     # [3] Point-
    masses = np.array([1, 1.5, 0.5])
    rate = mv.xy                                                                # [] Rate+
    mirror = mv.x                                                               # [] Plane+
    n_frames = 81
    times = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    positions, velocities = core.motion(points, rate, mirror, times)            # [n_frames, 2, 3] Point-, Point-
    return positions, velocities, masses, core.energies(points, masses, rate, mirror)


def main() -> None:
    import matplotlib.pyplot as plt
    from examples.util.animation import save_animation, save_figure
    from examples.orientation.mirror_energy import render

    positions, velocities, masses, energies = rotating_masses()
    for label, energy in zip(("Energy", "Reflected", "Rebuilt", "Without orientation"), energies):
        print(label)
        print(energy.coefficient_table())
    figure = render.motion_figure(positions, velocities, masses)
    save_figure(figure, "mirror_energy")
    plt.close(figure)
    save_animation(render.motion_animation(positions, velocities, masses), "mirror_energy", frame_duration_ms)

    # checks
    expected_energy = (masses * velocities[:, 0].scalar_norm_squared()).sum(axis=-1) / 2  # [81] Scalar+
    assert ((energies[:3] - expected_energy[0]).abs() < 1e-12).all()
    assert ((energies[3] + expected_energy[0]).abs() < 1e-12).all()
    assert ((expected_energy - expected_energy[0]).abs() < 1e-12).all()
    assert ((positions.scalar_norm_squared() - 1).abs() < 1e-12).all()
    reflected_rate = mv.x >> mv.xy                                              # [] Rate+
    rebuilt_velocities = positions[:, 1].commutator(reflected_rate)             # [81, 3] Point-
    assert ((rebuilt_velocities - velocities[:, 1]).norm() < 1e-12).all()


def pinned() -> None:
    """Saves the repository-tracked animation of the turning masses and their mirror image."""
    from examples.util.animation import pin_animation
    from examples.orientation.mirror_energy import render

    positions, velocities, masses, _ = rotating_masses()
    pin_animation(render.motion_animation(positions, velocities, masses), "mirror_energy", frame_duration_ms)


if __name__ == "__main__":
    main()
