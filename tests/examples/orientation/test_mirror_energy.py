"""Kinetic energy agrees with speed and with rebuilding the reflected mass distribution."""

import matplotlib.pyplot as plt

from examples.orientation.mirror_energy import core, render, scenarios


def test_reflected_energy_and_spherical_motion():
    positions, velocities, masses, energies = scenarios.rotating_masses()
    expected = (masses * velocities.scalar_norm_squared()).sum(axis=-1) / 2

    assert ((energies[:3] - 1.07).abs() < 1e-12).all()
    assert ((energies[3] + 1.07).abs() < 1e-12).all()
    assert ((expected - 1.07).abs() < 1e-12).all()
    assert ((positions.scalar_norm_squared() - 1).abs() < 1e-12).all()

    # The reflected points obey the reflected turning rate at every sampled instant.
    reflected_rate = core.mv.x >> core.mv.xy
    rebuilt_velocities = positions[:, 1].commutator(reflected_rate)
    assert ((rebuilt_velocities - velocities[:, 1]).norm() < 1e-12).all()

    figure = render.motion_figure(positions, velocities, masses)
    figure.canvas.draw()
    plt.close(figure)
    frame = next(render.motion_animation(positions[:3], velocities[:3], masses))
    assert frame.ndim == 3 and frame.shape[1:] == (600, 3)
