"""Energy agrees with speed and with rebuilding the reflected mass distribution."""

import matplotlib.pyplot as plt
import numpy as np

from examples.orientation.mirror_energy import core, render, scenarios


def test_reflected_energy_and_spherical_motion():
    positions, velocities, masses, energies = scenarios.rotating_masses()
    expected = (masses * velocities.scalar_norm_squared()).sum(axis=-1) / 2

    np.testing.assert_allclose(energies.kernel[:3], 1.07, atol=1e-12)
    np.testing.assert_allclose(energies.kernel[3], -1.07, atol=1e-12)
    np.testing.assert_allclose(expected.kernel, 1.07, atol=1e-12)
    np.testing.assert_allclose(positions.scalar_norm_squared().kernel, 1, atol=1e-12)

    # The reflected points obey the reflected rate at every sampled instant.
    reflected_rate = core.mv.x >> core.mv.xy
    rebuilt_velocities = positions[:, 1].commutator(reflected_rate)
    np.testing.assert_allclose((rebuilt_velocities - velocities[:, 1]).kernel, 0, atol=1e-12)

    figure = render.draw_motion(positions, velocities, masses)
    figure.canvas.draw()
    plt.close(figure)
