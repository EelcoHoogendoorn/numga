"""A reflected crank has the same applied mechanical power."""

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.orientation.mechanical_work import core, render, scenarios
from examples.orientation.mechanical_work.core import mv


def test_reflected_work_agrees_with_rebuilding_the_load():
    handle = mv.point([1, 0])
    force = mv.direction([0, 2])
    rate = mv.xy
    mirror = mv.x
    times = np.linspace(0, 2 * np.pi, 13)
    handles, velocities, forces, powers = core.turn(
        stack([handle, mirror >> handle]),
        stack([force, mirror >> force]),
        stack([rate, mirror >> rate]),
        times,
    )

    # Reflecting each ingredient gives the reflection of the constructed load.
    loads = handles & forces
    np.testing.assert_allclose(loads[:, 1].kernel, (mirror >> loads[:, 0]).kernel, atol=1e-12)
    np.testing.assert_allclose(velocities[:, 1].kernel, (mirror >> velocities[:, 0]).kernel, atol=1e-12)
    np.testing.assert_allclose(powers.kernel, 2, atol=1e-12)
    np.testing.assert_allclose((mirror >> powers[:, 0]).kernel, powers[:, 1].kernel, atol=1e-12)

    pivots, handles, velocities, forces, powers = scenarios.mirrored_cranks()
    np.testing.assert_allclose(powers[:, 0].kernel, powers[:, 1].kernel, atol=1e-12)
    figure = render.draw_work(pivots, handles[0], velocities[0], forces[0], handles)
    figure.canvas.draw()
    plt.close(figure)
