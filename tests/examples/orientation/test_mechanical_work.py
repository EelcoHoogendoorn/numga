"""A reflected crank has the same applied mechanical power as the original crank."""

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.orientation.mechanical_work import core, render, scenarios
from examples.orientation.mechanical_work.core import mv


def test_reflected_work_agrees_with_rebuilding_the_forque():
    handle = mv.point([1, 0])
    force = mv.direction([0, 2])
    rate = mv.xy
    mirror = mv.x
    times = np.linspace(0, 2 * np.pi, 5, endpoint=False)
    handles, velocities, forces, powers = core.crank_motion(
        stack([handle, mirror >> handle]),
        stack([force, mirror >> force]),
        stack([rate, mirror >> rate]),
        times,
    )

    # Reflecting each ingredient gives the reflection of the constructed forque.
    forques = handles & forces
    forque_difference = forques[:, 1] - (mirror >> forques[:, 0])
    assert (forque_difference.norm() + forque_difference.antinorm() < 1e-12).all()
    assert ((velocities[:, 1] - (mirror >> velocities[:, 0])).antinorm() < 1e-12).all()
    assert ((powers - 2).abs() < 1e-12).all()
    assert (((mirror >> powers[:, 0]) - powers[:, 1]).abs() < 1e-12).all()

    pivots, handles, velocities, forces, powers = scenarios.mirrored_cranks()
    assert ((powers[:, 0] - powers[:, 1]).abs() < 1e-12).all()
    figure = render.work_figure(pivots, handles[0], velocities[0], forces[0], handles)
    figure.canvas.draw()
    plt.close(figure)
    frame = next(render.work_animation(pivots, handles[:3], velocities[:3], forces[:3]))
    assert frame.ndim == 3 and frame.shape[2] == 3
