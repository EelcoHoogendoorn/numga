"""Quartz returns light through the polarizer it entered by; a Faraday rotator blocks it."""

import matplotlib.pyplot as plt
import numpy as np

from examples.orientation.optical_rotation import render, scenarios


def test_isolator_and_its_mirror_image():
    polarizations, depths = scenarios.isolator()
    entering = polarizations[:, :, 0, 0]
    leaving = polarizations[:, :, 1, -1]
    # Malus's law: the share of the returning light that passes the polarizer it entered through.
    passed = leaving.dot(entering).squared()

    np.testing.assert_allclose(passed.to_array()[0], 1, atol=1e-12)
    np.testing.assert_allclose(passed.to_array()[1], 0, atol=1e-12)

    figure = render.draw_isolator(polarizations, depths)
    figure.canvas.draw()
    plt.close(figure)
