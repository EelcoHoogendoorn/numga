"""Quartz returns light through the polarizer it entered by; a Faraday rotator blocks it."""

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.orientation.optical_rotation import core, render, scenarios


def test_isolator_and_its_mirror_image():
    fields, depths = scenarios.isolator()
    leaving = fields[:, :, 1, -1]                                                  # [media, setups] Field+
    # The polarizer each setup's light entered through, and its mirror image.
    axes = stack([scenarios.polarizer_axis, scenarios.mirror >> scenarios.polarizer_axis])   # [setups] Direction-
    # Malus's law: the share of the returning light that passes the polarizer it entered through.
    passed = (leaving & (axes & core.time)).squared()                              # [media, setups] Scalar+

    np.testing.assert_allclose(passed.to_array()[0], 1, atol=1e-12)
    np.testing.assert_allclose(passed.to_array()[1], 0, atol=1e-12)

    figure = render.draw_isolator(fields, depths)
    figure.canvas.draw()
    plt.close(figure)
