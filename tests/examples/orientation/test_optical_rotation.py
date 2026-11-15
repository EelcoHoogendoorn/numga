"""Tests for round-trip optical rotation and its mirror image."""

import matplotlib.pyplot as plt

from numga import stack
from examples.orientation.optical_rotation import core, render, scenarios


def test_isolator_and_mirror_image():
    fields, depths = scenarios.isolator_comparison()
    exit_fields = fields[:, :, 1, -1]                                             # [2, 2] Field+
    polarizer_axes = stack([scenarios.polarizer_axis,
                            scenarios.mirror >> scenarios.polarizer_axis])        # [2] Direction-
    # Malus's law gives the transmitted fraction returning through each setup's entrance polarizer.
    transmitted_shares = (exit_fields & (polarizer_axes & core.time)).squared()   # [2, 2] Scalar+

    assert ((transmitted_shares[0] - 1).abs() < 1e-12).all()
    assert (transmitted_shares[1].abs() < 1e-12).all()

    figure = render.isolator_figure(fields, depths)
    figure.canvas.draw()
    plt.close(figure)
