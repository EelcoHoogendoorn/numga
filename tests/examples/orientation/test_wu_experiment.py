"""Polarization moves electrons from one side to the other without changing how many; the mirrored
apparatus does not reproduce the mirror image."""

import matplotlib.pyplot as plt
import numpy as np

from examples.orientation.wu_experiment import render, scenarios


def test_emission_and_its_mirror_image():
    spins, directions, rates = scenarios.experiment()
    original, image, built = rates.to_array()

    # Around the circle the rate averages to that of unpolarized nuclei; the closing sample repeats the first.
    np.testing.assert_allclose(original[:-1].mean(), 1, atol=1e-12)
    np.testing.assert_allclose(image, original[::-1], atol=1e-12)
    assert np.abs(built - image).max() > 1

    figure = render.draw_experiment(spins, directions, rates)
    figure.canvas.draw()
    plt.close(figure)
