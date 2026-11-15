"""Tests for polarized cobalt-60 beta emission and parity violation in the Wu experiment."""

import matplotlib.pyplot as plt
import numpy as np

from examples.orientation.wu_experiment import render, scenarios


def test_emission_and_mirror_image():
    spins, directions, rates, _, _ = scenarios.wu_comparison()
    original_rates, mirror_image_rates, mirrored_apparatus_rates = rates          # [n_directions] Scalar+

    # Angular averaging around the meridian leaves total emission equal to the unpolarized unit rate.
    assert ((original_rates.mean() - 1).abs() < 1e-12).all()
    assert ((mirror_image_rates - original_rates[-np.arange(scenarios.n_directions)]).abs() < 1e-12).all()
    assert ((mirrored_apparatus_rates - mirror_image_rates).abs().max() > 1).all()

    figure = render.experiment_figure(spins, directions, rates)
    figure.canvas.draw()
    plt.close(figure)
