"""Motor estimation from point correspondences in flat space and on the three-sphere."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from examples.geometry.registration import render, scenarios
from examples.geometry.registration.render import coordinates


SPREAD = scenarios.CLOUD_SPREAD


@pytest.mark.parametrize("core", scenarios.SCENES.values(), ids=scenarios.SCENES.keys())
def test_correspondence_residual_is_linear_in_the_motor(core):
    truth = scenarios.true_motor(core)
    source = core.cloud(20, SPREAD, np.random.default_rng(1))
    target = truth >> source
    residual = target * core.Motor - core.Motor * source
    assert residual.arity == 1
    np.testing.assert_allclose(residual(truth).kernel, 0, atol=1e-8)
    misfit = (residual.scalar_norm_squared() + residual.dual().scalar_norm_squared()).sum(axis=0)
    assert misfit.kernel.shape == (1, 8, 8)
    np.testing.assert_allclose(misfit.kernel[0], misfit.kernel[0].T, atol=1e-12)


@pytest.mark.parametrize("core", scenarios.SCENES.values(), ids=scenarios.SCENES.keys())
def test_exact_correspondences_recover_the_motor(core):
    truth = scenarios.true_motor(core)
    source = core.cloud(30, SPREAD, np.random.default_rng(2))
    target = truth >> source
    estimate = core.fit_motor(source, target)
    np.testing.assert_allclose(coordinates(estimate >> source), coordinates(target), atol=1e-10)


@pytest.mark.parametrize("core", scenarios.SCENES.values(), ids=scenarios.SCENES.keys())
def test_noisy_correspondences_recover_the_pose_within_noise(core):
    truth = scenarios.true_motor(core)
    rng = np.random.default_rng(3)
    source = core.cloud(200, SPREAD, rng)
    target = core.jitter(truth >> source, 0.05, rng)
    estimate = core.fit_motor(source, target)
    error = coordinates(estimate >> source) - coordinates(truth >> source)
    assert np.linalg.norm(error, axis=-1).mean() < 0.03


@pytest.mark.parametrize("core", scenarios.SCENES.values(), ids=scenarios.SCENES.keys())
def test_estimate_is_a_trusted_unit_motor(core):
    truth = scenarios.true_motor(core)
    source = core.cloud(30, SPREAD, np.random.default_rng(5))
    estimate = core.fit_motor(source, truth >> source)
    assert estimate.gatype == core.Motor
    np.testing.assert_allclose((estimate * estimate.reverse()).kernel, core.mv.rotor().kernel, atol=1e-12)


@pytest.mark.parametrize("core", scenarios.SCENES.values(), ids=scenarios.SCENES.keys())
def test_translation_is_recovered_by_the_same_fit(core):
    mv = core.mv
    source = core.cloud(100, SPREAD, np.random.default_rng(6))
    translation = (mv.xw * 0.75 - mv.yw * 0.25 + mv.zw).exp()
    target = translation >> source
    estimate = core.fit_motor(source, target)
    np.testing.assert_allclose(coordinates(estimate >> source), coordinates(target), atol=1e-10)


def test_scenarios_render():
    """The registration scenario renders as a figure."""
    source, target, aligned = scenarios.registration(scenarios.SCENES["euclidean"])
    figure = render.draw_registration(source, target, aligned)
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
