"""PGA motor estimation from one-sided point correspondence equations."""

import matplotlib.pyplot as plt
import numpy as np

from examples.geometry.registration import render, scenarios
from examples.geometry.registration.core import Motor, cloud, fit_motor, jitter, mv
from examples.geometry.registration.render import coordinates


TRUTH = (mv.xw * 0.75 - mv.yw * 0.25 + mv.zw).exp() * (mv.xy * 0.4 - mv.yz * 0.3 + mv.zx * 0.7).exp()


def test_correspondence_residual_is_linear_in_the_motor():
    source = cloud(20, np.random.default_rng(1))
    target = TRUTH >> source
    residual = target * Motor - Motor * source
    assert residual.arity == 1
    np.testing.assert_allclose(residual(TRUTH).kernel, 0, atol=1e-8)
    misfit = (residual.reverse().scalar_product(residual) + residual.dual().reverse().scalar_product(residual.dual())).sum(axis=0)
    assert misfit.kernel.shape == (1, 8, 8)
    np.testing.assert_allclose(misfit.kernel[0], misfit.kernel[0].T, atol=1e-12)


def test_exact_correspondences_recover_the_motor():
    source = cloud(30, np.random.default_rng(2))
    target = TRUTH >> source
    estimate = fit_motor(source, target)
    np.testing.assert_allclose(coordinates(estimate >> source), coordinates(target), atol=1e-10)


def test_noisy_correspondences_recover_the_pose_within_noise():
    rng = np.random.default_rng(3)
    source = cloud(200, rng)
    target = jitter(TRUTH >> source, 0.05, rng)
    estimate = fit_motor(source, target)
    error = coordinates(estimate >> source) - coordinates(TRUTH >> source)
    assert np.linalg.norm(error, axis=-1).mean() < 0.01


def test_estimate_is_a_trusted_unit_motor():
    source = cloud(30, np.random.default_rng(5))
    estimate = fit_motor(source, TRUTH >> source)
    assert estimate.gatype == Motor
    np.testing.assert_allclose((estimate * estimate.reverse()).kernel, mv.rotor().kernel, atol=1e-12)


def test_translation_is_recovered_by_the_same_fit():
    source = cloud(100, np.random.default_rng(6))
    translation = (mv.xw * 0.75 - mv.yw * 0.25 + mv.zw).exp()
    target = translation >> source
    estimate = fit_motor(source, target)
    np.testing.assert_allclose(coordinates(estimate >> source), coordinates(target), atol=1e-10)


def test_sandwich_alignment_is_the_cartesian_optimum():
    """Centering then aligning minimizes the summed squared point distances, so the one-sided
    fit cannot beat it on that measure."""
    _, target, aligned = scenarios.sandwich_alignment()
    _, _, one_sided = scenarios.one_sided_residual()
    cartesian = ((coordinates(aligned) - coordinates(target)) ** 2).sum()
    coefficient = ((coordinates(one_sided) - coordinates(target)) ** 2).sum()
    assert cartesian <= coefficient * (1 + 1e-12)


def test_scenarios_render():
    """Each scenario's geometry renders as a figure."""
    figures = [
        render.draw_registration(*scenarios.sandwich_alignment(), "Centered sandwich alignment"),
        render.draw_registration(*scenarios.one_sided_residual(), "One-sided motor residual"),
    ]
    assert all(isinstance(figure, plt.Figure) for figure in figures)
