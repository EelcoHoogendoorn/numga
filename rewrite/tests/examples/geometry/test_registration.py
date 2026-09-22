"""PGA motor estimation from one-sided point correspondence equations."""

import numpy as np

from examples.geometry.registration import Motor, cloud, coordinates, fit_motor, jitter, mv


TRUTH = (mv.xw * 0.75 - mv.yw * 0.25 + mv.zw).exp() * (mv.xy * 0.4 - mv.yz * 0.3 + mv.zx * 0.7).exp()


def test_correspondence_residual_is_linear_in_the_motor():
    source = cloud(20, np.random.default_rng(1))
    target = TRUTH >> source
    residual = target * Motor - Motor * source
    assert residual.arity == 1
    np.testing.assert_allclose(residual(TRUTH).kernel, 0, atol=1e-10)
    misfit = residual.transpose()(residual).sum(axis=0)
    assert misfit.kernel.shape == (8, 8)
    np.testing.assert_allclose(misfit.kernel, misfit.kernel.T, atol=1e-12)


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


def test_tutorial_runs_and_saves(tmp_path, monkeypatch):
    from examples.geometry import registration
    monkeypatch.setattr(registration, "PLOT_DIR", tmp_path)
    registration.sandwich_alignment()
    registration.one_sided_residual()
    assert (tmp_path / "registration_sandwich_alignment.png").exists()
    assert (tmp_path / "registration_one_sided_residual.png").exists()
