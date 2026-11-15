"""Coverage tests for point-cloud registration in three-dimensional Euclidean space and on the three-sphere."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from examples.geometry.registration import render, scenarios


def point_mismatch(first, second):
    """Squared displacement norm between two normalized point clouds (`[...] Scalar+`)."""
    return (first.normalized() - second.normalized()).dual().scalar_norm_squared()


@pytest.mark.parametrize("core", scenarios.scenes.values(), ids=scenarios.scenes.keys())
def test_correspondence_residual_is_linear_in_the_motor(core):
    truth = scenarios.ground_truth_motor(core)
    source = core.point_cloud(6, scenarios.cloud_spread, np.random.default_rng(1))
    target = truth >> source
    residual = target * core.Motor - core.Motor * source
    assert residual.arity == 1
    evaluated = residual(truth)
    assert ((evaluated.scalar_norm_squared() + evaluated.dual().scalar_norm_squared()) < 1e-12).all()


@pytest.mark.parametrize("core", scenarios.scenes.values(), ids=scenarios.scenes.keys())
def test_exact_correspondences_recover_the_motor(core):
    truth = scenarios.ground_truth_motor(core)
    source = core.point_cloud(8, scenarios.cloud_spread, np.random.default_rng(2))
    target = truth >> source
    estimate = core.fitted_motor(source, target)
    assert estimate.gatype == core.Motor
    assert ((estimate * estimate.reverse() - core.mv.rotor()).scalar_norm_squared() < 1e-12).all()
    assert (point_mismatch(estimate >> source, target) < 1e-12).all()


@pytest.mark.parametrize("core", scenarios.scenes.values(), ids=scenarios.scenes.keys())
def test_noisy_correspondences_and_pure_shift_align(core):
    mv = core.mv
    rng = np.random.default_rng(3)
    source = core.point_cloud(60, scenarios.cloud_spread, rng)
    truth = scenarios.ground_truth_motor(core)
    noisy_target = core.noisy_cloud(truth >> source, 0.02, rng)
    noisy_estimate = core.fitted_motor(source, noisy_target)
    assert point_mismatch(noisy_estimate >> source, truth >> source).mean(axis=0).square_root() < 0.04

    shift_motor = (mv.xw * 0.75 - mv.yw * 0.25 + mv.zw).exp()
    shifted_target = shift_motor >> source
    shift_estimate = core.fitted_motor(source, shifted_target)
    assert (point_mismatch(shift_estimate >> source, shifted_target) < 1e-12).all()


def test_registration_figure_renders():
    core = scenarios.scenes["euclidean"]
    source = core.point_cloud(6, scenarios.cloud_spread, np.random.default_rng(4))
    target = scenarios.ground_truth_motor(core) >> source
    aligned = core.fitted_motor(source, target) >> source
    plt.close(render.registration_figure(source, target, aligned))
