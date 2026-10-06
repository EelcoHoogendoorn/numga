"""The Hopfion solves vacuum Maxwell equations and carries energy along the twistor rays."""

import numpy as np

from examples.relativity.twistors import core


def test_hopfion_is_null_and_its_energy_follows_the_robinson_congruence():
    rng = np.random.default_rng(41)
    events = core.mv(core.Event, rng.uniform(-2, 2, (128, 4)))
    field = core.hopfion(events)
    electric = field | core.mv.t
    magnetic = (core.mv.xyzt * field) | core.mv.t
    energy = (electric.squared() + magnetic.squared()) / 2
    flow = -core.SPATIAL_VOLUME * (electric ^ magnetic)
    direction = core.robinson(events)

    np.testing.assert_allclose(field.squared().kernel, 0, atol=1e-12)
    np.testing.assert_allclose((flow / energy + core.mv.t - direction).kernel, 0, atol=1e-12)
    np.testing.assert_allclose((field | direction).kernel, 0, atol=1e-12)
    assert np.all(energy.kernel > 0)


def test_hopfion_satisfies_vacuum_maxwell_equations():
    rng = np.random.default_rng(73)
    events = core.mv(core.Event, rng.uniform(-2, 2, (128, 4)))
    steps = core.mv(core.Event, np.eye(4))
    reciprocal = steps / steps.scalar_norm_squared()
    step_size = 1e-4
    centers = events[:, None]
    offsets = steps[None, :] * step_size

    # Differentiate the complete field independently in all four spacetime directions.
    gradient = (
        -core.hopfion(centers + 2 * offsets)
        + 8 * core.hopfion(centers + offsets)
        - 8 * core.hopfion(centers - offsets)
        + core.hopfion(centers - 2 * offsets)
    ) / (12 * step_size)
    maxwell = (reciprocal * gradient).sum(axis=1)

    np.testing.assert_allclose(maxwell.kernel, 0, atol=1e-9)
