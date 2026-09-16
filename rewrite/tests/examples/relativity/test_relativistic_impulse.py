"""Geometric invariants of finite velocity steps and their composition."""

from __future__ import annotations

import numpy as np
import pytest

from examples.relativity.relativistic_impulse import mv, small_impulses, velocity_step


ATOL = 1e-9


def test_velocity_step_preserves_proper_spacing_and_requested_directions():
    """Both unit tangents measure the same oriented proper rod length."""
    before_rapidity, after_rapidity = 0.6, -0.75
    events, directions = velocity_step(before_rapidity, after_rapidity)

    assert events.shape == directions.shape == (2,)
    np.testing.assert_allclose(events[0].kernel, 0.0, atol=ATOL)
    np.testing.assert_allclose(directions.squared().kernel, 1.0, atol=ATOL)
    np.testing.assert_allclose(
        directions.kernel[:, 1] / directions.kernel[:, 0],
        np.tanh([before_rapidity, after_rapidity]), atol=ATOL,
    )
    separation = events[1] - events[0]
    assert separation.squared().kernel.item() < 0.0
    np.testing.assert_allclose(
        (directions.wedge(separation) - mv.tx).kernel, 0.0, atol=ATOL,
    )
    np.testing.assert_allclose(
        ((directions[0] + directions[1]) | separation).kernel,
        0.0, atol=ATOL,
    )


@pytest.mark.parametrize("final_velocity, dt", [(0.8, 0.1), (-0.8, 0.2)])
def test_impulse_train_forms_continuous_worldlines_with_fixed_proper_spacing(
    final_velocity, dt,
):
    """Every segment joins its kinks and every step preserves the rod length."""
    events, velocities = small_impulses(
        np.arctanh(final_velocity), count=10, dt=dt,
    )
    assert events.shape == (10, 2)
    assert velocities.shape == (11,)
    np.testing.assert_allclose(velocities[[0, -1]], [0.0, final_velocity], atol=ATOL)

    coefficients = np.column_stack([np.ones_like(velocities), velocities])
    directions = mv.vector(coefficients / np.sqrt(1 - velocities**2)[:, None])
    separation = events[:, 1] - events[:, 0]
    for adjacent_directions in (directions[:-1], directions[1:]):
        np.testing.assert_allclose(
            (adjacent_directions.wedge(separation) - mv.tx).kernel,
            0.0, atol=ATOL,
        )

    advances = events[1:] - events[:-1]
    assert np.all(advances.kernel[:, :, 0] > 0.0)
    np.testing.assert_allclose(advances[:, 0].norm().kernel, dt, atol=ATOL)
    np.testing.assert_allclose(
        events[-1, 0].kernel[0],
        (dt / np.sqrt(1 - velocities[1:-1]**2)).sum(), atol=ATOL,
    )
    np.testing.assert_allclose(
        advances.wedge(directions[1:-1, None]).kernel, 0.0, atol=ATOL,
    )

    # Compare the ends at one common observer time after their final kinks.
    coasting_time = events[-1].kernel[:, 0].max() + 0.5
    final_positions = events[-1].kernel[:, 1] + final_velocity * (
        coasting_time - events[-1].kernel[:, 0]
    )
    np.testing.assert_allclose(
        final_positions[1] - final_positions[0],
        np.sqrt(1 - final_velocity**2), atol=ATOL,
    )


@pytest.mark.parametrize("final_rapidity", [-1.1, 1.1])
def test_one_impulse_matches_a_single_velocity_step(final_rapidity):
    train, velocities = small_impulses(final_rapidity, count=1, dt=0.2)
    events, directions = velocity_step(0.0, final_rapidity)
    np.testing.assert_allclose(train[0].kernel, events.kernel, atol=ATOL)
    np.testing.assert_allclose(
        velocities, directions.kernel[:, 1] / directions.kernel[:, 0], atol=ATOL,
    )
