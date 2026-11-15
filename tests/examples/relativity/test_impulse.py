"""Coverage tests verify proper-length preservation, impulse-train worldlines, and figure rendering for relativistic impulses."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from examples.relativity.impulse import render, scenarios
from examples.relativity.impulse.core import Point, Scalar, impulse_train, mv, space, time, velocity_step, worldline_events


def coordinate_velocity(directions: Point) -> Scalar:
    """Spatial displacement per unit observer time gives coordinate velocity (`[...] Scalar+`)."""
    return (mv.x & directions) / (mv.t & directions)                            # [...] Scalar+


def test_velocity_step_preserves_proper_spacing_and_requested_directions():
    """Both incoming and outgoing four-velocities measure unit oriented proper rod length across the kink step."""
    before_rapidity, after_rapidity = 0.6, -0.75
    events, directions = velocity_step(before_rapidity, after_rapidity)         # [2] Point-, [2] Point-

    assert events.shape == directions.shape == (2,)
    assert events[0].scalar_norm_squared() < 1e-8
    assert ((directions.squared() - 1).abs() < 1e-8).all()
    assert ((coordinate_velocity(directions) - np.tanh([before_rapidity, after_rapidity])).abs() < 1e-8).all()
    separation = events[1] - events[0]                                          # [] Point-
    assert separation.squared() < 0
    assert (((directions & separation) - (time & space)).abs() < 1e-8).all()
    assert (((directions[0] + directions[1]) | separation).abs() < 1e-8).all()


@pytest.mark.parametrize("final_velocity, proper_step", [(0.8, 0.1), (-0.8, 0.2)])
def test_impulse_train_forms_continuous_worldlines_with_fixed_proper_spacing(final_velocity: float, proper_step: float):
    """Every segment joins consecutive kinks in chronological order while preserving unit proper rod length."""
    n_steps = 10
    events, directions = impulse_train(np.arctanh(final_velocity), n_steps, proper_step)  # [n_steps, 2] Point-, [n_steps + 1] Point-
    assert events.shape == (n_steps, 2)
    assert directions.shape == (n_steps + 1,)
    velocities = coordinate_velocity(directions)                                # [n_steps + 1] Scalar+
    assert ((velocities[0]).abs() < 1e-8).all()
    assert ((velocities[-1] - final_velocity).abs() < 1e-8).all()

    separation = events[:, 1] - events[:, 0]                                    # [n_steps] Point-
    for adjacent_directions in (directions[:-1], directions[1:]):
        assert (((adjacent_directions & separation) - (time & space)).abs() < 1e-8).all()

    advances = events[1:] - events[:-1]                                         # [n_steps - 1, 2] Point-
    assert ((mv.t & advances) > 0).all()
    assert ((advances[:, 0].norm() - proper_step).abs() < 1e-8).all()
    assert ((advances & directions[1:-1, None]).abs() < 1e-8).all()

    coasting_time = (mv.t & events[-1, 0]) + (mv.t & events[-1, 1]).clip(0.0, np.inf) + 0.5  # [] Scalar+
    ends = worldline_events(coasting_time, events, directions)                  # [2] Point-
    assert (((mv.x & (ends[1] - ends[0])) - np.sqrt(1 - final_velocity**2)).abs() < 1e-8).all()


@pytest.mark.parametrize("final_rapidity", [-1.1, 1.1])
def test_one_impulse_matches_single_velocity_step(final_rapidity: float):
    """A single-step impulse train reproduces `velocity_step` from rest."""
    train, train_directions = impulse_train(final_rapidity, 1, 0.2)             # [1, 2] Point-, [2] Point-
    events, directions = velocity_step(0.0, final_rapidity)                     # [2] Point-, [2] Point-
    assert ((train[0] - events).scalar_norm_squared() < 1e-8).all()
    assert ((coordinate_velocity(train_directions) - coordinate_velocity(directions)).abs() < 1e-8).all()


def test_scenes_render():
    """All three impulse scenario figures render and close without writing files."""
    plt.close(render.impulse_figure(*scenarios.impulse_scene()))
    plt.close(render.ladder_figure(*scenarios.ladder_scene()))
    plt.close(render.spaceships_figure(*scenarios.spaceships_scene()))
