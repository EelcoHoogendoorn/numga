"""Geometric invariants of finite velocity steps and their composition, and the three figures."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from examples.relativity.impulse import render, scenarios
from examples.relativity.impulse.core import mv, small_impulses, velocity_step, worldline_events


ATOL = 1e-9


def velocity(directions):
    return (-(directions | mv.x) / (directions | mv.t)).to_array()


def test_mathematics_does_not_import_plotting():
    """The math layer must stay free of the plotting stack, transitively."""
    probe = (
        "import examples.relativity.impulse.core as c, sys; "
        "bad = [m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'PIL')]; "
        "print(bad)"
    )
    repo_rewrite = Path(__file__).resolve().parents[3]
    env = {**os.environ, "PYTHONPATH": f"{repo_rewrite / 'src'}:{repo_rewrite}"}
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, env=env
    )
    assert out.stdout.strip() == "[]", f"plotting reached the math layer: {out.stdout}"


def test_velocity_step_preserves_proper_spacing_and_requested_directions():
    """Both unit tangents measure the same oriented proper rod length."""
    before_rapidity, after_rapidity = 0.6, -0.75
    events, directions = velocity_step(before_rapidity, after_rapidity)

    assert events.shape == directions.shape == (2,)
    np.testing.assert_allclose(events[0].kernel, 0.0, atol=ATOL)
    np.testing.assert_allclose(directions.squared().kernel, 1.0, atol=ATOL)
    np.testing.assert_allclose(velocity(directions), np.tanh([before_rapidity, after_rapidity]), atol=ATOL)
    separation = events[1] - events[0]
    assert separation.squared().kernel.item() < 0.0
    np.testing.assert_allclose((directions.wedge(separation) - mv.tx).kernel, 0.0, atol=ATOL)
    np.testing.assert_allclose(((directions[0] + directions[1]) | separation).kernel, 0.0, atol=ATOL)


@pytest.mark.parametrize("final_velocity, dt", [(0.8, 0.1), (-0.8, 0.2)])
def test_impulse_train_forms_continuous_worldlines_with_fixed_proper_spacing(final_velocity, dt):
    """Every segment joins its kinks and every step preserves the rod length."""
    events, directions = small_impulses(np.arctanh(final_velocity), 10, dt)
    assert events.shape == (10, 2)
    assert directions.shape == (11,)
    np.testing.assert_allclose(velocity(directions)[[0, -1]], [0.0, final_velocity], atol=ATOL)

    separation = events[:, 1] - events[:, 0]
    for adjacent_directions in (directions[:-1], directions[1:]):
        np.testing.assert_allclose((adjacent_directions.wedge(separation) - mv.tx).kernel, 0.0, atol=ATOL)

    advances = events[1:] - events[:-1]
    assert np.all((advances | mv.t).to_array() > 0.0)
    np.testing.assert_allclose(advances[:, 0].norm().kernel, dt, atol=ATOL)
    np.testing.assert_allclose(advances.wedge(directions[1:-1, None]).kernel, 0.0, atol=ATOL)

    # Compare the ends at one common observer time after their final kinks.
    coasting_time = (events[-1] | mv.t).to_array().max() + 0.5
    ends = worldline_events(mv.scalar([coasting_time]), events, directions)
    np.testing.assert_allclose((-((ends[1] - ends[0]) | mv.x)).to_array(), np.sqrt(1 - final_velocity**2), atol=ATOL)


@pytest.mark.parametrize("final_rapidity", [-1.1, 1.1])
def test_one_impulse_matches_a_single_velocity_step(final_rapidity):
    train, train_directions = small_impulses(final_rapidity, 1, 0.2)
    events, directions = velocity_step(0.0, final_rapidity)
    np.testing.assert_allclose(train[0].kernel, events.kernel, atol=ATOL)
    np.testing.assert_allclose(velocity(train_directions), velocity(directions), atol=ATOL)


@pytest.mark.parametrize("name", ["impulse", "ladder", "spaceships"])
def test_figures_draw(name):
    """Each scenario runs its checks and draws a figure."""
    geometry = getattr(scenarios, name)()
    assert isinstance(getattr(render, f"draw_{name}")(*geometry), plt.Figure)
