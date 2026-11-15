"""Step the GENERIC damped oscillator once and render its first frame."""

from __future__ import annotations

from examples.thermodynamics.generic import render, scenarios


def test_release_runs():
    states = list(scenarios.release(2, 1, 0.04))
    next(render.damping(scenarios.energies_of, scenarios.surface_entropy_of, 0.04, iter(states)))
