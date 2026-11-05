"""The GENERIC oscillator: the mass released for a step, its energies and its entropy at fixed energy."""
from __future__ import annotations

from examples.thermodynamics.generic import scenarios


def test_release_runs():
    *_, released = scenarios.release(2, 1, 0.04)
    scenarios.energies_of(released)
    scenarios.surface_entropy_of(released)
