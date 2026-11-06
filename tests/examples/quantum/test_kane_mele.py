"""The Kane–Mele flake: its levels, its states nearest zero, and its helical edge states, computed and drawn on a small
flake."""

from __future__ import annotations

from examples.quantum.kane_mele import core, render, scenarios


def test_a_small_flake_is_swept_spread_and_drawn():
    flake = core.flake(8, scenarios.RIM)
    _, _, spins = scenarios.helical(flake, n_frames=1)
    render.draw_levels(scenarios.CASES, *scenarios.sweep(flake, scenarios.CASES), 3 * 3**0.5 * scenarios.SPIN_ORBIT)
    render.draw_states(flake, scenarios.nearest(flake, scenarios.CASES))
    next(render.animate_spin(flake, spins, 64))
