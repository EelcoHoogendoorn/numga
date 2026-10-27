"""Low-Reynolds-number swimming and Stokeslets in PGA2D: every step is free of force and torque, the
reciprocal stroke comes back to where it started while Purcell's advances, and the scene draws."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.fluids.stokes_swimmer import core, render, scenarios


def test_strokes_are_force_free_and_only_the_loop_advances():
    _, sites, _, forces, motors, positions, _, _ = scenarios.strokes()
    rigid = sites.commutator(core.Twist)
    np.testing.assert_allclose(rigid.adjugate()(forces).kernel, 0.0, atol=1e-12)
    every_cycle = slice(scenarios.N_STEPS_PER_CYCLE, None, scenarios.N_STEPS_PER_CYCLE)
    # The tolerance tests the steps per cycle.
    np.testing.assert_allclose((motors[every_cycle, 1] - motors[0, 1]).kernel, 0.0, atol=1e-4)
    advances = np.diff((core.mv.x & (positions[every_cycle, 0] - core.ORIGIN)).to_array(), prepend=0.0)
    assert np.all(np.abs(advances) > 0.04)


def test_the_scene_draws():
    angles, _, _, _, _, positions, world_sites, world_forces = scenarios.strokes()
    grid = scenarios.sample_grid()
    flow = next(scenarios.flows(world_sites, world_forces, grid))
    plt.close(render.draw_summary(angles, positions, world_sites, world_forces, grid, flow))
    assert len(list(render.animate_swimmers(positions[:4], world_sites[:3], grid, iter([flow]), 3))) == 1
