"""Low-Reynolds-number swimming and Stokeslets in PGA2D: every step is free of force and torque, the
reciprocal stroke comes back to where it started while Purcell's advances, and the first frame of the flow draws."""

from __future__ import annotations

import numpy as np

from examples.fluids.stokes_swimmer import core, render, scenarios


def test_strokes_are_force_free_and_only_the_loop_advances():
    _, sites, forces, motors, positions, _, _ = scenarios.strokes()
    rigid_velocities = sites.commutator(core.Twist)
    np.testing.assert_allclose(rigid_velocities.adjugate()(forces).kernel, 0.0, atol=1e-9)
    every_cycle = slice(None, None, scenarios.n_steps_per_cycle)
    # The tolerance tests the steps per cycle.
    cycle_motors = motors[every_cycle, 1]
    np.testing.assert_allclose((cycle_motors[1:] - cycle_motors[0]).kernel, 0.0, atol=1e-4)
    cycle_centres = positions[every_cycle, 0]
    assert ((core.mv.x & (cycle_centres[1:] - cycle_centres[:-1])).abs() > 0.04).all()


def test_the_first_frame_of_the_flow_draws():
    _, _, _, _, positions, world_sites, world_forces = scenarios.strokes()
    grid = scenarios.sample_grid()
    flow = next(scenarios.flows(world_sites, world_forces, grid))
    next(render.animate_swimmers(positions[:2], world_sites[:1], grid, [flow], 1))
