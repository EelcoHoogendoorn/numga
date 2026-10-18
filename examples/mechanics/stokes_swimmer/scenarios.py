"""Scenes for low-Reynolds-number swimming in PGA2D: Purcell's non-reciprocal three-link swimmer
advancing through a viscous fluid beside a reciprocal flapper that returns to where it started."""

from __future__ import annotations

import numpy as np

from examples.mechanics.stokes_swimmer import core

mv = core.mv
# Swimmer geometry: half-length of the centre link, bead offsets along the centre link and each arm,
# and the regularized blob radius.
HINGE_HALF_SPAN = 0.8
CENTRE_OFFSETS = np.array([-0.5, 0.0, 0.5])
ARM_OFFSETS = np.array([0.35, 0.7, 1.05])
BEAD_RADIUS = 0.14
# Stroke amplitude in radians, cycles, steps per cycle, and animation frame stride.
AMPLITUDE = 1.05
CYCLES = 3
STEPS_PER_CYCLE = 60
EVERY = 3
# Sample grid for the fluid velocity field around the swimmer.
GRID_X = np.linspace(-2.8, 2.8, 25)
GRID_Y = np.linspace(-1.9, 1.9, 17)


# --- math -----------------------------------------------------------------------------
def strokes():
    """Purcell's non-reciprocal stroke and a reciprocal in-phase stroke over `CYCLES` periods: hinge
    angles `[steps, swimmers, 2]`, pose motors `[steps + 1, swimmers] Motor`, centre trajectories
    `[steps + 1, swimmers] Point`, world-frame beads `[steps, swimmers, beads] Point`, world-frame
    forces `[steps, swimmers, beads] Force`, sample grid `[grid_y, grid_x] Point`, and fluid velocity
    field `[frames, grid_y, grid_x] Direction` around Purcell's swimmer."""
    steps = CYCLES * STEPS_PER_CYCLE
    dt = 2 * np.pi / STEPS_PER_CYCLE
    phases = np.arange(steps) * dt                                              # [steps]

    # Swimmer 0 traces a circle in hinge space; swimmer 1 retraces an asymmetric line.
    angles = np.stack([
        np.stack([AMPLITUDE * np.cos(phases), AMPLITUDE * np.sin(phases)], axis=-1),
        np.stack([AMPLITUDE * np.cos(phases), -0.5 * AMPLITUDE * np.cos(phases)], axis=-1),
    ], axis=1)                                                                  # [steps, swimmers, 2]
    rates = np.stack([
        np.stack([-AMPLITUDE * np.sin(phases), AMPLITUDE * np.cos(phases)], axis=-1),
        np.stack([-AMPLITUDE * np.sin(phases), 0.5 * AMPLITUDE * np.sin(phases)], axis=-1),
    ], axis=1)                                                                  # [steps, swimmers, 2]

    sites, shape_velocity = core.three_link_shape(HINGE_HALF_SPAN, CENTRE_OFFSETS, ARM_OFFSETS, angles, rates)
    body_rate, forces = core.swim(sites, shape_velocity, BEAD_RADIUS)           # [steps, swimmers] Twist, Force[beads]

    start_motor = (mv.xy * np.array([0.0, 0.0])).exp()                          # [swimmers] Motor
    motors = core.integrate_pose(body_rate, dt, start_motor)                    # [steps + 1, swimmers] Motor
    positions = motors >> core.ORIGIN                                           # [steps + 1, swimmers] Point
    world_sites = motors[:-1, :, None] >> sites                                 # [steps, swimmers, beads] Point
    world_forces = (motors[:-1, :, None] >> forces.batch().dual()).dual()       # [steps, swimmers, beads] Force

    gx, gy = np.meshgrid(GRID_X, GRID_Y, indexing="xy")
    grid = mv.point(np.stack([gx, gy], axis=-1))                               # [grid_y, grid_x] Point
    frame_slice = slice(0, steps, EVERY)
    flow = core.flow_field(
        grid[None],
        world_sites[frame_slice, 0, None, None],
        world_forces[frame_slice, 0, None, None],
        BEAD_RADIUS,
    )                                                                           # [frames, grid_y, grid_x] Direction

    # --- checks
    # At zero separation the regularized Stokeslet is `(2 / BEAD_RADIUS) * Force.dual()`.
    self_mobility = core.stokeslet(0.0 * mv.x + 0.0 * mv.y, BEAD_RADIUS)
    np.testing.assert_allclose((self_mobility - (2.0 / BEAD_RADIUS) * core.Force.dual()).kernel, 0.0, atol=1e-12)
    # Every step is force-free and torque-free: `rigid.adjugate()(forces) == 0`.
    rigid = sites.commutator(core.Twist).field(0)                               # [steps, swimmers] Direction[beads] <- Twist
    net_forque = rigid.adjugate()(forces)                                       # [steps, swimmers] Forque
    np.testing.assert_allclose(net_forque.kernel, 0.0, atol=1e-12)
    # Scallop theorem: after each full cycle the reciprocal swimmer returns to its starting motor,
    # while Purcell's swimmer advances by the same nonzero distance each cycle.
    cycle_motors = motors[STEPS_PER_CYCLE::STEPS_PER_CYCLE]                     # [cycles, swimmers] Motor
    np.testing.assert_allclose((cycle_motors[:, 1] - start_motor[1]).kernel, 0.0, atol=1e-4)
    cycle_centres = positions[STEPS_PER_CYCLE::STEPS_PER_CYCLE]                 # [cycles, swimmers] Point
    purcell_x = (core.separation(cycle_centres[:, 0], core.ORIGIN) | mv.x).to_array()
    step_advances = np.diff(np.concatenate([[0.0], purcell_x]))
    assert np.all(np.abs(step_advances) > 0.04)
    np.testing.assert_allclose(step_advances, step_advances[0], rtol=1e-2)
    return angles, motors, positions, world_sites, world_forces, grid, flow


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.mechanics.stokes_swimmer import render

    angles, motors, positions, world_sites, world_forces, grid, flow = strokes()
    save_figure(
        render.draw_summary(angles, positions, world_sites, world_forces, grid, flow[0]),
        "stokes_swimmer",
    )
    save_animation(
        render.animate_swimmers(positions, world_sites, grid, flow, EVERY),
        "stokes_swimmer",
        60,
    )
