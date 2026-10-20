"""Scenes for low-Reynolds-number swimming in PGA2D: Purcell's non-reciprocal three-link swimmer
advancing through a viscous fluid beside a reciprocal flapper that returns to where it started."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.fluids.stokes_swimmer import core

mv = core.mv
# Swimmer geometry: half-length of the centre link, bead offsets along the centre link and each arm,
# and the regularized blob radius.
HINGE_HALF_SPAN = 0.8
CENTRE_OFFSETS = np.array([-0.5, 0.0, 0.5])
ARM_OFFSETS = np.array([0.35, 0.7, 1.05])
BEAD_RADIUS = 0.14
# Stroke amplitude in radians, cycles, steps per cycle, the steps per frame, and the duration of a frame.
AMPLITUDE = 1.05
N_CYCLES = 3
N_STEPS_PER_CYCLE = 60
N_STEPS_PER_FRAME = 3
DURATION_MS = 60
# Sample grid for the fluid velocity field around the swimmer.
GRID_X = np.linspace(-2.8, 2.8, 25)
GRID_Y = np.linspace(-1.9, 1.9, 17)


# --- math -----------------------------------------------------------------------------
def strokes() -> tuple[np.ndarray, core.Point, core.Twist, core.Force, core.Motor, core.Point, core.Point, core.Force]:
    """Purcell's non-reciprocal stroke and a reciprocal in-phase stroke over `N_CYCLES` periods: hinge angles
    `[steps, swimmers, 2]`, the beads in the body's frame and the body's rate and the beads' forces at every
    step, pose motors `[steps + 1, swimmers] Motor`, centre trajectories `[steps + 1, swimmers] Point`, and
    world-frame beads and forces `[steps, swimmers] Point[beads]`, `Force[beads]`."""
    n_steps = N_CYCLES * N_STEPS_PER_CYCLE
    dt = 2 * np.pi / N_STEPS_PER_CYCLE
    phases = np.arange(n_steps) * dt                                            # [steps]

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
    world_sites = motors[:-1] >> sites                                          # [steps, swimmers] Point[beads]
    world_forces = (motors[:-1] >> forces.dual()).dual()                        # [steps, swimmers] Force[beads]
    return angles, sites, body_rate, forces, motors, positions, world_sites, world_forces


def flows(world_sites: core.Point, world_forces: core.Force, grid: core.Point) -> Iterator[core.Direction]:
    """Per frame, the flow Purcell's swimmer drives at the points of the grid `[grid_y, grid_x] Direction`."""
    for step in range(0, world_sites.shape[0], N_STEPS_PER_FRAME):
        yield core.flow_field(grid, world_sites[step, 0, None, None], world_forces[step, 0, None, None], BEAD_RADIUS)


# --- plumbing -------------------------------------------------------------------------
def sample_grid() -> core.Point:
    """The points around the swimmer at which the flow is drawn `[grid_y, grid_x] Point`."""
    gx, gy = np.meshgrid(GRID_X, GRID_Y, indexing="xy")
    return mv.point(np.stack([gx, gy], axis=-1))


def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.fluids.stokes_swimmer import render

    angles, sites, _, forces, motors, positions, world_sites, world_forces = strokes()
    grid = sample_grid()
    save_figure(render.draw_summary(angles, positions, world_sites, world_forces, grid,
                                    next(flows(world_sites, world_forces, grid))), "stokes_swimmer")
    save_animation(render.animate_swimmers(positions, world_sites, grid, flows(world_sites, world_forces, grid),
                                           N_STEPS_PER_FRAME), "stokes_swimmer", DURATION_MS)

    # checks
    # At zero separation the regularized Stokeslet is `(2 / BEAD_RADIUS) * Force.dual()`.
    self_mobility = core.stokeslet(0.0 * mv.x + 0.0 * mv.y, BEAD_RADIUS)
    np.testing.assert_allclose((self_mobility - (2.0 / BEAD_RADIUS) * core.Force.dual()).kernel, 0.0, atol=1e-12)
    # Every step is force-free and torque-free: `rigid.adjugate()(forces) == 0`.
    rigid = sites.commutator(core.Twist)                                        # [steps, swimmers] Direction[beads] <- Twist
    net_forque = rigid.adjugate()(forces)                                       # [steps, swimmers] Forque
    np.testing.assert_allclose(net_forque.kernel, 0.0, atol=1e-12)
    # Scallop theorem: after each full cycle the reciprocal swimmer returns to its starting motor,
    # while Purcell's swimmer advances by the same nonzero distance each cycle.
    cycle_motors = motors[N_STEPS_PER_CYCLE::N_STEPS_PER_CYCLE]                 # [cycles, swimmers] Motor
    np.testing.assert_allclose((cycle_motors[:, 1] - motors[0, 1]).kernel, 0.0, atol=1e-4)
    cycle_centres = positions[N_STEPS_PER_CYCLE::N_STEPS_PER_CYCLE]             # [cycles, swimmers] Point
    purcell_x = (core.separation(cycle_centres[:, 0], core.ORIGIN) | mv.x).to_array()
    step_advances = np.diff(np.concatenate([[0.0], purcell_x]))
    assert np.all(np.abs(step_advances) > 0.04)
    np.testing.assert_allclose(step_advances, step_advances[0], rtol=1e-2)


if __name__ == "__main__":
    main()
