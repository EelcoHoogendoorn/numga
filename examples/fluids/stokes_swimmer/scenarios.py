"""Scenes for low-Reynolds-number swimming in PGA2D: Purcell's non-reciprocal three-link swimmer advancing through a
viscous fluid beside a reciprocal flapper that returns to where it started."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.fluids.stokes_swimmer import core

mv = core.mv
# Swimmer geometry: half-length of the centre link, bead offsets along the centre link and each arm,
# and the regularized blob radius.
hinge_half_span = 0.8
centre_offsets = np.array([-0.5, 0.0, 0.5])
arm_offsets = np.array([0.35, 0.7, 1.05])
bead_radius = 0.14
# Stroke amplitude in radians, cycles, steps per cycle, the steps per frame, and the duration of a frame.
amplitude = 1.05
n_cycles = 3
n_steps_per_cycle = 60
n_steps_per_frame = 2
duration_ms = 40
# Sample grid for the fluid velocity field around the swimmer.
n_grid_columns, n_grid_rows = 81, 55
grid_x = np.linspace(-2.8, 2.8, n_grid_columns)
grid_y = np.linspace(-1.9, 1.9, n_grid_rows)


# --- math -----------------------------------------------------------------------------
def strokes() -> tuple[np.ndarray, core.Point, core.Force, core.Motor, core.Point, core.Point, core.Force]:
    """Purcell's non-reciprocal stroke and a reciprocal in-phase stroke over the cycles: the hinge angles, the beads
    in the body's frame and their forces at every step, the poses and the centres' places, and the beads and their
    forces carried into the world."""
    n_steps = n_cycles * n_steps_per_cycle
    dt = 2 * np.pi / n_steps_per_cycle
    phases = np.arange(n_steps) * dt                                            # [n_steps]

    # Swimmer 0 traces a circle in hinge space; swimmer 1 retraces an asymmetric line.
    angles = np.stack([
        np.stack([amplitude * np.cos(phases), amplitude * np.sin(phases)], axis=-1),
        np.stack([amplitude * np.cos(phases), -0.5 * amplitude * np.cos(phases)], axis=-1),
    ], axis=1)                                                                  # [n_steps, 2, 2]
    rates = np.stack([
        np.stack([-amplitude * np.sin(phases), amplitude * np.cos(phases)], axis=-1),
        np.stack([-amplitude * np.sin(phases), 0.5 * amplitude * np.sin(phases)], axis=-1),
    ], axis=1)                                                                  # [n_steps, 2, 2]

    sites, shape_velocities = core.three_link_shape(hinge_half_span, centre_offsets, arm_offsets, angles, rates)
    body_rate, forces = core.force_free_motion(sites, shape_velocities, bead_radius)   # [n_steps, 2] Twist+, Force+[n_beads]

    start_motor = (mv.xy * np.zeros(2)).exp()                                   # [2] Motor+
    motors = core.poses(body_rate, dt, start_motor)                             # [n_steps + 1, 2] Motor+
    positions = motors >> core.origin                                           # [n_steps + 1, 2] Point-
    # The beads and their forces carried from the body's frame into the world; a force is moved as its
    # complement, a direction, and complemented again.
    world_sites = motors[:-1] >> sites                                          # [n_steps, 2] Point-[n_beads]
    world_forces = (motors[:-1] >> forces.dual()).dual()                        # [n_steps, 2] Force+[n_beads]
    return angles, sites, forces, motors, positions, world_sites, world_forces


def flows(world_sites: core.Point, world_forces: core.Force, grid: core.Point) -> Iterator[core.Direction]:
    """Per frame, both swimmers' flows at the points of the grid."""
    for step in range(0, world_sites.shape[0], n_steps_per_frame):
        yield core.flow_field(grid, world_sites[step, :, None, None], world_forces[step, :, None, None], bead_radius)


# --- plumbing -------------------------------------------------------------------------
def sample_grid() -> core.Point:
    """The points around the swimmer at which the flow is drawn."""
    return mv.point(np.stack(np.meshgrid(grid_x, grid_y, indexing="xy"), axis=-1))   # [n_grid_rows, n_grid_columns] Point-


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.fluids.stokes_swimmer import render

    angles, sites, forces, motors, positions, world_sites, world_forces = strokes()
    grid = sample_grid()
    first_flow = core.flow_field(grid, world_sites[0, :, None, None], world_forces[0, :, None, None], bead_radius)
    save_figure(render.draw_summary(angles, positions, world_sites, grid, first_flow), "stokes_swimmer")
    save_animation(render.animate_swimmers(positions, world_sites, grid, flows(world_sites, world_forces, grid),
                                           n_steps_per_frame), "stokes_swimmer", duration_ms)

    # checks
    # Every step is free of force and torque: `rigid_velocities.adjugate()(forces) == 0`.
    rigid_velocities = sites.commutator(core.Twist)                             # [n_steps, 2] Direction-[n_beads] <- Twist+
    net_forques = rigid_velocities.adjugate()(forces)                           # [n_steps, 2] Forque-
    np.testing.assert_allclose(net_forques.kernel, 0.0, atol=1e-9)
    # Scallop theorem: after each full cycle the reciprocal flapper returns to its starting pose, while Purcell's
    # swimmer advances by the same nonzero distance each cycle.
    cycle_motors = motors[::n_steps_per_cycle]                                  # [n_cycles + 1, 2] Motor+
    np.testing.assert_allclose((cycle_motors[1:, 1] - cycle_motors[0, 1]).kernel, 0.0, atol=1e-4)
    cycle_centres = positions[::n_steps_per_cycle, 0]                           # [n_cycles + 1] Point-
    cycle_advances = mv.x & (cycle_centres[1:] - cycle_centres[:-1])            # [n_cycles] Scalar+
    assert (cycle_advances.abs() > 0.04).all()
    assert ((cycle_advances - cycle_advances[0]).abs() < 1e-3).all()


if __name__ == "__main__":
    main()
